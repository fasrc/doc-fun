"""Document standardization core class."""

import os
import yaml
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging

from ..extractors import ContentExtractor, ExtractedContent, HTMLContentExtractor
from ..providers.manager import ProviderManager
from ..providers.base import CompletionRequest, CompletionResponse
from ..plugin_manager import PluginManager
from ..error_handler import ErrorHandler
from ..exceptions import DocumentStandardizerError


class DocumentStandardizer:
    """Core class for standardizing documents to organizational standards.
    
    Takes non-conforming documentation and rewrites it to match prompt standards
    while preserving original knowledge and information.
    """
    
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, 
                 temperature: float = 0.3, output_format: str = 'html'):
        """Initialize DocumentStandardizer.
        
        Args:
            provider: LLM provider to use ('openai' or 'claude')
            model: Specific model to use
            temperature: Temperature for generation (0.0-1.0)
            output_format: Output format ('html' or 'markdown')
        """
        self.logger = logging.getLogger(__name__)
        self.error_handler = ErrorHandler()
        
        # Configuration
        self.provider_name = provider
        self.model = model
        self.temperature = temperature
        self.output_format = output_format.lower()
        
        # Paths
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.prompts_dir = self.project_root / "prompts"
        self.terminology_path = self.project_root / "terminology.yaml"
        
        # Initialize components
        self.provider_manager = ProviderManager()
        self.plugin_manager = PluginManager()
        
        # Load configuration
        self.prompt_config = self._load_standardization_prompts()
        self.terminology = self._load_terminology()
        
        # Setup provider
        self.client = self._setup_provider()
        
        # Initialize extractors
        self.extractors = self._initialize_extractors()
    
    def _setup_provider(self) -> Any:
        """Set up the LLM provider."""
        try:
            if self.provider_name:
                provider = self.provider_manager.get_provider(self.provider_name)
            else:
                # Use default provider
                provider_name = self.provider_manager.get_default_provider()
                if provider_name:
                    provider = self.provider_manager.get_provider(provider_name)
                    self.provider_name = provider_name
                else:
                    raise DocumentStandardizerError("No available providers found")
            
            if not provider:
                raise DocumentStandardizerError(f"Provider '{self.provider_name}' not found")
            
            # Set default model if not provided
            if not self.model:
                self.model = self.provider_manager.get_default_model()
                if not self.model:
                    raise DocumentStandardizerError("No default model available")
            
            # Validate model/provider combination
            is_valid, error_msg = self.provider_manager.validate_model_provider_combination(
                self.model, provider.get_provider_name()
            )
            if not is_valid:
                raise DocumentStandardizerError(error_msg)
            
            self.logger.info(f"Using provider: {provider.get_provider_name()}, model: {self.model}")
            return provider
            
        except Exception as e:
            # Log error and re-raise
            self.logger.error(f"Failed to setup provider: {e}")
            raise
    
    def _load_standardization_prompts(self) -> Dict[str, Any]:
        """Load standardization prompt configuration."""
        prompt_file = self.prompts_dir / "standardization" / f"{self.output_format}.yaml"
        
        # Fallback to default if format-specific doesn't exist
        if not prompt_file.exists():
            prompt_file = self.prompts_dir / "standardization" / "default.yaml"
        
        try:
            if prompt_file.exists():
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                self.logger.warning(f"Prompt file not found: {prompt_file}")
                return self._get_default_standardization_prompts()
                
        except Exception as e:
            self.logger.error(f"Error loading prompts from {prompt_file}: {e}")
            return self._get_default_standardization_prompts()
    
    def _get_default_standardization_prompts(self) -> Dict[str, Any]:
        """Get default standardization prompts if no file exists."""
        return {
            'system_prompt': """You are an expert technical documentation standardizer. Your task is to rewrite documents to match organizational standards while preserving all original knowledge and information.

Key requirements:
1. Maintain all factual content and technical accuracy
2. Preserve code examples and technical specifications
3. Follow clear, professional documentation structure
4. Use consistent terminology and formatting
5. Ensure completeness - don't omit important information
6. Create proper hierarchical section organization""",
            
            'user_prompt_template': """Please standardize the following document to match our organizational documentation standards.

Original Document Title: {title}
Format: {format_type}
Content Sections: {sections_summary}

Original Content:
{content}

Requirements:
- Preserve all technical information and knowledge
- Use clear, professional language
- Follow proper documentation structure
- Include all code examples and specifications
- Use consistent terminology from our glossary: {terminology}
- Output in {output_format} format
- Create appropriate section hierarchy

Please rewrite this document to meet these standards while ensuring no important information is lost."""
        }
    
    def _load_terminology(self) -> Dict[str, Any]:
        """Load terminology configuration."""
        try:
            if self.terminology_path.exists():
                with open(self.terminology_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Error loading terminology: {e}")
            return {}
    
    def _initialize_extractors(self) -> List[ContentExtractor]:
        """Initialize content extractors."""
        extractors = [
            HTMLContentExtractor(),
            # Additional extractors can be added here
        ]
        return extractors
    
    def _find_suitable_extractor(self, content: str, file_path: Optional[str] = None) -> Optional[ContentExtractor]:
        """Find a suitable extractor for the given content."""
        for extractor in self.extractors:
            if extractor.can_extract(content, file_path):
                return extractor
        return None
    
    def standardize_document(self, content: str, file_path: Optional[str] = None, 
                           target_format: Optional[str] = None) -> Dict[str, Any]:
        """Standardize a document to organizational standards.
        
        Args:
            content: Raw document content
            file_path: Optional file path for format detection
            target_format: Optional target format override
            
        Returns:
            Dict containing standardized content and metadata
        """
        try:
            # Use target format if provided
            if target_format:
                original_format = self.output_format
                self.output_format = target_format.lower()
            
            # Extract structured content
            extractor = self._find_suitable_extractor(content, file_path)
            if not extractor:
                raise DocumentStandardizerError(
                    "No suitable extractor found for the provided content"
                )
            
            extracted = extractor.extract(content, file_path)
            
            # Build standardization prompt
            prompt_data = self._build_standardization_prompt(extracted)
            
            # Generate standardized content
            response = self._generate_standardized_content(prompt_data)
            
            # Post-process response
            result = self._process_standardization_response(response, extracted)
            
            # Restore original format if changed
            if target_format:
                self.output_format = original_format
            
            return result
            
        except Exception as e:
            self.error_handler.handle_error(
                DocumentStandardizerError(f"Standardization failed: {e}")
            )
            raise
    
    def _build_standardization_prompt(self, extracted: ExtractedContent) -> Dict[str, str]:
        """Build standardization prompt from extracted content."""
        # Create sections summary
        sections_summary = ", ".join(extracted.sections.keys()) if extracted.sections else "No sections detected"
        
        # Build content string
        content_parts = []
        if extracted.title:
            content_parts.append(f"Title: {extracted.title}")
        
        for section_name, section_content in extracted.sections.items():
            content_parts.append(f"\n## {section_name.replace('_', ' ').title()}\n{section_content}")
        
        content_str = "\n".join(content_parts) or extracted.raw_content
        
        # Build terminology context
        terminology_str = self._build_terminology_context()
        
        # Fill template
        user_prompt = self.prompt_config.get('user_prompt_template', '').format(
            title=extracted.title or "Untitled Document",
            format_type=extracted.format_type,
            sections_summary=sections_summary,
            content=content_str,
            terminology=terminology_str,
            output_format=self.output_format.upper()
        )
        
        return {
            'system_prompt': self.prompt_config.get('system_prompt', ''),
            'user_prompt': user_prompt
        }
    
    def _build_terminology_context(self) -> str:
        """Build terminology context string."""
        if not self.terminology:
            return "No specific terminology provided"
        
        context_parts = []
        
        # Add definitions if available
        if 'definitions' in self.terminology:
            context_parts.append("Key Terms:")
            for term, definition in self.terminology['definitions'].items():
                context_parts.append(f"- {term}: {definition}")
        
        # Add modules if available  
        if 'modules' in self.terminology:
            context_parts.append("\nRelevant Modules/Categories:")
            for category, modules in self.terminology['modules'].items():
                if isinstance(modules, list):
                    context_parts.append(f"- {category}: {', '.join(modules)}")
                else:
                    context_parts.append(f"- {category}: {modules}")
        
        return "\n".join(context_parts) if context_parts else "No specific terminology provided"
    
    def _generate_standardized_content(self, prompt_data: Dict[str, str]) -> CompletionResponse:
        """Generate standardized content using LLM."""
        request = CompletionRequest(
            messages=[
                {"role": "system", "content": prompt_data['system_prompt']},
                {"role": "user", "content": prompt_data['user_prompt']}
            ],
            model=self.model,
            temperature=self.temperature,
            max_tokens=4000
        )
        
        return self.client.generate_completion(request)
    
    def _process_standardization_response(self, response: CompletionResponse, 
                                        original: ExtractedContent) -> Dict[str, Any]:
        """Process the standardization response."""
        return {
            'standardized_content': response.content,
            'original_title': original.title,
            'original_format': original.format_type,
            'target_format': self.output_format,
            'sections_processed': list(original.sections.keys()),
            'metadata': {
                'provider': self.provider_name,
                'model': self.model,
                'temperature': self.temperature,
                'tokens_used': getattr(response, 'usage', {}).get('total_tokens', None) if hasattr(response, 'usage') else None,
                'original_metadata': original.metadata
            }
        }
    
    def standardize_file(self, file_path: Union[str, Path], 
                        output_path: Optional[Union[str, Path]] = None,
                        target_format: Optional[str] = None) -> Dict[str, Any]:
        """Standardize a document file.
        
        Args:
            file_path: Path to input file
            output_path: Optional output path
            target_format: Optional target format override
            
        Returns:
            Dict containing standardized content and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DocumentStandardizerError(f"File not found: {file_path}")
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin1') as f:
                content = f.read()
        
        # Standardize content
        result = self.standardize_document(content, str(file_path), target_format)
        
        # Save output if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result['standardized_content'])
            
            result['output_path'] = str(output_path)
        
        return result