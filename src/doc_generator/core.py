"""
Documentation Generator Core Module
Version 1.1.0

Advanced documentation generation with plugin-based recommendation system.
"""

__version__ = "1.1.0"

import os
import re
import json
import time
import yaml
import hashlib
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from bs4 import BeautifulSoup

from .plugin_manager import PluginManager
from .providers import ProviderManager, CompletionRequest
from .config import get_settings
from .exceptions import DocGeneratorError, ConfigurationError, ProviderError, FileOperationError
from .error_handler import ErrorHandler, retry_on_failure, handle_gracefully
from .cache import cached
from .command_tracker import CommandTracker
from .utils import get_output_directory

try:
    import pandas as pd
except ImportError:
    pd = None


class DocumentationGenerator:
    """Main class for generating documentation using multiple LLM providers with plugin support."""
    
    def __init__(self, prompt_yaml_path: str = None, shots_dir: str = None,
                 terminology_path: str = None, provider: str = 'auto', 
                 logger: Optional[logging.Logger] = None):
        """Initialize the documentation generator with configuration."""
        # Load settings from new configuration system
        self.settings = get_settings()
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize error handler
        self.error_handler = ErrorHandler(
            max_retries=self.settings.performance.retry_max_attempts,
            backoff_factor=self.settings.performance.retry_backoff_factor,
            logger=self.logger
        )
        
        # Use settings with fallback to provided values
        self.prompt_yaml_path = Path(
            prompt_yaml_path or self.settings.paths.prompts_dir / 'generator' / 'default.yaml'
        )
        self.shots_dir = Path(shots_dir or self.settings.paths.shots_dir)
        self.terminology_path = Path(
            terminology_path or self.settings.paths.terminology_path
        )
        
        # Initialize provider manager
        self.provider_manager = ProviderManager(logger=self.logger)
        
        # Set default provider with validation
        self._setup_provider(provider)
        
        # Load configurations with caching
        self.prompt_config = self._load_prompt_config()
        self.terminology = self._load_terminology()
        self.examples = self._load_examples()
        
        # Initialize plugin manager
        self.plugin_manager = PluginManager(
            terminology=self.terminology,
            logger=self.logger
        )
        self.plugin_manager.load_plugins()
        
        self.logger.info(
            f"DocumentationGenerator initialized with provider: {self.default_provider}"
        )
        
        # Keep legacy client for backward compatibility (deprecated)
        try:
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        except:
            self.client = None
    
    def _setup_provider(self, provider: str) -> None:
        """Setup and validate provider selection."""
        if provider == 'auto':
            self.default_provider = self.provider_manager.get_default_provider()
            if not self.default_provider:
                available = self.provider_manager.get_available_providers()
                raise ProviderError(
                    "No LLM providers are available. Check API keys.",
                    context={'available_providers': available}
                )
        else:
            available = self.provider_manager.get_available_providers()
            if provider not in available:
                raise ProviderError(
                    f"Provider '{provider}' is not available",
                    context={'available_providers': available}
                )
            self.default_provider = provider
        
    @cached(ttl=86400)  # Cache for 24 hours
    def _load_prompt_config(self) -> dict:
        """Load prompt configuration with caching and validation."""
        if not self.prompt_yaml_path.exists():
            self.logger.warning(
                f"Prompt config not found at {self.prompt_yaml_path}, using defaults"
            )
            return {
                'system_prompt': 'You are a technical documentation expert.',
                'documentation_structure': [
                    'Description', 'Installation', 'Usage', 'Examples', 'References'
                ]
            }
            
        try:
            with open(self.prompt_yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # Validate required fields
            if 'system_prompt' not in config:
                raise ConfigurationError(
                    "Missing 'system_prompt' in configuration",
                    context={'config_path': str(self.prompt_yaml_path)}
                )
                
            return config
            
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in prompt configuration: {e}",
                context={'config_path': str(self.prompt_yaml_path)}
            )
    
    @cached(ttl=86400)  # Cache for 24 hours
    def _load_terminology(self) -> dict:
        """Load terminology with caching and error handling."""
        if not self.terminology_path.exists():
            self.logger.info(f"No terminology file at {self.terminology_path}")
            return {}
            
        try:
            with open(self.terminology_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
                
        except Exception as e:
            self.logger.warning(f"Failed to load terminology: {e}")
            return {}
    
    @cached(ttl=86400)  # Cache for 24 hours
    def _load_examples(self) -> List[Dict[str, str]]:
        """Load few-shot examples with caching and validation."""
        examples = []
        
        if not self.shots_dir.exists():
            self.logger.info(f"No shots directory at {self.shots_dir}")
            return examples
            
        # Load examples from YAML files
        for yaml_file in self.shots_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    example = yaml.safe_load(f)
                    if example and isinstance(example, dict):
                        examples.append(example)
                        
            except Exception as e:
                self.logger.warning(f"Failed to load example {yaml_file}: {e}")
                continue
                
        self.logger.info(f"Loaded {len(examples)} few-shot examples")
        return examples
    
    def _extract_topic_from_query(self, query: str) -> str:
        """Extract the main topic from the query for filename generation."""
        # Try to extract topic using various patterns that capture multi-word phrases
        patterns = [
            r'documentation for (.+?)(?:\s+on|\s+using|\s+with|\s*$)',  # "documentation for Python Programming"
            r'using (.+?)(?:\s+on|\s+to|\s+for|\s*$)',                  # "using Machine Learning"
            r'about (.+?)(?:\s+on|\s+in|\s+with|\s*$)',                 # "about Parallel Computing"
            r'for (.+?) documentation',                                  # "for Deep Learning documentation"
            r'(?:create|write|generate).*?(?:for|about)\s+(.+?)(?:\s+on|\s+using|\s*$)',  # "create docs for Data Science"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                # Clean and format the extracted topic
                return topic.lower().replace(' ', '_').replace('-', '_')
        
        # Fallback: if no patterns match, clean the entire query as topic
        # Remove common directive words and use the remainder
        directive_words = ['create', 'make', 'generate', 'write', 'documentation', 'for', 'about', 'using', 'on', 'with']
        words = query.lower().split()
        
        # Filter out directive words and keep meaningful content
        meaningful_words = [word for word in words if word not in directive_words and len(word) > 2]
        
        if meaningful_words:
            return '_'.join(meaningful_words)
        
        # Final fallback
        return 'documentation'
    
    def _extract_topic_keywords(self, topic: str) -> List[str]:
        """Extract meaningful keywords from the topic for module matching."""
        import re
        
        # Convert to lowercase and extract words
        words = re.findall(r'\b\w+\b', topic.lower())
        
        # Filter out common stop words that aren't useful for module matching
        stop_words = {
            'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'using', 'how', 'what', 'when', 'where', 'why', 'computing', 'cluster',
            'fasrc', 'cannon', 'hpc', 'high', 'performance'
        }
        
        # Keep words that are 2+ characters and not stop words
        keywords = [word for word in words if len(word) >= 2 and word not in stop_words]
        
        return keywords
    
    def _extract_target_language(self, topic_keywords: List[str]) -> Optional[str]:
        """Extract target programming language from topic keywords."""
        language_mapping = {
            'python': 'python',
            'c': 'c', 
            'fortran': 'fortran',
            'mpi': 'c',  # MPI often associated with C
            'openmp': 'c'  # OpenMP often associated with C
        }
        
        for keyword in topic_keywords:
            if keyword.lower() in language_mapping:
                return language_mapping[keyword.lower()]
        return None
    
    def _calculate_example_relevance(self, example: Dict, topic_keywords: List[str]) -> float:
        """Calculate relevance score for a code example based on topic keywords."""
        score = 0.0
        
        # Convert strings to lowercase for matching
        file_path_lower = example.get('file_path', '').lower()
        description_lower = example.get('description', '').lower()
        name_lower = example.get('name', '').lower()
        
        for keyword in topic_keywords:
            keyword_lower = keyword.lower()
            
            # Directory keywords (high weight - indicates topic area)
            if keyword_lower in file_path_lower:
                score += 5.0
            
            # Description keywords (medium weight - indicates content relevance)
            if keyword_lower in description_lower:
                score += 3.0
            
            # File name keywords (lower weight - may be coincidental)
            if keyword_lower in name_lower:
                score += 2.0
        
        return score
    
    def _find_relevant_code_examples(self, topic_keywords: List[str]) -> List[Dict]:
        """Find code examples relevant to topic keywords."""
        if 'code_examples' not in self.terminology:
            return []
        
        relevant_examples = []
        code_examples = self.terminology['code_examples']
        
        # Extract primary language from topic keywords
        target_language = self._extract_target_language(topic_keywords)
        
        # Search through all languages, prioritizing target language
        for language, examples in code_examples.items():
            if target_language and language != target_language:
                continue  # Skip if we have a target language and this isn't it
                
            for example in examples:
                relevance_score = self._calculate_example_relevance(example, topic_keywords)
                if relevance_score > 0:
                    example_copy = example.copy()
                    example_copy['relevance_score'] = relevance_score
                    relevant_examples.append(example_copy)
        
        # If no target language specified, include examples from all languages
        if not target_language:
            for language, examples in code_examples.items():
                for example in examples:
                    relevance_score = self._calculate_example_relevance(example, topic_keywords)
                    if relevance_score > 0:
                        example_copy = example.copy()
                        example_copy['relevance_score'] = relevance_score
                        relevant_examples.append(example_copy)
        
        # Sort by relevance and return top results
        relevant_examples.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_examples[:8]
    
    def _find_relevant_modules(self, topic_keywords: List[str]) -> List[Dict]:
        """Find modules relevant to the topic keywords."""
        if not self.terminology.get('hpc_modules') or not topic_keywords:
            return []
        
        relevant_modules = []
        
        for module in self.terminology['hpc_modules']:
            score = 0
            
            # Check for keyword matches in module name
            module_name_lower = module['name'].lower()
            for keyword in topic_keywords:
                if keyword in module_name_lower:
                    score += 3  # High weight for name matches
            
            # Check for keyword matches in module description
            module_desc_lower = module['description'].lower()
            for keyword in topic_keywords:
                if keyword in module_desc_lower:
                    score += 2  # Medium weight for description matches
            
            # Check for keyword matches in category
            module_category_lower = module['category'].lower()
            for keyword in topic_keywords:
                if keyword in module_category_lower:
                    score += 1  # Lower weight for category matches
            
            if score > 0:
                module_with_score = module.copy()
                module_with_score['relevance_score'] = score
                relevant_modules.append(module_with_score)
        
        # Sort by relevance score (highest first) and limit results
        relevant_modules.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_modules[:15]  # Limit to top 15 to avoid token overflow
    
    def _build_terminology_context(self, topic: str) -> str:
        """Build context-aware terminology based on the topic."""
        if not self.terminology:
            return ""
        
        context_parts = []
        topic_keywords = self._extract_topic_keywords(topic)
        
        # Get recommendations from all loaded plugins
        plugin_recommendations = self.plugin_manager.get_formatted_recommendations(topic)
        if plugin_recommendations.strip():
            context_parts.append(plugin_recommendations)
        
        # Include cluster commands for most topics
        if 'cluster_commands' in self.terminology:
            context_parts.append("\nCommon SLURM Commands:")
            for cmd in self.terminology['cluster_commands'][:6]:  # Limit to most important
                context_parts.append(f"- {cmd['name']}: {cmd['description']}")
                if 'usage' in cmd:
                    context_parts.append(f"  Usage: {cmd['usage']}")
        
        # Include relevant filesystems
        if 'filesystems' in self.terminology:
            context_parts.append("\nFASRC Filesystems:")
            for fs in self.terminology['filesystems']:
                context_parts.append(f"- {fs['name']}: {fs['description']}")
        
        # Include partition information for GPU/parallel topics
        if 'partitions' in self.terminology:
            gpu_parallel_keywords = ['gpu', 'parallel', 'mpi', 'cuda', 'computing']
            if any(keyword in topic_keywords for keyword in gpu_parallel_keywords):
                context_parts.append("\nCluster Partitions:")
                for partition in self.terminology['partitions']:
                    context_parts.append(f"- {partition['name']}: {partition['description']}")
        
        # Add relevant code examples section
        relevant_examples = self._find_relevant_code_examples(topic_keywords)
        if relevant_examples:
            context_parts.append("\nRelevant Code Examples:")
            for example in relevant_examples:
                context_parts.append(f"- {example['name']} ({example['language']})")
                context_parts.append(f"  Path: {example['file_path']}")
                context_parts.append(f"  Description: {example['description']}")
        
        return "\n".join(context_parts)
    
    def _build_system_prompt(self, topic: str = "", **kwargs) -> str:
        """Build the system prompt with topic and parameter substitution."""
        system_template = self.prompt_config.get('system_prompt', 
            'You are a technical documentation expert creating HTML knowledge base articles.')
        
        # Build placeholders dictionary
        placeholders = {
            'topic': topic,
            'organization': 'FASRC',
            'cluster_name': 'FASRC cluster',
            'audience': 'graduate-level researchers'
        }
        
        # Add config-defined placeholders
        if 'placeholders' in self.prompt_config:
            placeholders.update(self.prompt_config['placeholders'])
        
        # Override with any passed kwargs
        placeholders.update(kwargs)
        
        # Format the system prompt template
        try:
            formatted_prompt = system_template.format(**placeholders)
        except KeyError as e:
            print(f"Warning: Missing placeholder {e} in system prompt template")
            formatted_prompt = system_template
        
        # Add terminology context
        terminology_context = self._build_terminology_context(topic)
        if terminology_context:
            formatted_prompt += f"\n\nRelevant HPC Environment Information:\n{terminology_context}"
            formatted_prompt += "\n\nWhen writing documentation, reference these specific modules, commands, and resources where appropriate. Use exact module names as listed above."
        
        # Add any terms/definitions from prompt config
        if 'terms' in self.prompt_config:
            formatted_prompt += "\n\nAdditional Key Terms:\n"
            for term, definition in self.prompt_config['terms'].items():
                formatted_prompt += f"- {term}: {definition}\n"
        
        return formatted_prompt
    
    def generate_documentation(self, query: str, runs: int = 5, 
                             model: str = None, 
                             temperature: float = 0.7,
                             topic_filename: str = None,
                             output_dir: str = 'output',
                             provider: Optional[str] = None) -> List[str]:
        """Generate multiple documentation pages based on the query using multiple LLM providers."""
        # Set default model if not provided
        if model is None:
            model = self.provider_manager.get_default_model()
            if not model:
                raise ProviderError("No default model available. Check provider configuration.")
        
        # Determine which provider to use
        if provider:
            llm_provider = self.provider_manager.get_provider(provider)
            if not llm_provider:
                available = self.provider_manager.get_available_providers()
                raise ProviderError(f"Provider '{provider}' not available. Available providers: {available}")
        else:
            llm_provider = self.provider_manager.get_provider_for_model(model)
            if not llm_provider:
                # Fallback to default provider
                llm_provider = self.provider_manager.get_provider(self.default_provider)
                self.logger.warning(f"Model '{model}' not found, using default provider '{self.default_provider}' with default model")
                model = self.provider_manager.get_default_model()
        
        # Validate model/provider combination
        is_valid, error_msg = self.provider_manager.validate_model_provider_combination(
            model, llm_provider.get_provider_name()
        )
        if not is_valid:
            raise ProviderError(error_msg)
        
        if topic_filename is None:
            topic_filename = self._extract_topic_from_query(query)
        
        # Extract topic for terminology context
        topic = self._extract_topic_from_query(query).replace('_', ' ')
        
        generated_files = []
        
        # Build messages with topic-aware system prompt
        system_prompt = self._build_system_prompt(topic)
        
        for i in range(runs):
            try:
                messages = [
                    {'role': 'system', 'content': system_prompt}
                ]
                
                # Add few-shot examples
                for example in self.examples:
                    if 'role' in example and 'content' in example:
                        messages.append({
                            'role': example['role'],
                            'content': example['content']
                        })
                
                # Add the actual query
                messages.append({'role': 'user', 'content': query})
                
                # Create completion request
                request = CompletionRequest(
                    messages=messages,
                    model=model,
                    temperature=temperature
                )
                
                # Generate completion using provider
                response = llm_provider.generate_completion(request)
                content = response.content
                
                # Generate filename with provider info
                provider_name = response.provider
                model_name = model.replace('-', '').replace('.', '')
                temp_str = str(temperature).replace('.', '')

                if runs == 1:
                    filename = f'{topic_filename}_{provider_name}_{model_name}_temp{temp_str}.html'
                else:
                    filename = f'{topic_filename}_{provider_name}_{model_name}_temp{temp_str}_v{i+1}.html'
                
                # Save the response
                output_path = Path(output_dir)
                output_path.mkdir(exist_ok=True)
                
                filepath = output_path / filename
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                generated_files.append(str(filepath))
                self.logger.info(f"Generated: {filename} using {provider_name} ({model})")
                print(f"✓ Generated: {filepath}")
                
                # Save command file alongside the generated document
                command_file = CommandTracker.save_command_file(str(filepath))
                if command_file:
                    print(f"✓ Command saved: {Path(command_file).name}")
                
                # Log cost estimate if available
                if response.usage:
                    cost_estimate = llm_provider.estimate_cost(request, response)
                    if cost_estimate:
                        self.logger.debug(f"Estimated cost: ${cost_estimate:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error generating documentation (run {i+1}): {e}")
                print(f"✗ Error generating documentation (run {i+1}): {e}")
        
        return generated_files



class DocumentAnalyzer:
    """Analyze and extract sections from HTML documentation."""
    
    def __init__(self, section_headers: List[str] = None):
        self.section_headers = section_headers or [
            'Description', 'Installation', 'Usage', 'Examples', 'References'
        ]
        
    def extract_sections(self, html_content: str) -> Dict[str, str]:
        """Extract sections from HTML content based on headers."""
        soup = BeautifulSoup(html_content, 'html.parser')
        sections = {}
        
        # Find all headers (h1, h2, h3, etc.)
        headers = soup.find_all(['h1', 'h2', 'h3', 'h4'])
        
        for i, header in enumerate(headers):
            header_text = header.get_text().strip()
            
            # Check if this header matches any of our target sections
            for section_name in self.section_headers:
                if section_name.lower() in header_text.lower():
                    # Extract content between this header and the next
                    content_parts = []
                    
                    # Get all siblings until the next header
                    for sibling in header.find_next_siblings():
                        if sibling.name in ['h1', 'h2', 'h3', 'h4']:
                            break
                        content_parts.append(str(sibling))
                    
                    sections[section_name] = '\n'.join(content_parts)
                    break
        
        return sections
    
    def calculate_section_score(self, section_content: str, section_name: str) -> float:
        """Calculate a quality score for a section."""
        if not section_content:
            return 0.0
        
        soup = BeautifulSoup(section_content, 'html.parser')
        text = soup.get_text().strip()
        
        # Base score on multiple factors
        score = 0.0
        
        # Length (not too short, not too long)
        word_count = len(text.split())
        if section_name == "Description":
            ideal_length = 150
            score += max(0, 1 - abs(word_count - ideal_length) / ideal_length) * 20
        else:
            score += min(word_count / 100, 1) * 20  # Longer is generally better for other sections
        
        # Code examples (for Installation, Usage, Examples)
        if section_name in ["Installation", "Usage", "Examples"]:
            code_blocks = soup.find_all(['code', 'pre'])
            score += min(len(code_blocks) * 10, 30)
        
        # Lists (good for organization)
        lists = soup.find_all(['ul', 'ol'])
        score += min(len(lists) * 5, 15)
        
        # Links (good for References)
        if section_name == "References":
            links = soup.find_all('a')
            score += min(len(links) * 10, 30)
        else:
            links = soup.find_all('a')
            score += min(len(links) * 2, 10)
        
        # Formatting variety (bold, italic, etc.)
        formatting_tags = soup.find_all(['strong', 'em', 'b', 'i'])
        score += min(len(formatting_tags) * 2, 10)
        
        # Clarity (sentences not too long)
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        if 10 <= avg_sentence_length <= 25:
            score += 15
        
        return min(score, 100)  # Cap at 100


class GPTQualityEvaluator:
    """Evaluate documentation quality using GPT for subjective metrics."""
    
    def __init__(self, client, model='gpt-4', analysis_prompt_path='./prompts/analysis/default.yaml'):
        self.client = client
        self.model = model
        self.analysis_prompt_path = analysis_prompt_path
        self.analysis_config = self._load_analysis_config(analysis_prompt_path)
        
    def _load_analysis_config(self, path: str) -> dict:
        """Load analysis prompt configuration from YAML file."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Analysis config file not found at {path}")
            return {}
        
    def create_evaluation_prompt(self, section_content: str, section_name: str, 
                               topic: str, criteria: str) -> str:
        """Create a prompt for evaluating specific quality criteria."""
        
        # Only use configured prompts from YAML file
        if not self.analysis_config or 'analysis_prompts' not in self.analysis_config:
            raise ValueError(f"Analysis configuration not found. Please ensure {self.analysis_prompt_path} exists and contains 'analysis_prompts' section.")
        
        template = self.analysis_config['analysis_prompts'].get(criteria)
        if not template:
            raise ValueError(f"Analysis prompt for criteria '{criteria}' not found in configuration.")
        
        return template.format(
            section_name=section_name,
            topic=topic,
            content=section_content
        )
    
    def parse_gpt_response(self, response: str) -> Tuple[float, str]:
        """Parse the GPT response to extract score and explanation."""
        try:
            # Try to parse as JSON first
            result = json.loads(response)
            return result['score'], result['explanation']
        except:
            # Fallback: extract number and text
            import re
            score_match = re.search(r'\b(\d+)\b', response)
            score = float(score_match.group(1)) if score_match else 50.0
            
            # Extract explanation (everything after the score)
            explanation = response.split(str(int(score)), 1)[-1].strip()
            return score, explanation
    
    def evaluate_section(self, section_content: str, section_name: str, 
                        topic: str, criteria: List[str] = None) -> Dict[str, Dict]:
        """Evaluate a section on multiple criteria."""
        if criteria is None:
            criteria = ['technical_accuracy', 'writing_style', 'completeness']
        
        results = {}
        
        for criterion in criteria:
            if not section_content.strip():
                results[criterion] = {
                    'score': 0,
                    'explanation': 'Section is empty'
                }
                continue
            
            # Truncate very long sections
            max_chars = 3000
            content = section_content[:max_chars] + "..." if len(section_content) > max_chars else section_content
            
            prompt = self.create_evaluation_prompt(content, section_name, topic, criterion)
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert technical documentation reviewer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=150
                )
                
                score, explanation = self.parse_gpt_response(response.choices[0].message.content)
                results[criterion] = {
                    'score': score,
                    'explanation': explanation
                }
                
            except Exception as e:
                results[criterion] = {
                    'score': 0,
                    'explanation': f'Error: {str(e)}'
                }
            
            # Rate limiting
            time.sleep(0.5)
        
        return results


class CodeExampleScanner:
    """Scan filesystem for code examples and manage terminology metadata."""
    
    def __init__(self):
        # Try to import pygments for language detection
        try:
            from pygments.lexers import get_lexer_for_filename
            from pygments.util import ClassNotFound
            self.get_lexer_for_filename = get_lexer_for_filename
            self.ClassNotFound = ClassNotFound
            self.has_pygments = True
        except ImportError:
            print("Warning: pygments not found. Install with: pip install pygments")
            self.has_pygments = False
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect language using pygments or fallback to extension mapping."""
        if self.has_pygments:
            try:
                lexer = self.get_lexer_for_filename(str(file_path))
                # Map pygments names to our preferred names
                name_mapping = {
                    'python': 'python',
                    'julia': 'julia',
                    'matlab': 'matlab',
                    'fortran': 'fortran',
                    'c': 'c',
                    'c++': 'cpp',
                    'cpp': 'cpp',
                    'cuda': 'cuda',
                    'r': 'r',
                    'bash': 'bash',
                    'shell': 'bash',
                    'perl': 'perl',
                    'idl': 'idl'
                }
                detected_name = lexer.name.lower()
                return name_mapping.get(detected_name, detected_name)
            except self.ClassNotFound:
                pass
        
        # Fallback to extension mapping
        extension_mapping = {
            '.py': 'python',
            '.pyw': 'python',
            '.jl': 'julia',
            '.m': 'matlab',
            '.f90': 'fortran',
            '.f95': 'fortran',
            '.f': 'fortran',
            '.for': 'fortran',
            '.c': 'c',
            '.h': 'c',
            '.cpp': 'cpp',
            '.cxx': 'cpp',
            '.cc': 'cpp',
            '.hpp': 'cpp',
            '.cu': 'cuda',
            '.cuh': 'cuda',
            '.cuf': 'cuda-fortran',
            '.R': 'r',
            '.r': 'r',
            '.sh': 'bash',
            '.bash': 'bash',
            '.pl': 'perl',
            '.pro': 'idl',
            '.do': 'stata'
        }
        
        return extension_mapping.get(file_path.suffix.lower(), 'unknown')
    
    def _detect_comment_style(self, content: str) -> str:
        """Detect comment style from file content."""
        lines = content.split('\n')[:10]
        
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                return '#'
            elif line.startswith('//'):
                return '//'
            elif line.startswith('!'):
                return '!'
            elif line.startswith('%'):
                return '%'
            elif line.startswith(';'):
                return ';'
            elif '/*' in line:
                return '/* */'
        
        return '#'  # Default fallback
    
    def _extract_description(self, content: str) -> str:
        """Extract description using detected comment style."""
        comment_char = self._detect_comment_style(content)
        lines = content.split('\n')[:20]
        
        descriptions = []
        for line in lines:
            line = line.strip()
            if comment_char == '/* */' and ('/*' in line or '*' in line):
                # Handle C-style comments
                if '/*' in line:
                    desc = line.split('/*')[1].split('*/')[0].strip()
                elif line.startswith('*') and not line.startswith('*/'):
                    desc = line[1:].strip()
                else:
                    continue
            elif line.startswith(comment_char) and len(line) > len(comment_char) + 2:
                desc = line[len(comment_char):].strip()
            else:
                continue
            
            # Skip decorative lines and very short descriptions
            if desc and not desc.startswith(('=', '-', '*', '+', '#')) and len(desc) > 10:
                descriptions.append(desc)
                if len(descriptions) >= 2:
                    break
        
        return ' '.join(descriptions)[:200] if descriptions else "Code example"
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file contents."""
        try:
            return hashlib.sha256(file_path.read_bytes()).hexdigest()
        except Exception:
            return ""
    
    def _find_slurm_files(self, file_path: Path) -> List[str]:
        """Find associated SLURM batch files."""
        directory = file_path.parent
        slurm_patterns = ['*.sbatch', '*.slurm', 'run.*']
        
        slurm_files = []
        for pattern in slurm_patterns:
            matches = list(directory.glob(pattern))
            slurm_files.extend(str(f) for f in matches if f.is_file())
        
        return list(set(slurm_files))  # Remove duplicates
    
    def _extract_file_info(self, file_path: Path) -> Optional[Dict]:
        """Extract metadata from a code file."""
        if not file_path.is_file():
            return None
        
        try:
            # Read file content for description extraction
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)  # First 1000 chars
            
            # Detect language using pygments
            language = self._detect_language(file_path)
            
            if language == 'unknown':
                return None  # Skip unknown file types
            
            # Extract description from comments
            description = self._extract_description(content)
            
            # Find associated SLURM files
            slurm_files = self._find_slurm_files(file_path)
            
            file_info = {
                'name': file_path.stem.replace('_', ' ').title(),
                'file_path': str(file_path.relative_to(Path.cwd()) if file_path.is_relative_to(Path.cwd()) else file_path),
                'language': language,
                'description': description,
                'file_hash': self._calculate_file_hash(file_path),
                'last_modified': file_path.stat().st_mtime,
                'file_size': file_path.stat().st_size,
                'directory': file_path.parent.name,
                'scanned_at': datetime.now().isoformat()
            }
            
            if slurm_files:
                file_info['slurm_files'] = slurm_files
            
            return file_info
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def scan_directory(self, scan_path: str, max_files: int = 100) -> List[Dict]:
        """Scan directory for code examples."""
        scan_path = Path(scan_path).resolve()
        
        if not scan_path.exists():
            raise ValueError(f"Path does not exist: {scan_path}")
        
        print(f"🔍 Scanning {scan_path} for code examples...")
        
        code_examples = []
        file_count = 0
        
        # Walk through directory tree
        for file_path in scan_path.rglob('*'):
            if file_count >= max_files:
                print(f"⚠️  Reached maximum file limit ({max_files})")
                break
                
            if file_path.is_file():
                file_info = self._extract_file_info(file_path)
                if file_info:
                    code_examples.append(file_info)
                    file_count += 1
                    if file_count % 10 == 0:
                        print(f"   Processed {file_count} files...")
        
        print(f"✅ Found {len(code_examples)} code examples")
        return code_examples
    
    def update_terminology_file(self, terminology_path: str, code_examples: List[Dict]) -> None:
        """Update terminology file with code examples."""
        try:
            with open(terminology_path, 'r') as f:
                terminology = yaml.safe_load(f) or {}
        except FileNotFoundError:
            terminology = {}
        
        # Organize examples by language
        examples_by_language = {}
        for example in code_examples:
            language = example['language']
            if language not in examples_by_language:
                examples_by_language[language] = []
            examples_by_language[language].append(example)
        
        # Update terminology with code examples
        if 'code_examples' not in terminology:
            terminology['code_examples'] = {}
        
        for language, examples in examples_by_language.items():
            terminology['code_examples'][language] = examples
        
        # Add metadata about the scan
        terminology['code_examples_metadata'] = {
            'last_scan': datetime.now().isoformat(),
            'total_examples': len(code_examples),
            'languages': list(examples_by_language.keys()),
            'scan_summary': {lang: len(examples) for lang, examples in examples_by_language.items()}
        }
        
        # Write updated terminology
        with open(terminology_path, 'w') as f:
            yaml.dump(terminology, f, default_flow_style=False, sort_keys=False, indent=2)
        
        print(f"✅ Updated {terminology_path} with {len(code_examples)} code examples")
        print(f"   Languages found: {', '.join(examples_by_language.keys())}")
    
    def check_for_updates(self, terminology_path: str) -> List[str]:
        """Check if any code examples have been updated since last scan."""
        try:
            with open(terminology_path, 'r') as f:
                terminology = yaml.safe_load(f) or {}
        except FileNotFoundError:
            return []
        
        code_examples = terminology.get('code_examples', {})
        updated_files = []
        
        for language, examples in code_examples.items():
            for example in examples:
                file_path = Path(example['file_path'])
                if file_path.exists():
                    current_hash = self._calculate_file_hash(file_path)
                    stored_hash = example.get('file_hash', '')
                    
                    if current_hash != stored_hash:
                        updated_files.append(str(file_path))
        
        return updated_files


def load_and_analyze_versions(topic_filename: str, model: str, temperature: str, num_versions: int = 5, output_dir: str = 'output'):
    """Load all versions and extract their sections."""
    analyzer = DocumentAnalyzer()
    output_path = Path(output_dir)
    all_sections = {}
    
    for version in range(1, num_versions + 1):
        # Construct filename
        if num_versions == 1:
            filename = f'{topic_filename}_{model}_temp{temperature}.html'
        else:
            filename = f'{topic_filename}_{model}_temp{temperature}_v{version}.html'
        
        filepath = output_path / filename
        
        if not filepath.exists():
            print(f"⚠️  File not found: {filepath}")
            continue
            
        # Load and extract sections
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        sections = analyzer.extract_sections(content)
        all_sections[f'Version {version}'] = sections
        
        print(f"✅ Loaded Version {version}: {filename}")
        print(f"   Found sections: {', '.join(sections.keys())}")
    
    return all_sections, analyzer


def analyze_sections(all_sections: Dict[str, Dict[str, str]], analyzer: DocumentAnalyzer) -> List[Dict]:
    """Analyze and score all sections across versions."""
    scores_data = []
    
    for section_name in analyzer.section_headers:
        section_scores = {}
        
        for version, sections in all_sections.items():
            if section_name in sections:
                score = analyzer.calculate_section_score(
                    sections[section_name], 
                    section_name
                )
                section_scores[version] = score
            else:
                section_scores[version] = 0
        
        # Find best version for this section
        if section_scores:
            best_version = max(section_scores, key=section_scores.get)
            best_score = section_scores[best_version]
        else:
            best_version = "N/A"
            best_score = 0
        
        scores_data.append({
            'Section': section_name,
            **section_scores,
            'Best Version': best_version,
            'Best Score': best_score
        })
    
    return scores_data


def compile_best_document(all_sections: Dict[str, Dict[str, str]], 
                         scores_data: List[Dict],
                         topic: str,
                         model: str,
                         temperature: float,
                         manual_overrides: Dict[str, str] = None) -> str:
    """Compile the best sections into a final document."""
    
    # Allow manual overrides if needed
    manual_overrides = manual_overrides or {}
    
    # Start building the final HTML
    html_parts = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f'<title>{topic} Documentation - Best Compilation</title>',
        '<style>',
        'body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; ',
        '       line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; }',
        'h1, h2, h3 { color: #2c3e50; }',
        'code { background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }',
        'pre { background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }',
        '.metadata { background-color: #e8f4f8; padding: 15px; border-radius: 5px; ',
        '            margin-bottom: 30px; font-size: 0.9em; }',
        '.section { margin-bottom: 40px; }',
        '.version-note { color: #7f8c8d; font-size: 0.85em; font-style: italic; }',
        '</style>',
        '</head>',
        '<body>',
        f'<h1>{topic} Documentation</h1>',
        '<div class="metadata">',
        f'<strong>Compiled from best sections across {len(all_sections)} versions</strong><br>',
        f'Generated using: {model} (Temperature: {temperature})<br>',
        f'Compilation date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '</div>'
    ]
    
    # Add each section
    for row in scores_data:
        section_name = row['Section']
        best_version = manual_overrides.get(section_name, row['Best Version'])
        
        if best_version != "N/A" and best_version in all_sections:
            if section_name in all_sections[best_version]:
                html_parts.append('<div class="section">')
                html_parts.append(f'<h2>{section_name}</h2>')
                html_parts.append(f'<span class="version-note">From {best_version}</span>')
                html_parts.append(all_sections[best_version][section_name])
                html_parts.append('</div>')
    
    # Close HTML
    html_parts.extend([
        '<div class="metadata" style="margin-top: 50px;">',
        '<strong>Section Sources:</strong><br>',
    ])
    
    # Add section source summary
    for row in scores_data:
        section_name = row['Section']
        best_version = manual_overrides.get(section_name, row['Best Version'])
        score = row['Best Score']
        html_parts.append(f'{section_name}: {best_version} (Score: {score:.1f})<br>')
    
    html_parts.extend([
        '</div>',
        '</body>',
        '</html>'
    ])
    
    return '\n'.join(html_parts)


def generate_analysis_report(topic: str, model: str, temperature: float, 
                           all_sections: Dict, scores_data: List[Dict]) -> str:
    """Generate a detailed analysis report of the compilation process."""
    
    report = []
    report.append(f"# Documentation Analysis Report for {topic}")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Model: {model}, Temperature: {temperature}")
    report.append(f"Analyzed {len(all_sections)} versions\n")
    
    report.append("## Section Scores\n")
    report.append("| Section | " + " | ".join(all_sections.keys()) + " | Best Version | Best Score |")
    report.append("|---------|" + "---|" * (len(all_sections) + 2))
    
    for row in scores_data:
        line = f"| {row['Section']} | "
        for version in all_sections.keys():
            score = row.get(version, 0)
            line += f"{score:.1f} | "
        line += f"{row['Best Version']} | {row['Best Score']:.1f} |"
        report.append(line)
    
    report.append("\n## Key Findings\n")
    
    # Find most consistent version
    version_totals = {}
    for version in all_sections.keys():
        total = sum(row.get(version, 0) for row in scores_data)
        version_totals[version] = total
    
    if version_totals:
        best_overall = max(version_totals, key=version_totals.get)
        report.append(f"- **Best Overall Version**: {best_overall} (Total Score: {version_totals[best_overall]:.1f})")
    
    # Find sections with high variance
    report.append("\n### Section Quality Variance")
    for row in scores_data:
        version_scores = [row.get(v, 0) for v in all_sections.keys()]
        if version_scores:
            variance = max(version_scores) - min(version_scores)
            if variance > 20:
                report.append(f"- **{row['Section']}**: High variance ({variance:.1f} points) - quality varies significantly between versions")
    
    report.append("\n## Recommendations")
    report.append("- Review sections with high variance manually")
    report.append("- Consider regenerating sections with scores below 50")
    report.append("- The compiled document uses the best scoring section from each category")
    
    return '\n'.join(report)


def compare_versions(topic_filename: str, model: str, temperature: str, runs: int, output_dir: str = 'output'):
    """Compare key differences between generated versions."""
    output_path = Path(output_dir)
    
    # Build filename pattern
    model_clean = model.replace('-', '').replace('.', '')
    temp_str = str(temperature).replace('.', '')
    
    if runs == 1:
        pattern = f'{topic_filename}_{model_clean}_temp{temp_str}.html'
        files = [output_path / pattern] if (output_path / pattern).exists() else []
    else:
        files = [output_path / f'{topic_filename}_{model_clean}_temp{temp_str}_v{i}.html' 
                for i in range(1, runs + 1) if (output_path / f'{topic_filename}_{model_clean}_temp{temp_str}_v{i}.html').exists()]
    
    if len(files) < 2:
        print("Need at least 2 versions to compare")
        return
    
    print(f"📊 Comparing {len(files)} versions of documentation:\n")
    
    for i, file in enumerate(files, 1):
        with open(file, 'r') as f:
            content = f.read()
            
        # Extract some metrics
        word_count = len(content.split())
        line_count = len(content.splitlines())
        has_examples = 'example' in content.lower()
        has_code_blocks = '<code>' in content or '<pre>' in content
        
        print(f"Version {i} ({file.name}):")
        print(f"  - Words: {word_count}")
        print(f"  - Lines: {line_count}")
        print(f"  - Has examples: {'Yes' if has_examples else 'No'}")
        print(f"  - Has code blocks: {'Yes' if has_code_blocks else 'No'}")
        print()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Generate technical documentation using OpenAI GPT models')
    parser.add_argument('--topic', required=True, help='Topic for documentation generation')
    parser.add_argument('--runs', type=int, default=5, help='Number of variations to generate (1-10)')
    parser.add_argument('--model', default='gpt-4o-mini', help='OpenAI model to use')
    parser.add_argument('--temperature', type=float, default=0.7, help='Creativity level (0.0-1.0)')
    parser.add_argument('--query-template', default="Create a knowledge base article with regards to using {topic} on the FASRC cluster, using the tone of graduate level Academic Computing documentation.", 
                        help='Query template (use {topic} placeholder)')
    parser.add_argument('--analyze', action='store_true', help='Run analysis on generated files')
    parser.add_argument('--compare', action='store_true', help='Compare generated versions')
    parser.add_argument('--install-deps', action='store_true', help='Install required dependencies')
    parser.add_argument('--skip-generation', action='store_true', help='Skip generation and only run analysis')
    parser.add_argument('--terminology', default='terminology.yaml', help='Path to terminology configuration file')
    parser.add_argument('--scan-code-examples', metavar='PATH', help='Scan filesystem path for code examples and update terminology file')
    parser.add_argument('--update-code-examples', action='store_true', help='Update existing code examples in terminology file')
    parser.add_argument('--generator-prompt', default='./prompts/generator/default.yaml', help='Path to generator prompt configuration file')
    parser.add_argument('--analysis-prompt', default='./prompts/analysis/default.yaml', help='Path to analysis prompt configuration file')
    parser.add_argument('--output-dir', default='output', help='Output directory for generated files (default: output)')
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        print("📦 Installing dependencies...")
        install_dependencies()
        print("✅ Dependencies installed")
    
    # Load environment variables
    load_dotenv()
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables!")
        print("   Please create a .env file with: OPENAI_API_KEY=your-key-here")
        return 1
    
    # Handle code example scanning
    if args.scan_code_examples:
        print(f"🔍 Scanning code examples from: {args.scan_code_examples}")
        scanner = CodeExampleScanner()
        try:
            code_examples = scanner.scan_directory(args.scan_code_examples)
            scanner.update_terminology_file(args.terminology, code_examples)
            print("✅ Code example scanning complete")
            return 0
        except Exception as e:
            print(f"❌ Error scanning code examples: {e}")
            return 1
    
    # Handle code example updates
    if args.update_code_examples:
        print(f"🔄 Checking for code example updates...")
        scanner = CodeExampleScanner()
        try:
            updated_files = scanner.check_for_updates(args.terminology)
            if updated_files:
                print(f"📝 Found {len(updated_files)} updated files:")
                for file_path in updated_files:
                    print(f"   - {file_path}")
                
                # Ask user if they want to rescan
                response = input("Rescan updated files? [y/N]: ").lower()
                if response == 'y':
                    # Rescan only the directories containing updated files
                    scan_dirs = set(Path(f).parent for f in updated_files)
                    all_examples = []
                    for scan_dir in scan_dirs:
                        examples = scanner.scan_directory(str(scan_dir))
                        all_examples.extend(examples)
                    scanner.update_terminology_file(args.terminology, all_examples)
                    print("✅ Code examples updated")
            else:
                print("✅ All code examples are up to date")
            return 0
        except Exception as e:
            print(f"❌ Error checking code example updates: {e}")
            return 1
    
    # Generate topic filename
    topic_filename = args.topic.lower().replace(' ', '_')
    
    # Get the proper output directory (with timestamp if using default)
    logger = logging.getLogger(__name__)
    output_dir = get_output_directory(args.output_dir, logger)
    
    generated_files = []
    
    if not args.skip_generation:
        # Initialize generator
        print("🔧 Initializing documentation generator...")
        try:
            generator = DocumentationGenerator(
                prompt_yaml_path=args.generator_prompt,
                shots_dir='shots/',
                terminology_path=args.terminology
            )
            print("✅ Generator initialized successfully")
            print(f"📁 Found {len(generator.examples)} examples")
        except Exception as e:
            print(f"❌ Error initializing generator: {e}")
            return 1
        
        # Build the query from template
        query = args.query_template.format(topic=args.topic)
        
        print(f"\n{'='*60}")
        print(f"📝 Generating documentation for: {args.topic}")
        print(f"📋 Query: {query}")
        print(f"🔄 Generating {args.runs} variations...")
        print(f"{'='*60}\n")
        
        # Track generation time
        start_time = datetime.now()
        
        # Generate the documentation
        generated_files = generator.generate_documentation(
            query=query,
            runs=args.runs,
            model=args.model,
            temperature=args.temperature,
            topic_filename=topic_filename,
            output_dir=output_dir
        )
        
        # Calculate elapsed time
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\n{'='*60}")
        print(f"✅ Generation complete!")
        print(f"⏱️  Time taken: {elapsed:.2f} seconds")
        print(f"📁 Generated {len(generated_files)} files:")
        for file in generated_files:
            print(f"   - {file}")
        print(f"{'='*60}")
    
    # Run comparison if requested
    if args.compare:
        print(f"\n📊 Running comparison...")
        compare_versions(topic_filename, args.model, args.temperature, args.runs, output_dir)
    
    # Run analysis if requested
    if args.analyze:
        print(f"\n🔍 Running analysis...")
        
        # Load and analyze versions
        model_clean = args.model.replace('-', '').replace('.', '')
        temp_str = str(args.temperature).replace('.', '')
        
        all_sections, analyzer = load_and_analyze_versions(
            topic_filename=topic_filename,
            model=model_clean,
            temperature=temp_str,
            num_versions=args.runs,
            output_dir=output_dir
        )
        
        if not all_sections:
            print("⚠️  No files found for analysis")
            return 1
        
        # Analyze sections
        scores_data = analyze_sections(all_sections, analyzer)
        
        print("\n📊 Section Analysis Results:")
        print("="*80)
        
        # Create a simple table if pandas is not available
        if pd is not None:
            df = pd.DataFrame(scores_data)
            print(df.to_string(index=False))
        else:
            # Simple table format
            print(f"{'Section':<12} {'Best Version':<15} {'Best Score':<10}")
            print("-" * 40)
            for row in scores_data:
                print(f"{row['Section']:<12} {row['Best Version']:<15} {row['Best Score']:<10.1f}")
        
        print("="*80)
        
        print("\n🏆 Best Version for Each Section:")
        for row in scores_data:
            print(f"   {row['Section']}: {row['Best Version']} (Score: {row['Best Score']:.1f})")
        
        # Generate best compilation
        best_document_html = compile_best_document(
            all_sections, scores_data, args.topic, args.model, args.temperature
        )
        
        # Save the best compilation
        best_output_path = Path(output_dir) / f'{topic_filename}_best_compilation.html'
        with open(best_output_path, 'w', encoding='utf-8') as f:
            f.write(best_document_html)
        
        # Save command file for the best compilation
        command_file = CommandTracker.save_command_file(str(best_output_path))
        if command_file:
            print(f"✓ Command saved: {Path(command_file).name}")
        
        # Generate analysis report
        report_content = generate_analysis_report(
            args.topic, args.model, args.temperature, all_sections, scores_data
        )
        report_path = Path(output_dir) / f'{topic_filename}_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n📄 Analysis complete!")
        print(f"✅ Best compilation: {best_output_path}")
        print(f"📊 Analysis report: {report_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())