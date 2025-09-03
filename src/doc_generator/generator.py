"""
Documentation Generator Module

Handles the core documentation generation functionality with LLM providers.
Extracted from core.py for better modularity and single responsibility principle.
"""

import os
import re
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI

from .plugin_manager import PluginManager
from .providers import ProviderManager, CompletionRequest
from .config import get_settings
from .exceptions import ConfigurationError, ProviderError
from .error_handler import ErrorHandler
from .cache import cached, get_cache_manager
from .command_tracker import CommandTracker


class DocumentationGenerator:
    """
    Main class for generating documentation using multiple LLM providers with plugin support.
    
    Follows dependency injection patterns for better testability and modularity.
    """
    
    def __init__(
        self, 
        settings=None,
        provider_manager=None,
        plugin_manager=None,
        error_handler=None,
        logger: Optional[logging.Logger] = None,
        # Legacy parameters for backward compatibility
        prompt_yaml_path: str = None, 
        shots_dir: str = None,
        terminology_path: str = None, 
        provider: str = 'auto'
    ):
        """
        Initialize the documentation generator with configuration and dependencies.
        
        Args:
            settings: Configuration settings (injected dependency)
            provider_manager: LLM provider manager (injected dependency)
            plugin_manager: Plugin manager for recommendations (injected dependency)
            error_handler: Error handling service (injected dependency)
            logger: Logger instance (injected dependency)
            prompt_yaml_path: Legacy parameter for backward compatibility
            shots_dir: Legacy parameter for backward compatibility
            terminology_path: Legacy parameter for backward compatibility
            provider: Provider selection string
        """
        # Dependency injection with fallback to default instances
        self.settings = settings or get_settings()
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize error handler with dependency injection
        if error_handler:
            self.error_handler = error_handler
        else:
            self.error_handler = ErrorHandler(
                max_retries=self.settings.performance.retry_max_attempts,
                backoff_factor=self.settings.performance.retry_backoff_factor,
                logger=self.logger
            )
        
        # Initialize provider manager with dependency injection
        if provider_manager:
            self.provider_manager = provider_manager
        else:
            self.provider_manager = ProviderManager(logger=self.logger)
        
        # Use settings with fallback to provided values for backward compatibility
        self.prompt_yaml_path = Path(
            prompt_yaml_path or self.settings.paths.prompts_dir / 'generator' / 'default.yaml'
        )
        self.shots_dir = Path(shots_dir or self.settings.paths.shots_dir)
        self.terminology_path = Path(
            terminology_path or self.settings.paths.terminology_path
        )
        
        # Set default provider with validation
        self._setup_provider(provider)
        
        # Load configurations with caching
        self.prompt_config = self._load_prompt_config()
        self.terminology = self._load_terminology()
        self.examples = self._load_examples()
        
        # Initialize plugin manager with dependency injection
        if plugin_manager:
            self.plugin_manager = plugin_manager
        else:
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
    
    @classmethod
    def create_with_dependencies(
        cls,
        provider: str = 'auto',
        logger: Optional[logging.Logger] = None
    ) -> 'DocumentationGenerator':
        """
        Factory method to create DocumentationGenerator with properly injected dependencies.
        
        This method follows the factory pattern to ensure all dependencies are properly
        configured and injected.
        
        Args:
            provider: Provider selection string
            logger: Optional logger instance
            
        Returns:
            Configured DocumentationGenerator instance
        """
        # Initialize core dependencies
        settings = get_settings()
        logger = logger or logging.getLogger(__name__)
        
        # Create error handler
        error_handler = ErrorHandler(
            max_retries=settings.performance.retry_max_attempts,
            backoff_factor=settings.performance.retry_backoff_factor,
            logger=logger
        )
        
        # Create provider manager
        provider_manager = ProviderManager(logger=logger)
        
        # Load terminology for plugin manager
        terminology_path = settings.paths.terminology_path
        terminology = {}
        if terminology_path.exists():
            try:
                with open(terminology_path, 'r', encoding='utf-8') as f:
                    terminology = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load terminology: {e}")
        
        # Create plugin manager
        plugin_manager = PluginManager(
            terminology=terminology,
            logger=logger
        )
        plugin_manager.load_plugins()
        
        # Create generator with injected dependencies
        return cls(
            settings=settings,
            provider_manager=provider_manager,
            plugin_manager=plugin_manager,
            error_handler=error_handler,
            logger=logger,
            provider=provider
        )
    
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
        
    def _load_prompt_config(self) -> dict:
        """Load prompt configuration with caching and validation."""
        # Generate cache key that includes the file path to avoid conflicts
        cache_key = f"prompt_config_{hash(str(self.prompt_yaml_path))}"
        
        # Use manual caching to include file path in key
        cache_manager = get_cache_manager()
        cached_result = cache_manager.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        if not self.prompt_yaml_path.exists():
            self.logger.warning(
                f"Prompt config not found at {self.prompt_yaml_path}, using defaults"
            )
            result = {
                'system_prompt': 'You are a technical documentation expert.',
                'documentation_structure': [
                    'Description', 'Installation', 'Usage', 'Examples', 'References'
                ]
            }
            cache_manager.set(cache_key, result)
            return result
            
        try:
            with open(self.prompt_yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # Validate required fields
            if 'system_prompt' not in config:
                raise ConfigurationError(
                    "Missing 'system_prompt' in configuration",
                    context={'config_path': str(self.prompt_yaml_path)}
                )
                
            cache_manager.set(cache_key, config)
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