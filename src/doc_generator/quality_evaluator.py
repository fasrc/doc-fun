"""
Quality Evaluation Module

Provides AI-based quality assessment for generated documentation.
Extracted from core.py following single responsibility principle.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .config import get_settings
from .providers import ProviderManager, CompletionRequest
from .cache import cached
from .exceptions import DocGeneratorError, ProviderError
from .error_handler import ErrorHandler


@dataclass
class QualityMetrics:
    """Quality evaluation metrics."""
    overall_score: float
    clarity_score: float
    completeness_score: float
    accuracy_score: float
    usefulness_score: float
    feedback: str
    recommendations: List[str]


class GPTQualityEvaluator:
    """
    AI-based quality evaluator for documentation.
    
    Uses LLM providers to assess documentation quality across
    multiple dimensions including clarity, completeness, and usefulness.
    """
    
    def __init__(
        self,
        provider_manager: Optional[ProviderManager] = None,
        analysis_prompt_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize quality evaluator.
        
        Args:
            provider_manager: LLM provider manager instance
            analysis_prompt_path: Path to analysis prompt configuration
            logger: Optional logger instance
        """
        self.settings = get_settings()
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize provider manager
        self.provider_manager = provider_manager or ProviderManager(logger=self.logger)
        
        # Setup error handler
        self.error_handler = ErrorHandler(
            max_retries=self.settings.performance.retry_max_attempts,
            backoff_factor=self.settings.performance.retry_backoff_factor,
            logger=self.logger
        )
        
        # Load analysis configuration
        self.analysis_prompt_path = Path(
            analysis_prompt_path or 
            self.settings.paths.prompts_dir / 'analysis' / 'default.yaml'
        )
        self.analysis_config = self._load_analysis_config()
        
        self.logger.info("GPTQualityEvaluator initialized")
    
    @cached(ttl=86400)  # Cache for 24 hours
    def _load_analysis_config(self) -> Dict[str, Any]:
        """
        Load analysis prompt configuration.
        
        Returns:
            Analysis configuration dictionary
            
        Raises:
            DocGeneratorError: If configuration loading fails
        """
        if not self.analysis_prompt_path.exists():
            self.logger.warning(
                f"Analysis prompt not found at {self.analysis_prompt_path}, using defaults"
            )
            return self._get_default_analysis_config()
        
        try:
            import yaml
            with open(self.analysis_prompt_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # Validate required fields
            if 'system_prompt' not in config:
                raise DocGeneratorError(
                    "Missing 'system_prompt' in analysis configuration",
                    error_code="INVALID_ANALYSIS_CONFIG",
                    context={'config_path': str(self.analysis_prompt_path)}
                )
                
            return config
            
        except Exception as e:
            if isinstance(e, DocGeneratorError):
                raise
            raise DocGeneratorError(
                f"Failed to load analysis configuration: {e}",
                error_code="CONFIG_LOAD_ERROR",
                context={'config_path': str(self.analysis_prompt_path)}
            )
    
    def _get_default_analysis_config(self) -> Dict[str, Any]:
        """Get default analysis configuration."""
        return {
            'system_prompt': """You are a technical documentation quality evaluator. 
            Analyze the provided documentation and rate it across multiple dimensions.
            
            Provide scores (0-10) for:
            - Clarity: How clear and understandable is the content?
            - Completeness: Does it cover all necessary topics?
            - Accuracy: Is the information correct and up-to-date?
            - Usefulness: How helpful is it for the intended audience?
            
            Also provide an overall score and specific recommendations for improvement.""",
            
            'evaluation_criteria': {
                'clarity': 'Clear language, good structure, easy to follow',
                'completeness': 'Covers all necessary topics, no major gaps',
                'accuracy': 'Information is correct and current',
                'usefulness': 'Provides practical value to users'
            }
        }
    
    def create_evaluation_prompt(self, content: str, context: Optional[Dict] = None) -> str:
        """
        Create evaluation prompt for the given content.
        
        Args:
            content: Documentation content to evaluate
            context: Optional context information
            
        Returns:
            Formatted evaluation prompt
        """
        context = context or {}
        
        prompt_parts = [
            "Please evaluate this documentation:",
            "",
            "=== DOCUMENTATION START ===",
            content,
            "=== DOCUMENTATION END ===",
            "",
            "Provide your evaluation in the following JSON format:",
            "{",
            '  "overall_score": 0-10,',
            '  "clarity_score": 0-10,',
            '  "completeness_score": 0-10,',
            '  "accuracy_score": 0-10,',
            '  "usefulness_score": 0-10,',
            '  "feedback": "Overall assessment and key observations",',
            '  "recommendations": ["Specific improvement suggestion 1", "Suggestion 2", ...]',
            "}"
        ]
        
        # Add context if provided
        if context:
            prompt_parts.insert(-8, f"Context: {json.dumps(context, indent=2)}")
            prompt_parts.insert(-8, "")
        
        return "\n".join(prompt_parts)
    
    def parse_gpt_response(self, response: str) -> QualityMetrics:
        """
        Parse GPT evaluation response into QualityMetrics.
        
        Args:
            response: Raw response from GPT
            
        Returns:
            Parsed QualityMetrics object
            
        Raises:
            DocGeneratorError: If response parsing fails
        """
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = [
                'overall_score', 'clarity_score', 'completeness_score',
                'accuracy_score', 'usefulness_score', 'feedback', 'recommendations'
            ]
            
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # Normalize scores to 0-1 range
            return QualityMetrics(
                overall_score=min(1.0, max(0.0, data['overall_score'] / 10.0)),
                clarity_score=min(1.0, max(0.0, data['clarity_score'] / 10.0)),
                completeness_score=min(1.0, max(0.0, data['completeness_score'] / 10.0)),
                accuracy_score=min(1.0, max(0.0, data['accuracy_score'] / 10.0)),
                usefulness_score=min(1.0, max(0.0, data['usefulness_score'] / 10.0)),
                feedback=data['feedback'],
                recommendations=data.get('recommendations', [])
            )
            
        except Exception as e:
            raise DocGeneratorError(
                f"Failed to parse quality evaluation response: {e}",
                error_code="RESPONSE_PARSE_ERROR",
                context={'response': response[:500]}  # Truncate for context
            )
    
    def evaluate_section(
        self,
        content: str,
        section_name: str,
        provider: str = 'auto',
        model: Optional[str] = None
    ) -> QualityMetrics:
        """
        Evaluate a documentation section.
        
        Args:
            content: Section content to evaluate
            section_name: Name of the section being evaluated
            provider: LLM provider to use
            model: Specific model to use
            
        Returns:
            Quality evaluation metrics
            
        Raises:
            DocGeneratorError: If evaluation fails
            ProviderError: If LLM provider fails
        """
        if not content or not content.strip():
            raise DocGeneratorError(
                "Cannot evaluate empty content",
                error_code="EMPTY_CONTENT",
                context={'section_name': section_name}
            )
        
        try:
            # Get provider and model
            if provider == 'auto':
                provider = self.provider_manager.get_default_provider()
                if not provider:
                    raise ProviderError("No LLM providers available for quality evaluation")
            
            llm_provider = self.provider_manager.get_provider(provider)
            if not llm_provider:
                raise ProviderError(
                    f"Provider '{provider}' not available",
                    context={'available_providers': self.provider_manager.get_available_providers()}
                )
            
            if not model:
                model = llm_provider.get_default_model()
            
            # Validate model
            if not self.provider_manager.validate_model_provider_combination(model, provider):
                available_models = llm_provider.get_available_models()
                raise DocGeneratorError(
                    f"Model '{model}' not available for provider '{provider}'",
                    error_code="INVALID_MODEL",
                    context={
                        'model': model,
                        'provider': provider,
                        'available_models': available_models
                    }
                )
            
            # Create evaluation prompt
            context = {
                'section_name': section_name,
                'content_length': len(content),
                'word_count': len(content.split())
            }
            
            prompt = self.create_evaluation_prompt(content, context)
            
            # Create completion request
            request = CompletionRequest(
                messages=[
                    {"role": "system", "content": self.analysis_config['system_prompt']},
                    {"role": "user", "content": prompt}
                ],
                model=model,
                temperature=0.2,  # Low temperature for consistent evaluation
                max_tokens=self.settings.providers.max_tokens
            )
            
            # Generate evaluation with retry logic
            response = self.error_handler.with_retry(
                llm_provider.generate_completion,
                request
            )
            
            # Parse response
            metrics = self.parse_gpt_response(response.content)
            
            self.logger.info(
                f"Quality evaluation complete for '{section_name}': "
                f"overall={metrics.overall_score:.2f}"
            )
            
            return metrics
            
        except Exception as e:
            if isinstance(e, (DocGeneratorError, ProviderError)):
                raise
            raise DocGeneratorError(
                f"Quality evaluation failed: {e}",
                error_code="EVALUATION_ERROR",
                context={'section_name': section_name, 'content_length': len(content)}
            )
    
    def evaluate_file(self, file_path: str, **kwargs) -> QualityMetrics:
        """
        Evaluate a documentation file.
        
        Args:
            file_path: Path to documentation file
            **kwargs: Additional arguments for evaluation
            
        Returns:
            Quality evaluation metrics
            
        Raises:
            DocGeneratorError: If file evaluation fails
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise DocGeneratorError(
                    f"File not found: {file_path}",
                    error_code="FILE_NOT_FOUND",
                    context={'file_path': str(file_path)}
                )
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self.evaluate_section(
                content=content,
                section_name=file_path.stem,
                **kwargs
            )
            
        except Exception as e:
            if isinstance(e, DocGeneratorError):
                raise
            raise DocGeneratorError(
                f"File evaluation failed: {e}",
                error_code="FILE_EVALUATION_ERROR",
                context={'file_path': str(file_path)}
            )
    
    def save_evaluation(self, metrics: QualityMetrics, output_path: str) -> None:
        """
        Save evaluation results to file.
        
        Args:
            metrics: Quality metrics to save
            output_path: Output file path
            
        Raises:
            DocGeneratorError: If saving fails
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            evaluation_data = {
                'overall_score': metrics.overall_score,
                'scores': {
                    'clarity': metrics.clarity_score,
                    'completeness': metrics.completeness_score,
                    'accuracy': metrics.accuracy_score,
                    'usefulness': metrics.usefulness_score
                },
                'feedback': metrics.feedback,
                'recommendations': metrics.recommendations,
                'evaluation_timestamp': str(Path().cwd()),  # Placeholder for timestamp
                'evaluator_version': '2.0'
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Quality evaluation saved to {output_path}")
            
        except Exception as e:
            raise DocGeneratorError(
                f"Failed to save evaluation: {e}",
                error_code="SAVE_ERROR",
                context={'output_path': str(output_path)}
            )