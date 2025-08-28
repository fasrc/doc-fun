"""
Token Machine Agent - Token Usage Analysis and Optimization Specialist

This module provides comprehensive token usage analysis, cost estimation,
and optimization recommendations for doc-generator operations.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import yaml

logger = logging.getLogger(__name__)


class AnalysisDepth(Enum):
    """Analysis depth levels"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class Provider(Enum):
    """LLM Provider enumeration"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model"""
    name: str
    provider: Provider
    context_window: int
    input_cost_per_1k: float
    output_cost_per_1k: float
    quality_score: float = 0.5  # 0-1 scale
    speed_score: float = 0.5    # 0-1 scale


@dataclass
class TokenEstimate:
    """Token usage estimate for an operation"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    confidence: float  # 0-1 scale
    breakdown: Dict[str, int] = field(default_factory=dict)


@dataclass
class CostEstimate:
    """Cost estimate for a specific model"""
    model_name: str
    provider: Provider
    input_cost: float
    output_cost: float
    total_cost: float
    tokens: TokenEstimate
    quality_score: float
    speed_score: float


@dataclass
class OptimizationStrategy:
    """Token optimization strategy"""
    name: str
    description: str
    potential_savings: float  # Percentage
    implementation_effort: str  # low, medium, high
    recommended: bool = False
    implementation_steps: List[str] = field(default_factory=list)


@dataclass
class TokenAnalysis:
    """Complete token usage analysis"""
    operation: str
    timestamp: datetime
    token_estimate: TokenEstimate
    cost_estimates: List[CostEstimate]
    optimization_strategies: List[OptimizationStrategy]
    recommended_model: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TokenMachine:
    """
    Token Machine Agent for comprehensive token usage analysis and optimization.
    """
    
    # Model configurations
    MODELS = {
        "gpt-3.5-turbo": ModelConfig(
            "gpt-3.5-turbo", Provider.OPENAI, 16385, 0.0015, 0.002, 0.7, 0.9
        ),
        "gpt-4": ModelConfig(
            "gpt-4", Provider.OPENAI, 8192, 0.03, 0.06, 0.95, 0.5
        ),
        "gpt-4o": ModelConfig(
            "gpt-4o", Provider.OPENAI, 128000, 0.005, 0.015, 0.9, 0.8
        ),
        "gpt-4o-mini": ModelConfig(
            "gpt-4o-mini", Provider.OPENAI, 128000, 0.00015, 0.0006, 0.75, 0.85
        ),
        "claude-3-5-haiku": ModelConfig(
            "claude-3-5-haiku", Provider.ANTHROPIC, 200000, 0.0008, 0.004, 0.8, 0.9
        ),
        "claude-3-5-sonnet": ModelConfig(
            "claude-3-5-sonnet", Provider.ANTHROPIC, 200000, 0.003, 0.015, 0.92, 0.7
        ),
        "claude-opus-4-1": ModelConfig(
            "claude-opus-4-1", Provider.ANTHROPIC, 200000, 0.015, 0.075, 0.98, 0.6
        ),
    }
    
    # Token estimation multipliers
    CONTENT_TYPE_MULTIPLIERS = {
        "code": 1.2,
        "markdown": 1.1,
        "plain_text": 1.0,
        "json": 1.15,
        "yaml": 1.1,
        "html": 1.2,
    }
    
    def __init__(self, 
                 config_path: Optional[Path] = None,
                 cache_enabled: bool = True,
                 analysis_depth: AnalysisDepth = AnalysisDepth.STANDARD):
        """
        Initialize Token Machine Agent.
        
        Args:
            config_path: Path to custom configuration file
            cache_enabled: Enable caching of analysis results
            analysis_depth: Default analysis depth level
        """
        self.config_path = config_path
        self.cache_enabled = cache_enabled
        self.analysis_depth = analysis_depth
        self._cache: Dict[str, TokenAnalysis] = {}
        self._usage_history: List[TokenAnalysis] = []
        
        # Load custom configuration if provided
        if config_path and config_path.exists():
            self._load_config(config_path)
    
    def analyze(self,
                operation: str,
                content: Optional[str] = None,
                content_type: str = "plain_text",
                context_size: Optional[int] = None,
                expected_output_size: Optional[int] = None,
                depth: Optional[AnalysisDepth] = None) -> TokenAnalysis:
        """
        Perform comprehensive token usage analysis for an operation.
        
        Args:
            operation: Description of the operation
            content: Optional content to analyze
            content_type: Type of content (affects token estimation)
            context_size: Additional context size in characters
            expected_output_size: Expected output size in characters
            depth: Analysis depth (overrides default)
        
        Returns:
            Complete token usage analysis
        """
        depth = depth or self.analysis_depth
        
        # Check cache
        cache_key = self._generate_cache_key(operation, content, content_type)
        if self.cache_enabled and cache_key in self._cache:
            cached = self._cache[cache_key]
            if (datetime.now() - cached.timestamp).seconds < 3600:  # 1 hour cache
                logger.info(f"Returning cached analysis for: {operation}")
                return cached
        
        # Estimate tokens
        token_estimate = self._estimate_tokens(
            content, content_type, context_size, expected_output_size
        )
        
        # Calculate costs for all models
        cost_estimates = self._calculate_costs(token_estimate)
        
        # Generate optimization strategies
        optimization_strategies = self._generate_optimization_strategies(
            token_estimate, operation, depth
        )
        
        # Select recommended model
        recommended_model = self._select_recommended_model(
            cost_estimates, operation
        )
        
        # Generate warnings
        warnings = self._generate_warnings(token_estimate, cost_estimates)
        
        # Create analysis
        analysis = TokenAnalysis(
            operation=operation,
            timestamp=datetime.now(),
            token_estimate=token_estimate,
            cost_estimates=cost_estimates,
            optimization_strategies=optimization_strategies,
            recommended_model=recommended_model,
            warnings=warnings,
            metadata={
                "content_type": content_type,
                "analysis_depth": depth.value,
                "cache_key": cache_key,
            }
        )
        
        # Cache and record
        if self.cache_enabled:
            self._cache[cache_key] = analysis
        self._usage_history.append(analysis)
        
        return analysis
    
    def estimate_tokens(self, text: str, content_type: str = "plain_text") -> int:
        """
        Estimate token count for given text.
        
        Args:
            text: Text to estimate tokens for
            content_type: Type of content
        
        Returns:
            Estimated token count
        """
        # Basic character-based estimation
        base_tokens = len(text) / 4
        
        # Apply content type multiplier
        multiplier = self.CONTENT_TYPE_MULTIPLIERS.get(content_type, 1.0)
        
        return int(base_tokens * multiplier)
    
    def calculate_cost(self, 
                       tokens: int, 
                       model: str,
                       is_input: bool = True) -> float:
        """
        Calculate cost for specific token count and model.
        
        Args:
            tokens: Number of tokens
            model: Model name
            is_input: Whether tokens are input (vs output)
        
        Returns:
            Cost in USD
        """
        if model not in self.MODELS:
            raise ValueError(f"Unknown model: {model}")
        
        model_config = self.MODELS[model]
        cost_per_1k = model_config.input_cost_per_1k if is_input else model_config.output_cost_per_1k
        
        return (tokens / 1000) * cost_per_1k
    
    def recommend_model(self,
                       max_cost: Optional[float] = None,
                       min_quality: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> Optional[str]:
        """
        Recommend best model based on constraints.
        
        Args:
            max_cost: Maximum cost constraint
            min_quality: Minimum quality score (0-1)
            max_tokens: Maximum token requirement
        
        Returns:
            Recommended model name or None if no model fits
        """
        candidates = []
        
        for model_name, config in self.MODELS.items():
            # Check token constraint
            if max_tokens and config.context_window < max_tokens:
                continue
            
            # Check quality constraint
            if min_quality and config.quality_score < min_quality:
                continue
            
            # Check cost constraint (estimate)
            if max_cost:
                est_cost = self.calculate_cost(1000, model_name, True)
                if est_cost > max_cost:
                    continue
            
            candidates.append((model_name, config))
        
        if not candidates:
            return None
        
        # Sort by value (quality/cost ratio)
        candidates.sort(
            key=lambda x: x[1].quality_score / (x[1].input_cost_per_1k + 0.001),
            reverse=True
        )
        
        return candidates[0][0]
    
    def generate_report(self,
                       period_days: int = 7,
                       include_predictions: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive usage report.
        
        Args:
            period_days: Report period in days
            include_predictions: Include future predictions
        
        Returns:
            Usage report dictionary
        """
        cutoff = datetime.now() - timedelta(days=period_days)
        recent_analyses = [
            a for a in self._usage_history 
            if a.timestamp >= cutoff
        ]
        
        if not recent_analyses:
            return {"error": "No usage data available for the specified period"}
        
        # Calculate statistics
        total_tokens = sum(a.token_estimate.total_tokens for a in recent_analyses)
        total_cost = sum(
            min(ce.total_cost for ce in a.cost_estimates)
            for a in recent_analyses
        )
        
        # Model usage distribution
        model_usage = {}
        for analysis in recent_analyses:
            if analysis.recommended_model:
                model_usage[analysis.recommended_model] = \
                    model_usage.get(analysis.recommended_model, 0) + 1
        
        report = {
            "period": {
                "start": cutoff.isoformat(),
                "end": datetime.now().isoformat(),
                "days": period_days,
            },
            "summary": {
                "total_operations": len(recent_analyses),
                "total_tokens": total_tokens,
                "estimated_cost": round(total_cost, 2),
                "avg_tokens_per_operation": total_tokens // len(recent_analyses),
            },
            "model_distribution": model_usage,
            "top_operations": self._get_top_operations(recent_analyses, 5),
            "optimization_opportunities": self._identify_optimization_opportunities(
                recent_analyses
            ),
        }
        
        if include_predictions:
            report["predictions"] = self._generate_predictions(recent_analyses)
        
        return report
    
    def _estimate_tokens(self,
                        content: Optional[str],
                        content_type: str,
                        context_size: Optional[int],
                        expected_output_size: Optional[int]) -> TokenEstimate:
        """Estimate tokens for an operation."""
        input_tokens = 0
        output_tokens = 0
        breakdown = {}
        
        # System prompt estimation
        system_tokens = 500  # Base system prompt
        breakdown["system_prompt"] = system_tokens
        input_tokens += system_tokens
        
        # Content tokens
        if content:
            content_tokens = self.estimate_tokens(content, content_type)
            breakdown["content"] = content_tokens
            input_tokens += content_tokens
        
        # Context tokens
        if context_size:
            context_tokens = context_size // 4
            breakdown["context"] = context_tokens
            input_tokens += context_tokens
        
        # Output estimation
        if expected_output_size:
            output_tokens = expected_output_size // 4
        else:
            # Default estimation based on input
            output_tokens = int(input_tokens * 0.8)
        
        breakdown["output"] = output_tokens
        
        # Calculate confidence
        confidence = 0.9 if content else 0.6
        if expected_output_size:
            confidence += 0.1
        
        return TokenEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            confidence=min(confidence, 1.0),
            breakdown=breakdown
        )
    
    def _calculate_costs(self, token_estimate: TokenEstimate) -> List[CostEstimate]:
        """Calculate costs for all models."""
        cost_estimates = []
        
        for model_name, config in self.MODELS.items():
            # Check if model can handle the tokens
            if token_estimate.total_tokens > config.context_window:
                continue
            
            input_cost = (token_estimate.input_tokens / 1000) * config.input_cost_per_1k
            output_cost = (token_estimate.output_tokens / 1000) * config.output_cost_per_1k
            
            cost_estimates.append(CostEstimate(
                model_name=model_name,
                provider=config.provider,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=input_cost + output_cost,
                tokens=token_estimate,
                quality_score=config.quality_score,
                speed_score=config.speed_score
            ))
        
        # Sort by total cost
        cost_estimates.sort(key=lambda x: x.total_cost)
        
        return cost_estimates
    
    def _generate_optimization_strategies(self,
                                         token_estimate: TokenEstimate,
                                         operation: str,
                                         depth: AnalysisDepth) -> List[OptimizationStrategy]:
        """Generate optimization strategies."""
        strategies = []
        
        # Caching strategy
        if "generation" in operation.lower() or "documentation" in operation.lower():
            strategies.append(OptimizationStrategy(
                name="Response Caching",
                description="Cache generated responses for similar requests",
                potential_savings=0.30,
                implementation_effort="low",
                recommended=True,
                implementation_steps=[
                    "Implement cache key generation based on input hash",
                    "Store responses with TTL of 1-24 hours",
                    "Check cache before making API calls",
                    "Monitor cache hit rate"
                ]
            ))
        
        # Prompt optimization
        if token_estimate.input_tokens > 2000:
            strategies.append(OptimizationStrategy(
                name="Prompt Optimization",
                description="Reduce prompt size while maintaining quality",
                potential_savings=0.20,
                implementation_effort="medium",
                recommended=True,
                implementation_steps=[
                    "Remove redundant instructions",
                    "Use shorter examples",
                    "Compress system prompts",
                    "Test quality impact"
                ]
            ))
        
        # Model selection
        if depth == AnalysisDepth.COMPREHENSIVE:
            strategies.append(OptimizationStrategy(
                name="Dynamic Model Selection",
                description="Use different models based on task complexity",
                potential_savings=0.50,
                implementation_effort="medium",
                recommended=True,
                implementation_steps=[
                    "Classify tasks by complexity",
                    "Use GPT-4o-mini for simple tasks",
                    "Reserve GPT-4o for complex tasks",
                    "Implement fallback logic"
                ]
            ))
        
        # Context pruning
        if token_estimate.breakdown.get("context", 0) > 1000:
            strategies.append(OptimizationStrategy(
                name="Context Pruning",
                description="Remove unnecessary context from requests",
                potential_savings=0.15,
                implementation_effort="medium",
                recommended=False,
                implementation_steps=[
                    "Identify essential context",
                    "Implement smart truncation",
                    "Use summarization for long contexts",
                    "Validate output quality"
                ]
            ))
        
        # Batch processing
        if "batch" not in operation.lower() and depth != AnalysisDepth.MINIMAL:
            strategies.append(OptimizationStrategy(
                name="Batch Processing",
                description="Group similar requests for efficiency",
                potential_savings=0.25,
                implementation_effort="high",
                recommended=False,
                implementation_steps=[
                    "Identify batchable operations",
                    "Implement request queuing",
                    "Create batch processing pipeline",
                    "Handle partial failures"
                ]
            ))
        
        return strategies
    
    def _select_recommended_model(self,
                                 cost_estimates: List[CostEstimate],
                                 operation: str) -> Optional[str]:
        """Select recommended model based on operation and estimates."""
        if not cost_estimates:
            return None
        
        # Default to most cost-effective
        if "test" in operation.lower() or "draft" in operation.lower():
            # Use cheapest for tests/drafts
            return cost_estimates[0].model_name
        
        # Balance quality and cost for production
        best_value = None
        best_score = 0
        
        for estimate in cost_estimates:
            # Value score: quality per dollar
            if estimate.total_cost > 0:
                value_score = estimate.quality_score / (estimate.total_cost + 0.001)
                if value_score > best_score:
                    best_score = value_score
                    best_value = estimate.model_name
        
        return best_value
    
    def _generate_warnings(self,
                         token_estimate: TokenEstimate,
                         cost_estimates: List[CostEstimate]) -> List[str]:
        """Generate warnings based on analysis."""
        warnings = []
        
        # Token warnings
        if token_estimate.total_tokens > 10000:
            warnings.append(f"High token usage: {token_estimate.total_tokens} tokens")
        
        if token_estimate.total_tokens > 50000:
            warnings.append("Critical: Very high token usage may hit rate limits")
        
        # Cost warnings
        if cost_estimates:
            min_cost = cost_estimates[0].total_cost
            if min_cost > 1.0:
                warnings.append(f"High cost operation: minimum ${min_cost:.2f}")
        
        # Context window warnings
        models_that_fit = len(cost_estimates)
        total_models = len(self.MODELS)
        if models_that_fit < total_models / 2:
            warnings.append(f"Only {models_that_fit}/{total_models} models can handle this request")
        
        return warnings
    
    def _generate_cache_key(self,
                           operation: str,
                           content: Optional[str],
                           content_type: str) -> str:
        """Generate cache key for analysis."""
        key_data = f"{operation}:{content_type}"
        if content:
            key_data += f":{len(content)}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _load_config(self, config_path: Path):
        """Load custom configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Update model configurations
                if 'providers' in config:
                    for provider_name, provider_config in config['providers'].items():
                        if 'models' in provider_config:
                            for model in provider_config['models']:
                                # Update or add model configuration
                                self.MODELS[model['name']] = ModelConfig(
                                    name=model['name'],
                                    provider=Provider[provider_name.upper()],
                                    context_window=model.get('context', 8192),
                                    input_cost_per_1k=model.get('input_cost', 0.001),
                                    output_cost_per_1k=model.get('output_cost', 0.002),
                                    quality_score=model.get('quality_score', 0.5),
                                    speed_score=model.get('speed_score', 0.5)
                                )
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
    
    def _get_top_operations(self,
                           analyses: List[TokenAnalysis],
                           limit: int = 5) -> List[Dict[str, Any]]:
        """Get top operations by token usage."""
        operation_stats = {}
        
        for analysis in analyses:
            op = analysis.operation
            if op not in operation_stats:
                operation_stats[op] = {
                    "count": 0,
                    "total_tokens": 0,
                    "avg_cost": 0
                }
            
            operation_stats[op]["count"] += 1
            operation_stats[op]["total_tokens"] += analysis.token_estimate.total_tokens
            
            if analysis.cost_estimates:
                operation_stats[op]["avg_cost"] += analysis.cost_estimates[0].total_cost
        
        # Calculate averages
        for op in operation_stats:
            count = operation_stats[op]["count"]
            operation_stats[op]["avg_tokens"] = operation_stats[op]["total_tokens"] // count
            operation_stats[op]["avg_cost"] = operation_stats[op]["avg_cost"] / count
        
        # Sort by total tokens and return top
        sorted_ops = sorted(
            operation_stats.items(),
            key=lambda x: x[1]["total_tokens"],
            reverse=True
        )
        
        return [
            {"operation": op, **stats}
            for op, stats in sorted_ops[:limit]
        ]
    
    def _identify_optimization_opportunities(self,
                                            analyses: List[TokenAnalysis]) -> List[Dict[str, Any]]:
        """Identify optimization opportunities from usage patterns."""
        opportunities = []
        
        # Check for repeated operations (caching opportunity)
        operation_counts = {}
        for analysis in analyses:
            operation_counts[analysis.operation] = \
                operation_counts.get(analysis.operation, 0) + 1
        
        repeated_ops = [op for op, count in operation_counts.items() if count > 2]
        if repeated_ops:
            opportunities.append({
                "type": "caching",
                "description": f"Implement caching for repeated operations: {', '.join(repeated_ops[:3])}",
                "potential_savings": "30-40%",
                "priority": "high"
            })
        
        # Check for high token usage
        high_token_ops = [
            a.operation for a in analyses
            if a.token_estimate.total_tokens > 10000
        ]
        if high_token_ops:
            opportunities.append({
                "type": "optimization",
                "description": f"Optimize high-token operations: {', '.join(high_token_ops[:3])}",
                "potential_savings": "20-30%",
                "priority": "medium"
            })
        
        return opportunities
    
    def _generate_predictions(self,
                            recent_analyses: List[TokenAnalysis]) -> Dict[str, Any]:
        """Generate future usage predictions."""
        if not recent_analyses:
            return {}
        
        # Calculate daily average
        days = (recent_analyses[-1].timestamp - recent_analyses[0].timestamp).days + 1
        daily_tokens = sum(a.token_estimate.total_tokens for a in recent_analyses) / days
        daily_cost = sum(
            min(ce.total_cost for ce in a.cost_estimates)
            for a in recent_analyses
        ) / days
        
        return {
            "next_week": {
                "estimated_tokens": int(daily_tokens * 7),
                "estimated_cost": round(daily_cost * 7, 2),
            },
            "next_month": {
                "estimated_tokens": int(daily_tokens * 30),
                "estimated_cost": round(daily_cost * 30, 2),
            },
            "growth_trend": "stable",  # Could be enhanced with actual trend analysis
        }


# Convenience functions for integration

def analyze_operation(operation: str,
                     content: Optional[str] = None,
                     **kwargs) -> Dict[str, Any]:
    """
    Quick analysis function for integration.
    
    Args:
        operation: Operation description
        content: Optional content to analyze
        **kwargs: Additional parameters
    
    Returns:
        Analysis results as dictionary
    """
    machine = TokenMachine()
    analysis = machine.analyze(operation, content, **kwargs)
    
    return {
        "operation": analysis.operation,
        "tokens": {
            "input": analysis.token_estimate.input_tokens,
            "output": analysis.token_estimate.output_tokens,
            "total": analysis.token_estimate.total_tokens,
        },
        "cost_range": {
            "min": analysis.cost_estimates[0].total_cost if analysis.cost_estimates else 0,
            "max": analysis.cost_estimates[-1].total_cost if analysis.cost_estimates else 0,
        },
        "recommended_model": analysis.recommended_model,
        "top_optimization": analysis.optimization_strategies[0].name if analysis.optimization_strategies else None,
        "warnings": analysis.warnings,
    }


def estimate_cost(text: str, model: str = "gpt-4o-mini") -> float:
    """
    Quick cost estimation for text and model.
    
    Args:
        text: Text to estimate cost for
        model: Model to use
    
    Returns:
        Estimated cost in USD
    """
    machine = TokenMachine()
    tokens = machine.estimate_tokens(text)
    return machine.calculate_cost(tokens, model, is_input=True)


def get_cheapest_model(max_tokens: int,
                      min_quality: float = 0.7) -> Optional[str]:
    """
    Get cheapest model that meets requirements.
    
    Args:
        max_tokens: Maximum tokens needed
        min_quality: Minimum quality score required
    
    Returns:
        Model name or None if no model fits
    """
    machine = TokenMachine()
    return machine.recommend_model(max_tokens=max_tokens, min_quality=min_quality)