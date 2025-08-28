#!/usr/bin/env python3
"""
Token Machine Agent Usage Examples

This script demonstrates how to use the token_machine agent for token analysis
and optimization recommendations in doc-generator operations.
"""

import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from doc_generator.agents import (
    TokenMachine,
    AnalysisDepth,
    analyze_operation,
    estimate_cost,
    get_cheapest_model
)


def example_basic_analysis():
    """Basic token analysis example."""
    print("=" * 60)
    print("BASIC TOKEN ANALYSIS EXAMPLE")
    print("=" * 60)
    
    # Initialize token machine
    token_machine = TokenMachine()
    
    # Analyze a documentation generation task
    analysis = token_machine.analyze(
        operation="Generate Python API documentation",
        content="def calculate_fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)",
        content_type="code",
        expected_output_size=2000  # Expected chars in output
    )
    
    print(f"Operation: {analysis.operation}")
    print(f"Estimated Tokens: {analysis.token_estimate.total_tokens:,}")
    print(f"Confidence: {analysis.token_estimate.confidence:.1%}")
    print()
    
    print("Cost Estimates:")
    for estimate in analysis.cost_estimates[:3]:  # Show top 3
        print(f"  {estimate.model_name:20} ${estimate.total_cost:6.3f} (Quality: {estimate.quality_score:.1%})")
    
    print(f"\nRecommended Model: {analysis.recommended_model}")
    
    if analysis.warnings:
        print("\nWarnings:")
        for warning in analysis.warnings:
            print(f"  ⚠️  {warning}")
    
    print(f"\nTop Optimization: {analysis.optimization_strategies[0].name}")
    print(f"Potential Savings: {analysis.optimization_strategies[0].potential_savings:.1%}")
    print()


def example_comprehensive_analysis():
    """Comprehensive analysis with all optimization strategies."""
    print("=" * 60)
    print("COMPREHENSIVE ANALYSIS EXAMPLE")
    print("=" * 60)
    
    token_machine = TokenMachine(analysis_depth=AnalysisDepth.COMPREHENSIVE)
    
    # Large documentation generation task
    large_content = """
    # Machine Learning Pipeline Documentation
    
    This module provides a comprehensive machine learning pipeline for training,
    evaluation, and deployment of ML models. It includes data preprocessing,
    feature engineering, model selection, and monitoring capabilities.
    
    ## Classes and Functions
    """ + "# Code example\n" * 100  # Simulate large content
    
    analysis = token_machine.analyze(
        operation="Generate comprehensive ML pipeline documentation",
        content=large_content,
        content_type="markdown",
        context_size=5000,  # Additional context
        depth=AnalysisDepth.COMPREHENSIVE
    )
    
    print(f"Operation: {analysis.operation}")
    print(f"Total Tokens: {analysis.token_estimate.total_tokens:,}")
    print()
    
    print("Token Breakdown:")
    for component, tokens in analysis.token_estimate.breakdown.items():
        print(f"  {component:15} {tokens:6,} tokens")
    print()
    
    print("All Optimization Strategies:")
    for i, strategy in enumerate(analysis.optimization_strategies, 1):
        print(f"{i}. {strategy.name}")
        print(f"   Savings: {strategy.potential_savings:.1%}")
        print(f"   Effort: {strategy.implementation_effort}")
        print(f"   Recommended: {'Yes' if strategy.recommended else 'No'}")
        if strategy.implementation_steps:
            print(f"   Steps: {len(strategy.implementation_steps)} implementation steps")
        print()


def example_cost_comparison():
    """Compare costs across different models."""
    print("=" * 60)
    print("COST COMPARISON EXAMPLE")
    print("=" * 60)
    
    operations = [
        ("Simple README generation", "Generate README for Python package", 1000),
        ("Complex API documentation", "Generate comprehensive API docs", 5000),
        ("Multi-run analysis", "Quality analysis with 3 runs", 15000),
    ]
    
    token_machine = TokenMachine()
    
    for name, operation, expected_tokens in operations:
        print(f"\n{name}:")
        print(f"Expected tokens: {expected_tokens:,}")
        
        # Get costs for popular models
        models_to_check = ["gpt-4o-mini", "gpt-4o", "claude-3-5-haiku", "claude-3-5-sonnet"]
        
        for model in models_to_check:
            try:
                cost = token_machine.calculate_cost(expected_tokens, model, is_input=True)
                quality = token_machine.MODELS[model].quality_score
                print(f"  {model:20} ${cost:6.3f} (Quality: {quality:.1%})")
            except ValueError:
                print(f"  {model:20} Not available")
        
        # Get recommendation
        recommended = token_machine.recommend_model(max_tokens=expected_tokens)
        if recommended:
            print(f"  💡 Recommended: {recommended}")


def example_usage_report():
    """Example of generating usage reports."""
    print("=" * 60)
    print("USAGE REPORT EXAMPLE")
    print("=" * 60)
    
    token_machine = TokenMachine()
    
    # Simulate multiple operations
    operations = [
        "Generate Python documentation",
        "Generate API reference",
        "Create README file",
        "Generate Python documentation",  # Repeat for caching opportunity
        "Quality analysis report",
        "Generate API reference",  # Repeat
    ]
    
    for operation in operations:
        token_machine.analyze(operation)
    
    # Generate report
    report = token_machine.generate_report(period_days=1)
    
    print("Usage Report:")
    print(json.dumps(report, indent=2))


def example_quick_functions():
    """Examples using convenience functions."""
    print("=" * 60)
    print("QUICK FUNCTIONS EXAMPLE")
    print("=" * 60)
    
    # Quick analysis
    result = analyze_operation(
        "Generate installation guide",
        content="pip install doc-generator"
    )
    
    print("Quick Analysis Result:")
    print(json.dumps(result, indent=2))
    print()
    
    # Quick cost estimation
    text = "This is a sample documentation that we want to estimate costs for. " * 20
    cost = estimate_cost(text, "gpt-4o-mini")
    print(f"Cost estimate for text: ${cost:.4f}")
    print()
    
    # Model recommendation
    cheapest = get_cheapest_model(max_tokens=50000, min_quality=0.8)
    print(f"Cheapest high-quality model: {cheapest}")


def example_integration_pattern():
    """Example of how to integrate token_machine into doc-generator workflow."""
    print("=" * 60)
    print("INTEGRATION PATTERN EXAMPLE")
    print("=" * 60)
    
    def generate_with_token_analysis(topic: str, content: str = None):
        """
        Example function showing integration pattern.
        
        This demonstrates how to integrate token analysis into the
        existing doc-generator workflow.
        """
        # Initialize token machine
        token_machine = TokenMachine()
        
        # Pre-generation analysis
        print("🔍 Analyzing token requirements...")
        analysis = token_machine.analyze(
            operation=f"Generate documentation for: {topic}",
            content=content,
            content_type="plain_text" if not content else "code"
        )
        
        # Show user the analysis
        print(f"📊 Estimated tokens: {analysis.token_estimate.total_tokens:,}")
        print(f"💰 Estimated cost: ${analysis.cost_estimates[0].total_cost:.3f} - ${analysis.cost_estimates[-1].total_cost:.3f}")
        print(f"🎯 Recommended model: {analysis.recommended_model}")
        
        # Check for warnings
        if analysis.warnings:
            print("\n⚠️  Warnings:")
            for warning in analysis.warnings:
                print(f"   {warning}")
        
        # Show optimization opportunities
        if analysis.optimization_strategies:
            best_optimization = analysis.optimization_strategies[0]
            if best_optimization.recommended:
                print(f"\n💡 Optimization opportunity: {best_optimization.name}")
                print(f"   Potential savings: {best_optimization.potential_savings:.1%}")
        
        # Here you would continue with actual generation...
        print("\n🚀 Would proceed with generation using recommended model...")
        
        return analysis
    
    # Example usage
    sample_code = """
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    """
    
    analysis = generate_with_token_analysis(
        topic="Fibonacci Algorithm Documentation",
        content=sample_code
    )


if __name__ == "__main__":
    print("TOKEN MACHINE AGENT - USAGE EXAMPLES")
    print("=" * 60)
    
    try:
        example_basic_analysis()
        example_comprehensive_analysis()
        example_cost_comparison()
        example_usage_report()
        example_quick_functions()
        example_integration_pattern()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()