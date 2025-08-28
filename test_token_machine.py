#!/usr/bin/env python3
"""
Test script for token machine functionality
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from doc_generator.agents.token_machine import TokenMachine, AnalysisDepth, analyze_operation
    
    print("✓ Successfully imported token machine components")
    
    # Test basic functionality
    print("\n=== Testing Basic Token Estimation ===")
    machine = TokenMachine()
    
    # Test 1: Simple token estimation
    test_text = "Generate comprehensive Python documentation covering classes, functions, and modules."
    tokens = machine.estimate_tokens(test_text, "plain_text")
    print(f"Text: {test_text}")
    print(f"Estimated tokens: {tokens}")
    
    # Test 2: Cost calculation
    cost = machine.calculate_cost(1000, "gpt-4o-mini", is_input=True)
    print(f"\nCost for 1000 input tokens with gpt-4o-mini: ${cost:.6f}")
    
    # Test 3: Model recommendation
    recommended = machine.recommend_model(max_tokens=5000, min_quality=0.7)
    print(f"Recommended model for 5000 tokens, min quality 0.7: {recommended}")
    
    # Test 4: Quick analysis function
    print("\n=== Testing Quick Analysis Function ===")
    result = analyze_operation(
        "Generate Python documentation for a web scraping library",
        content="A library with 15 classes and 80+ functions for web scraping",
        content_type="plain_text"
    )
    
    print(f"Operation: Generate Python documentation for a web scraping library")
    print(f"Total tokens: {result['tokens']['total']:,}")
    print(f"Cost range: ${result['cost_range']['min']:.4f} - ${result['cost_range']['max']:.4f}")
    print(f"Recommended model: {result['recommended_model']}")
    print(f"Top optimization: {result['top_optimization']}")
    
    if result['warnings']:
        print("Warnings:")
        for warning in result['warnings']:
            print(f"  ⚠️  {warning}")
    
    print("\n=== Testing Comprehensive Analysis ===")
    analysis = machine.analyze(
        operation="Generate README for ML project",
        content="Machine learning project with 5 Python files, 3 Jupyter notebooks, requirements.txt",
        content_type="plain_text",
        depth=AnalysisDepth.COMPREHENSIVE
    )
    
    print(f"Operation: {analysis.operation}")
    print(f"Total tokens: {analysis.token_estimate.total_tokens:,}")
    print(f"Confidence: {analysis.token_estimate.confidence:.1%}")
    
    if analysis.cost_estimates:
        print(f"Cheapest option: {analysis.cost_estimates[0].model_name} (${analysis.cost_estimates[0].total_cost:.4f})")
    
    print(f"Optimization strategies found: {len(analysis.optimization_strategies)}")
    for strategy in analysis.optimization_strategies[:2]:
        print(f"  • {strategy.name}: {strategy.potential_savings:.0%} savings")
    
    if analysis.warnings:
        print("Warnings:")
        for warning in analysis.warnings:
            print(f"  ⚠️  {warning}")
    
    print("\n✅ All token machine tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)