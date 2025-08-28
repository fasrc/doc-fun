"""
doc-generator agents module

This module contains specialized agents for various aspects of documentation generation
and analysis. Each agent provides focused capabilities for specific use cases.
"""

from .token_machine import (
    TokenMachine,
    TokenAnalysis,
    TokenEstimate,
    CostEstimate,
    OptimizationStrategy,
    AnalysisDepth,
    analyze_operation,
    estimate_cost,
    get_cheapest_model,
)

__all__ = [
    'TokenMachine',
    'TokenAnalysis', 
    'TokenEstimate',
    'CostEstimate',
    'OptimizationStrategy',
    'AnalysisDepth',
    'analyze_operation',
    'estimate_cost',
    'get_cheapest_model',
]

# Version information
__version__ = "1.0.0"
__author__ = "doc-generator team"