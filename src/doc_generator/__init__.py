"""
doc_generator: AI-powered documentation generator for FASRC

This package provides intelligent documentation generation using OpenAI's GPT models
with specialized recommendation engines for HPC modules, code examples, and more.
"""

__version__ = "1.1.0"

from .core import DocumentationGenerator, DocumentAnalyzer, GPTQualityEvaluator, CodeExampleScanner
from .plugin_manager import PluginManager
from .plugins import RecommendationEngine

__all__ = [
    "DocumentationGenerator",
    "DocumentAnalyzer", 
    "GPTQualityEvaluator",
    "CodeExampleScanner",
    "PluginManager",
    "RecommendationEngine"
]