# API Reference

This section provides comprehensive API documentation for doc-generator, automatically generated from source code docstrings.

## 📋 Overview

The doc-generator API is organized into these main modules:

- **[Core](core.md)** - Main documentation generation classes
- **[Plugins](plugins.md)** - Plugin system and base classes
- **[Extractors](extractors.md)** - Content extraction from various document formats
- **[Standardizers](standardizers.md)** - Document standardization and template application
- **[CLI](cli.md)** - Command-line interface
- **[Evaluator](evaluator.md)** - Document analysis and quality evaluation

## 🎯 Quick Reference

### Main Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `DocumentationGenerator` | `doc_generator.core` | Main API for generating documentation |
| `ReadmeGenerator` | `doc_generator.readme_generator` | Generate README.md files for directories |
| `DocumentStandardizer` | `doc_generator.standardizers` | Standardize documents to organizational templates |
| `HTMLContentExtractor` | `doc_generator.extractors` | Extract content from HTML documents |
| `SectionMapper` | `doc_generator.standardizers` | Map content to standardized section templates |
| `DocumentAnalyzer` | `doc_generator.core` | Analyze and score generated documentation |
| `GPTQualityEvaluator` | `doc_generator.core` | GPT-based quality evaluation |
| `PluginManager` | `doc_generator.plugin_manager` | Manage and execute plugins |
| `RecommendationEngine` | `doc_generator.plugins.base` | Base class for all plugins |

### Common Usage Patterns

#### Basic Generation
```python
from doc_generator import DocumentationGenerator

generator = DocumentationGenerator()
results = generator.generate_documentation("Python Programming")
```

#### With Custom Configuration
```python
generator = DocumentationGenerator(
    prompt_yaml_path="./prompts/custom.yaml",
    terminology_path="./my-terminology.yaml"
)
```

#### Plugin System
```python
from doc_generator import PluginManager

plugin_manager = PluginManager()
recommendations = plugin_manager.get_recommendations("Machine Learning")
```

## 🔗 Navigation

Choose a section to explore detailed API documentation:

<div class="grid cards" markdown>

-   **[Core API](core.md)**

    ---
    
    Main documentation generation classes including `DocumentationGenerator`, `DocumentAnalyzer`, and quality evaluation tools.

-   **[Plugin System](plugins.md)**

    ---
    
    Plugin architecture with `PluginManager`, `RecommendationEngine` base class, and built-in plugins like `ModuleRecommender`.

-   **[CLI Interface](cli.md)**

    ---
    
    Command-line interface implementation with argument parsing, configuration management, and execution logic.

</div>

## 💡 Examples

### Advanced Generation with Analysis

```python
from doc_generator import DocumentationGenerator, DocumentAnalyzer

# Initialize components
generator = DocumentationGenerator()
analyzer = DocumentAnalyzer()

# Generate multiple variants
results = generator.generate_documentation(
    query="Parallel Computing with MPI",
    runs=3,
    model="gpt-4",
    temperature=0.3
)

# Analyze each variant
for result in results:
    scores = analyzer.analyze_document(result['content'])
    print(f"Quality score: {scores['overall_score']}")
```

### Plugin Development

```python
from doc_generator.plugins import RecommendationEngine

class CustomRecommender(RecommendationEngine):
    def get_name(self) -> str:
        return "custom"
    
    def get_recommendations(self, topic: str, context=None):
        # Your recommendation logic
        return [{"title": "Custom Suggestion", "score": 8.5}]
```

## 📖 Related Documentation

- **[Getting Started Guide](../guides/getting-started.md)** - Learn basic usage
- **[Creating Plugins Guide](../guides/creating-plugins.md)** - Build custom plugins
- **[Configuration Guide](../guides/configuration.md)** - Customize behavior
- **[Testing Guide](../guides/testing.md)** - Test your code

---

*API documentation is automatically generated from source code docstrings using [mkdocstrings](https://mkdocstrings.github.io/).*