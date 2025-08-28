# Core API

The core module provides the primary classes for AI-powered documentation generation, quality analysis, and evaluation. This is the foundation of the doc-generator system.

## DocumentationGenerator

The main class for generating AI-powered documentation with plugin support and multi-provider capabilities.

### Overview

`DocumentationGenerator` is the central API for creating technical documentation using multiple LLM providers (OpenAI, Claude) with intelligent plugin recommendations.

### Key Features

- **Multi-Provider Support**: Automatic provider selection and fallback
- **Plugin Integration**: Extensible recommendation system
- **Quality Control**: Built-in analysis and evaluation capabilities
- **Format Flexibility**: HTML, Markdown, and custom output formats
- **Configuration Management**: YAML-based prompt and terminology configuration

### Basic Usage

```python
from doc_generator import DocumentationGenerator

# Initialize with default settings
generator = DocumentationGenerator()

# Generate documentation for a topic
results = generator.generate_documentation("Python Machine Learning")
print(f"Generated {len(results)} documentation variants")

# Access the generated content
for result in results:
    print(f"File: {result['filename']}")
    print(f"Content length: {len(result['content'])} characters")
```

### Advanced Configuration

```python
from doc_generator import DocumentationGenerator
import logging

# Custom logger for detailed monitoring
logger = logging.getLogger("my_doc_generator")
logger.setLevel(logging.DEBUG)

# Initialize with custom configuration
generator = DocumentationGenerator(
    prompt_yaml_path="./custom-prompts/technical.yaml",
    terminology_path="./config/modules.yaml",
    provider='claude',  # Force specific provider
    logger=logger
)

# Generate multiple variants with analysis
results = generator.generate_documentation(
    query="CUDA Programming with Python",
    runs=3,
    model="claude-3-5-sonnet-20241022",
    temperature=0.2,
    analyze=True,
    quality_eval=True
)

# Process results with error handling
for i, result in enumerate(results, 1):
    try:
        if result.get('error'):
            logger.error(f"Variant {i} failed: {result['error']}")
            continue
            
        # Save to file
        output_file = f"cuda_programming_v{i}.html"
        with open(output_file, 'w') as f:
            f.write(result['content'])
            
        logger.info(f"Saved variant {i}: {output_file}")
        
        # Display quality metrics if available
        if 'analysis' in result:
            score = result['analysis'].get('overall_score', 'N/A')
            logger.info(f"Quality score: {score}")
            
    except Exception as e:
        logger.error(f"Error processing variant {i}: {e}")
```

### Provider and Model Selection

```python
# Auto-select optimal provider and model
generator = DocumentationGenerator(provider='auto')

# Manual provider selection with specific models
openai_generator = DocumentationGenerator(provider='openai')
results_gpt = openai_generator.generate_documentation(
    query="Database Design Patterns",
    model="gpt-4o-mini",  # Cost-effective option
    temperature=0.1       # Focused technical output
)

claude_generator = DocumentationGenerator(provider='claude')
results_claude = claude_generator.generate_documentation(
    query="API Development Best Practices",
    model="claude-3-5-sonnet-20241022",  # Excellent for code documentation
    temperature=0.3
)

# Compare results between providers
print(f"OpenAI result length: {len(results_gpt[0]['content'])}")
print(f"Claude result length: {len(results_claude[0]['content'])}")
```

### Plugin Integration

```python
# Disable specific plugins
generator = DocumentationGenerator()
generator.plugin_manager.disable_plugins(['modules'])

# Enable only specific plugins
generator.plugin_manager.enable_only(['modules', 'datasets'])

# Get plugin recommendations before generation
recommendations = generator.plugin_manager.get_recommendations(
    "Python Data Science"
)

print("Available recommendations:")
for plugin_name, recs in recommendations.items():
    print(f"  {plugin_name}: {len(recs)} suggestions")
    for rec in recs[:3]:  # Show first 3
        print(f"    - {rec.get('title', 'N/A')}")
```

### Error Handling and Debugging

```python
import logging
from doc_generator.exceptions import DocGeneratorError, ProviderError

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("doc_generator")

generator = DocumentationGenerator(logger=logger)

try:
    results = generator.generate_documentation(
        query="Machine Learning Deployment",
        runs=2,
        model="gpt-4o-mini",
        temperature=0.3
    )
    
    if not results:
        logger.warning("No results generated - check API keys and configuration")
        
except ProviderError as e:
    logger.error(f"Provider error: {e}")
    print("Please check your API key configuration")
    
except DocGeneratorError as e:
    logger.error(f"Generation error: {e}")
    print("Check your input parameters and configuration files")
    
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    print("Check logs for detailed error information")
```

### Performance Optimization

```python
# Optimized for speed (development/testing)
fast_generator = DocumentationGenerator()
results = fast_generator.generate_documentation(
    query="Quick Test Documentation",
    runs=1,                     # Single run
    model="gpt-4o-mini",       # Fast, cost-effective
    temperature=0.2,           # Low temperature for speed
    analyze=False              # Skip analysis for speed
)

# Optimized for quality (production)
quality_generator = DocumentationGenerator()
results = quality_generator.generate_documentation(
    query="Critical Production Documentation",
    runs=5,                    # Multiple variants
    model="gpt-4o",           # High-quality model
    temperature=0.1,          # Focused output
    analyze=True,             # Full analysis
    quality_eval=True         # GPT-based evaluation
)
```

### Integration with Analysis Pipeline

```python
from doc_generator import DocumentationGenerator, DocumentAnalyzer, GPTQualityEvaluator

# Initialize components
generator = DocumentationGenerator()
analyzer = DocumentAnalyzer()
evaluator = GPTQualityEvaluator()

# Generate documentation
results = generator.generate_documentation(
    query="Microservices Architecture Patterns",
    runs=3,
    model="claude-3-5-sonnet-20241022"
)

# Analyze each variant
analyzed_results = []
for i, result in enumerate(results):
    # Structural analysis
    analysis = analyzer.analyze_document(result['content'])
    
    # Quality evaluation  
    quality = evaluator.evaluate_quality(result['content'])
    
    analyzed_results.append({
        'variant': i + 1,
        'content': result['content'],
        'filename': result['filename'],
        'structural_score': analysis['overall_score'],
        'quality_score': quality['overall_score'],
        'recommendations': analysis.get('recommendations', [])
    })

# Select best variant
best_variant = max(analyzed_results, key=lambda x: x['quality_score'])
print(f"Best variant: {best_variant['variant']}")
print(f"Quality score: {best_variant['quality_score']}")

# Save best result
with open("microservices_best.html", "w") as f:
    f.write(best_variant['content'])
```

::: doc_generator.core.DocumentationGenerator
    options:
      show_source: true
      show_root_heading: true
      show_root_members_full_path: false
      show_category_heading: true
      heading_level: 3

## DocumentAnalyzer

Advanced document analysis engine that provides algorithmic scoring, structural validation, and quality metrics for generated documentation.

### Overview

`DocumentAnalyzer` performs comprehensive analysis of documentation content, providing objective quality scores based on structure, completeness, readability, and technical accuracy.

### Key Features

- **Structural Analysis**: Evaluates document organization and hierarchy
- **Content Scoring**: Measures completeness and technical depth
- **Readability Assessment**: Analyzes clarity and accessibility
- **Best Practices Validation**: Checks adherence to documentation standards
- **Comparative Analysis**: Benchmarks against reference documentation

### Basic Usage

```python
from doc_generator import DocumentAnalyzer

# Initialize analyzer
analyzer = DocumentAnalyzer()

# Analyze a documentation file
with open("my-documentation.html", "r") as f:
    content = f.read()

# Get comprehensive analysis
analysis = analyzer.analyze_document(content)

print(f"Overall Score: {analysis['overall_score']}/100")
print(f"Structure Score: {analysis['structure_score']}/100")
print(f"Content Score: {analysis['content_score']}/100")
print(f"Recommendations: {len(analysis['recommendations'])}")

# Display recommendations
for rec in analysis['recommendations']:
    print(f"  - {rec['type']}: {rec['message']}")
```

### Detailed Analysis Workflow

```python
from doc_generator import DocumentationGenerator, DocumentAnalyzer
import json

# Generate documentation variants
generator = DocumentationGenerator()
results = generator.generate_documentation(
    query="REST API Design Principles",
    runs=3,
    model="gpt-4o-mini"
)

# Analyze each variant
analyzer = DocumentAnalyzer()
analysis_results = []

for i, result in enumerate(results):
    print(f"\n--- Analyzing Variant {i+1} ---")
    
    # Perform analysis
    analysis = analyzer.analyze_document(result['content'])
    
    # Extract key metrics
    metrics = {
        'variant': i + 1,
        'filename': result['filename'],
        'overall_score': analysis['overall_score'],
        'scores': {
            'structure': analysis['structure_score'],
            'content': analysis['content_score'],
            'readability': analysis['readability_score'],
            'completeness': analysis['completeness_score']
        },
        'word_count': analysis.get('word_count', 0),
        'section_count': analysis.get('section_count', 0),
        'code_examples': analysis.get('code_example_count', 0),
        'recommendations': analysis['recommendations'][:5]  # Top 5 recommendations
    }
    
    analysis_results.append(metrics)
    
    # Display summary
    print(f"Overall Score: {metrics['overall_score']}")
    print(f"Structure: {metrics['scores']['structure']}")
    print(f"Content: {metrics['scores']['content']}")
    print(f"Word Count: {metrics['word_count']}")
    print(f"Sections: {metrics['section_count']}")
    print(f"Code Examples: {metrics['code_examples']}")

# Find best variant
best_variant = max(analysis_results, key=lambda x: x['overall_score'])
print(f"\nBest variant: {best_variant['variant']} (Score: {best_variant['overall_score']})")

# Save analysis report
with open("analysis_report.json", "w") as f:
    json.dump(analysis_results, f, indent=2)
```

### Section-by-Section Analysis

```python
from doc_generator import DocumentAnalyzer

analyzer = DocumentAnalyzer()

# Load documentation content
with open("technical_documentation.html", "r") as f:
    content = f.read()

# Get detailed section analysis
section_analysis = analyzer.analyze_sections(content)

print("Section Analysis:")
print("=" * 50)

for section in section_analysis:
    print(f"Section: {section['title']}")
    print(f"  Score: {section['score']}/100")
    print(f"  Word Count: {section['word_count']}")
    print(f"  Issues:")
    for issue in section.get('issues', []):
        print(f"    - {issue['severity']}: {issue['message']}")
    print()

# Identify problematic sections
low_scoring_sections = [
    s for s in section_analysis 
    if s['score'] < 70
]

if low_scoring_sections:
    print("Sections needing improvement:")
    for section in low_scoring_sections:
        print(f"  - {section['title']}: {section['score']}/100")
```

### Quality Metrics and Benchmarking

```python
from doc_generator import DocumentAnalyzer, DocumentationGenerator
import statistics

# Generate multiple documentation variants
generator = DocumentationGenerator()
analyzer = DocumentAnalyzer()

# Test different models and compare quality
models_to_test = [
    "gpt-4o-mini",
    "gpt-4o",
    "claude-3-5-sonnet-20241022"
]

results_by_model = {}

for model in models_to_test:
    print(f"Testing model: {model}")
    
    # Generate documentation
    docs = generator.generate_documentation(
        query="Database Normalization Techniques",
        runs=3,
        model=model,
        temperature=0.2
    )
    
    # Analyze quality
    scores = []
    for doc in docs:
        analysis = analyzer.analyze_document(doc['content'])
        scores.append(analysis['overall_score'])
    
    results_by_model[model] = {
        'scores': scores,
        'average': statistics.mean(scores),
        'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0,
        'best': max(scores),
        'worst': min(scores)
    }

# Display comparison
print("\nModel Performance Comparison:")
print("=" * 60)
for model, stats in results_by_model.items():
    print(f"{model}:")
    print(f"  Average Score: {stats['average']:.1f}")
    print(f"  Best Score: {stats['best']}")
    print(f"  Consistency (std dev): {stats['std_dev']:.1f}")
    print()

# Recommend best model
best_model = max(
    results_by_model.items(), 
    key=lambda x: x[1]['average']
)
print(f"Recommended model: {best_model[0]} (avg: {best_model[1]['average']:.1f})")
```

### Integration with External Standards

```python
from doc_generator import DocumentAnalyzer

class CustomAnalyzer(DocumentAnalyzer):
    """Extended analyzer with organization-specific standards."""
    
    def __init__(self):
        super().__init__()
        # Add custom quality criteria
        self.required_sections = [
            "Overview",
            "Prerequisites", 
            "Installation",
            "Usage Examples",
            "API Reference",
            "Troubleshooting"
        ]
        self.min_code_examples = 3
        self.min_word_count = 500
        
    def analyze_custom_standards(self, content: str) -> dict:
        """Analyze against organization standards."""
        base_analysis = self.analyze_document(content)
        
        # Check required sections
        missing_sections = []
        for required in self.required_sections:
            if required.lower() not in content.lower():
                missing_sections.append(required)
        
        # Count code examples
        code_count = content.count('<code>') + content.count('```')
        
        # Custom scoring
        custom_score = base_analysis['overall_score']
        if missing_sections:
            custom_score -= len(missing_sections) * 10
        if code_count < self.min_code_examples:
            custom_score -= (self.min_code_examples - code_count) * 5
        
        return {
            **base_analysis,
            'custom_score': max(0, custom_score),
            'missing_sections': missing_sections,
            'code_example_count': code_count,
            'meets_standards': (
                not missing_sections and 
                code_count >= self.min_code_examples
            )
        }

# Use custom analyzer
analyzer = CustomAnalyzer()
with open("documentation.html", "r") as f:
    content = f.read()

analysis = analyzer.analyze_custom_standards(content)

print(f"Standard Score: {analysis['overall_score']}/100")
print(f"Custom Score: {analysis['custom_score']}/100")
print(f"Meets Standards: {analysis['meets_standards']}")

if analysis['missing_sections']:
    print("Missing required sections:")
    for section in analysis['missing_sections']:
        print(f"  - {section}")
```

### Batch Analysis and Reporting

```python
from doc_generator import DocumentAnalyzer
import glob
import pandas as pd
from pathlib import Path

analyzer = DocumentAnalyzer()

# Analyze all documentation files in a directory
doc_files = glob.glob("docs/**/*.html", recursive=True)
analysis_results = []

print(f"Analyzing {len(doc_files)} documentation files...")

for doc_file in doc_files:
    print(f"Processing: {doc_file}")
    
    try:
        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        analysis = analyzer.analyze_document(content)
        
        result = {
            'filename': Path(doc_file).name,
            'path': doc_file,
            'overall_score': analysis['overall_score'],
            'structure_score': analysis['structure_score'],
            'content_score': analysis['content_score'],
            'word_count': analysis.get('word_count', 0),
            'section_count': analysis.get('section_count', 0),
            'code_examples': analysis.get('code_example_count', 0),
            'recommendation_count': len(analysis['recommendations'])
        }
        
        analysis_results.append(result)
        
    except Exception as e:
        print(f"Error analyzing {doc_file}: {e}")

# Create summary report
df = pd.DataFrame(analysis_results)

print("\nAnalysis Summary:")
print(f"Total files analyzed: {len(df)}")
print(f"Average overall score: {df['overall_score'].mean():.1f}")
print(f"Score range: {df['overall_score'].min():.1f} - {df['overall_score'].max():.1f}")

# Identify files needing improvement
low_quality_docs = df[df['overall_score'] < 70]
if not low_quality_docs.empty:
    print("\nFiles needing improvement (score < 70):")
    for _, row in low_quality_docs.iterrows():
        print(f"  {row['filename']}: {row['overall_score']:.1f}")

# Save detailed report
df.to_csv("documentation_analysis_report.csv", index=False)
print("\nDetailed report saved to: documentation_analysis_report.csv")
```

::: doc_generator.core.DocumentAnalyzer
    options:
      show_source: true
      show_root_heading: true
      show_root_members_full_path: false
      show_category_heading: true
      heading_level: 3

## GPTQualityEvaluator

::: doc_generator.core.GPTQualityEvaluator
    options:
      show_source: true
      show_root_heading: true
      show_root_members_full_path: false
      show_category_heading: true
      heading_level: 3

## CodeExampleScanner

::: doc_generator.core.CodeExampleScanner
    options:
      show_source: true
      show_root_heading: true
      show_root_members_full_path: false
      show_category_heading: true
      heading_level: 3