# Documentation Comparison Guide

The documentation comparison module allows you to evaluate generated documentation against existing reference documentation, providing objective quality metrics and actionable recommendations.

## Overview

The comparison module helps you:
- **Benchmark quality** against gold-standard documentation
- **Identify gaps** in generated content
- **Optimize parameters** based on similarity scores
- **Track improvements** over time

## Quick Start

### Compare with a URL

```bash
# Generate and compare with existing documentation
doc-gen --topic "Python Lists" \
        --compare-url https://docs.python.org/3/tutorial/lists.html \
        --comparison-report comparison.md
```

### Compare with a Local File

```bash
# Compare with a local reference file
doc-gen --topic "API Documentation" \
        --compare-file reference_docs.html \
        --comparison-report api_comparison.md
```

## How It Works

### 1. Download & Extract
The module downloads the reference documentation and intelligently extracts:
- Main content (removing navigation, ads, etc.)
- Section structure and hierarchy
- Code examples and snippets
- Metadata (title, author, keywords)

### 2. Generate Documentation
Your topic is processed through the standard generation pipeline with your specified parameters.

### 3. Compare & Analyze
Seven different similarity metrics evaluate the documents:

| Metric | Description | Weight |
|--------|-------------|--------|
| **Content Similarity** | Overall text similarity using sequence matching | 30% |
| **Structural Similarity** | Section organization and hierarchy | 20% |
| **Code Similarity** | Code examples and snippets | 25% |
| **Semantic Similarity** | Keywords and concept overlap | 25% |
| **Jaccard Similarity** | Word/character set overlap | - |
| **Cosine Similarity** | TF-IDF vector similarity | - |
| **Levenshtein Distance** | Character-level edit distance | - |

### 4. Generate Recommendations
Based on the scores, the module provides specific recommendations:
- Missing sections to add
- Content length adjustments
- Code example suggestions
- Structural improvements

## Similarity Metrics Explained

### Composite Score
The overall quality score (0-100%) combining all metrics with configurable weights.

**Interpretation:**
- **80-100%**: Excellent match, high quality
- **60-79%**: Good similarity, minor improvements needed
- **40-59%**: Moderate similarity, several improvements recommended
- **Below 40%**: Low similarity, significant adjustments needed

### Individual Metrics

#### Content Similarity
Measures how similar the actual text content is between documents using sequence matching algorithms.

#### Structural Similarity
Compares:
- Number and naming of sections
- Hierarchy levels
- Section ordering

#### Code Similarity
Evaluates:
- Number of code examples
- Programming languages used
- Code content similarity

#### Semantic Similarity
Analyzes:
- Keyword overlap
- N-gram patterns
- Topic coherence

## Advanced Usage

### Python API

```python
from doc_generator.evaluator import DocumentationComparator

# Initialize comparator
comparator = DocumentationComparator()

# Compare with URL
results = comparator.compare_with_url(
    topic="Machine Learning Basics",
    reference_url="https://scikit-learn.org/stable/tutorial/basic/tutorial.html",
    generation_params={
        'model': 'gpt-4',
        'temperature': 0.3,
        'runs': 3
    }
)

# Generate detailed report
report = comparator.generate_report(results, "ml_comparison.md")
print(f"Composite Score: {results['scores']['composite_score']:.2%}")
```

### Custom Weights

```python
from doc_generator.evaluator import SimilarityMetrics

metrics = SimilarityMetrics()

# Calculate with custom weights
composite = metrics.calculate_composite_score(
    content_sim=0.75,
    structural_sim=0.80,
    code_sim=0.65,
    semantic_sim=0.70,
    weights={
        'content': 0.4,    # Increase content weight
        'structure': 0.15,
        'code': 0.20,
        'semantic': 0.25
    }
)
```

### Batch Comparison

```python
# Compare multiple topics against references
topics_and_refs = [
    ("Python Lists", "https://docs.python.org/3/tutorial/lists.html"),
    ("NumPy Arrays", "https://numpy.org/doc/stable/user/basics.array.html"),
    ("Pandas DataFrames", "https://pandas.pydata.org/docs/user_guide/dsintro.html")
]

for topic, ref_url in topics_and_refs:
    results = comparator.compare_with_url(topic, ref_url)
    print(f"{topic}: {results['scores']['composite_score']:.2%}")
```

## Platform Support

The downloader automatically detects and handles different documentation platforms:

- **Sphinx** (Python docs, NumPy, etc.)
- **MkDocs** (Material theme, etc.)
- **ReadTheDocs**
- **GitHub** (READMEs, wikis)
- **Generic** HTML documentation

## Caching

Downloaded documentation is cached locally to avoid repeated downloads:

```bash
# Cache location
.doc_cache/
├── [md5_hash].html     # Cached HTML content
└── [md5_hash].json     # Metadata
```

To bypass cache:
```python
downloader = DocumentationDownloader()
content = downloader.download_page(url, use_cache=False)
```

## Best Practices

### 1. Choose Good References
Select high-quality reference documentation that represents your target style and structure.

### 2. Iterate on Low Scores
If composite score < 60%, try:
- Adjusting temperature (lower for more consistent output)
- Modifying prompts to match reference style
- Adding more specific examples

### 3. Focus on Important Metrics
Not all metrics are equally important for every use case:
- **Technical docs**: Prioritize code similarity
- **Conceptual docs**: Focus on semantic similarity
- **Tutorials**: Emphasize structural similarity

### 4. Use Multiple References
Compare against several references to identify consistent patterns:

```bash
# Compare with multiple references
for url in reference_urls; do
    doc-gen --topic "$TOPIC" --compare-url "$url" --comparison-report "report_$i.md"
done
```

## Troubleshooting

### Low Similarity Scores

**Problem**: Consistently low scores across all metrics

**Solutions**:
1. Verify reference URL is accessible and contains main content
2. Check if topic matches reference content
3. Try different temperature settings
4. Add more specific examples to prompts

### Missing Sections

**Problem**: Important sections missing from generated docs

**Solutions**:
1. Update prompt template to include required sections
2. Add section examples to few-shot prompts
3. Use `--analyze` flag to identify section patterns

### Code Examples Not Detected

**Problem**: Code similarity score is 0 despite code being present

**Solutions**:
1. Ensure code is properly formatted in `<pre>` or `<code>` tags
2. Check language detection in generated HTML
3. Verify reference documentation has extractable code blocks

## Example Comparison Report

```markdown
# Documentation Comparison Report

Generated: 2024-01-15 10:30:00

## Metadata
- **Topic**: Python List Comprehensions
- **Reference URL**: https://docs.python.org/3/tutorial/datastructures.html
- **Model**: gpt-4
- **Temperature**: 0.3

## Similarity Scores
- **Composite Score**: 74.5%
- **Content Similarity**: 68.2%
- **Structural Similarity**: 85.0%
- **Code Similarity**: 71.3%
- **Semantic Similarity**: 73.5%

## Detailed Analysis

### Section Analysis
**Missing Sections**: Advanced techniques, Performance considerations
**Extra Sections**: Installation guide
**Common Sections**: Introduction, Basic syntax, Examples, Best practices

### Content Metrics
- **Reference Length**: 5,234 characters
- **Generated Length**: 4,876 characters
- **Length Ratio**: 0.93

### Code Examples
- **Reference Code Examples**: 8
- **Generated Code Examples**: 6
- **Languages**: python

## Recommendations
1. Add section on "Performance considerations" to match reference
2. Include 2 more code examples for completeness
3. Expand "Advanced techniques" section by ~500 characters
4. Good structural match - maintain current section organization

## Quality Assessment
Overall Quality: ⭐⭐⭐⭐ Good
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Documentation Quality Check

on: [push, pull_request]

jobs:
  compare-docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Install doc-generator
      run: pip install -e .
    
    - name: Generate and Compare
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        doc-gen --topic "${{ env.TOPIC }}" \
                --compare-url "${{ env.REFERENCE_URL }}" \
                --comparison-report comparison.md
    
    - name: Check Quality Threshold
      run: |
        score=$(grep "Composite Score" comparison.md | grep -oE '[0-9]+\.[0-9]+')
        if (( $(echo "$score < 60" | bc -l) )); then
          echo "Quality below threshold: $score%"
          exit 1
        fi
    
    - name: Upload Report
      uses: actions/upload-artifact@v2
      with:
        name: comparison-report
        path: comparison.md
```

## Next Steps

- [View API Reference](../api/evaluator.md) for detailed method documentation
- [See Advanced Examples](../examples/advanced.md) for more usage patterns
- [Read Testing Guide](testing.md) to test comparison features