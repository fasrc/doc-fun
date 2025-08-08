# Evaluator API Reference

The evaluator module provides classes for comparing and evaluating documentation quality.

## Module Overview

```python
from doc_generator.evaluator import (
    DocumentationDownloader,
    DocumentationComparator, 
    SimilarityMetrics
)
```

## DocumentationDownloader

Downloads and extracts content from existing documentation pages.

### Class Definition

```python
class DocumentationDownloader(cache_dir: str = ".doc_cache")
```

**Parameters:**
- `cache_dir` (str): Directory for caching downloaded pages. Default: `.doc_cache`

### Methods

#### download_page

```python
def download_page(url: str, use_cache: bool = True) -> str
```

Downloads a documentation page from the specified URL.

**Parameters:**
- `url` (str): URL of the documentation page
- `use_cache` (bool): Whether to use cached version if available

**Returns:**
- str: HTML content of the page

**Example:**
```python
downloader = DocumentationDownloader()
html = downloader.download_page("https://docs.python.org/3/")
```

#### extract_content

```python
def extract_content(html: str, url: str = None) -> Dict[str, any]
```

Extracts structured content from HTML documentation.

**Parameters:**
- `html` (str): HTML content to parse
- `url` (str, optional): URL for platform detection

**Returns:**
- Dict containing:
  - `platform` (str): Detected platform (sphinx, mkdocs, github, etc.)
  - `content` (BeautifulSoup): Main content area
  - `sections` (List[Dict]): Extracted sections with titles and content
  - `metadata` (Dict): Page metadata (title, author, keywords)
  - `code_examples` (List[Dict]): Extracted code blocks
  - `navigation` (List[Dict]): Navigation structure
  - `raw_text` (str): Plain text content

**Example:**
```python
content = downloader.extract_content(html, url="https://docs.python.org/3/")
print(f"Platform: {content['platform']}")
print(f"Sections: {len(content['sections'])}")
```

#### download_and_extract

```python
def download_and_extract(url: str, use_cache: bool = True) -> Dict[str, any]
```

Downloads and extracts content in one step.

**Parameters:**
- `url` (str): URL of the documentation page
- `use_cache` (bool): Whether to use cached version

**Returns:**
- Dict: Extracted content dictionary with added `url` field

## DocumentationComparator

Compares generated documentation with existing documentation.

### Class Definition

```python
class DocumentationComparator(generator: Optional[DocumentationGenerator] = None)
```

**Parameters:**
- `generator` (DocumentationGenerator, optional): Generator instance to use

### Methods

#### compare_with_url

```python
def compare_with_url(
    topic: str,
    reference_url: str,
    generation_params: Optional[Dict] = None
) -> Dict
```

Generates documentation and compares it with a reference URL.

**Parameters:**
- `topic` (str): Topic to generate documentation for
- `reference_url` (str): URL of reference documentation
- `generation_params` (Dict, optional): Parameters for generation
  - `runs` (int): Number of variants to generate
  - `model` (str): Model to use (e.g., 'gpt-4')
  - `temperature` (float): Generation temperature

**Returns:**
- Dict containing:
  - `scores` (Dict): All similarity scores
  - `details` (Dict): Detailed analysis
  - `recommendations` (List[str]): Improvement suggestions
  - `metadata` (Dict): Comparison metadata

**Example:**
```python
comparator = DocumentationComparator()
results = comparator.compare_with_url(
    topic="Python Lists",
    reference_url="https://docs.python.org/3/tutorial/lists.html",
    generation_params={'model': 'gpt-4', 'temperature': 0.3}
)
print(f"Composite Score: {results['scores']['composite_score']:.2%}")
```

#### compare_existing_files

```python
def compare_existing_files(
    generated_file: str,
    reference_file: str
) -> Dict
```

Compares existing generated file with a reference file.

**Parameters:**
- `generated_file` (str): Path to generated documentation
- `reference_file` (str): Path to reference documentation

**Returns:**
- Dict: Comparison results

#### generate_report

```python
def generate_report(
    comparison_results: Dict,
    output_file: Optional[str] = None
) -> str
```

Generates a detailed comparison report in Markdown format.

**Parameters:**
- `comparison_results` (Dict): Results from comparison methods
- `output_file` (str, optional): File path to save report

**Returns:**
- str: Formatted report text

## SimilarityMetrics

Calculates various similarity metrics between documents.

### Class Definition

```python
class SimilarityMetrics()
```

Static methods for calculating similarity metrics.

### Methods

#### sequence_similarity

```python
@staticmethod
def sequence_similarity(s1: str, s2: str) -> float
```

Calculates sequence similarity using difflib.

**Returns:** Float between 0 and 1

#### jaccard_similarity

```python
@staticmethod
def jaccard_similarity(
    text1: str,
    text2: str,
    use_words: bool = True
) -> float
```

Calculates Jaccard similarity coefficient.

**Parameters:**
- `text1`, `text2` (str): Texts to compare
- `use_words` (bool): Compare words (True) or characters (False)

**Returns:** Float between 0 and 1

#### cosine_similarity

```python
@staticmethod
def cosine_similarity(text1: str, text2: str) -> float
```

Calculates cosine similarity using TF-IDF vectors.

**Returns:** Float between 0 and 1

#### structural_similarity

```python
@staticmethod
def structural_similarity(
    sections1: List[Dict],
    sections2: List[Dict]
) -> float
```

Compares document structure (sections, hierarchy).

**Parameters:**
- `sections1`, `sections2` (List[Dict]): Section lists with 'title' and 'level' keys

**Returns:** Float between 0 and 1

#### code_similarity

```python
@staticmethod
def code_similarity(
    examples1: List[Dict],
    examples2: List[Dict]
) -> float
```

Compares code examples between documents.

**Parameters:**
- `examples1`, `examples2` (List[Dict]): Code examples with 'code' and 'language' keys

**Returns:** Float between 0 and 1

#### semantic_similarity

```python
@staticmethod
def semantic_similarity(text1: str, text2: str) -> float
```

Calculates semantic similarity using keywords and n-grams.

**Returns:** Float between 0 and 1

#### calculate_composite_score

```python
@staticmethod
def calculate_composite_score(
    content_sim: float,
    structural_sim: float,
    code_sim: float,
    semantic_sim: float,
    weights: Optional[Dict[str, float]] = None
) -> float
```

Calculates weighted composite similarity score.

**Parameters:**
- `content_sim` (float): Content similarity score
- `structural_sim` (float): Structural similarity score
- `code_sim` (float): Code similarity score
- `semantic_sim` (float): Semantic similarity score
- `weights` (Dict, optional): Custom weights for each metric

**Default weights:**
```python
{
    'content': 0.3,
    'structure': 0.2,
    'code': 0.25,
    'semantic': 0.25
}
```

**Returns:** Float between 0 and 1

## Complete Example

```python
from doc_generator.evaluator import (
    DocumentationDownloader,
    DocumentationComparator,
    SimilarityMetrics
)

# Initialize components
downloader = DocumentationDownloader()
comparator = DocumentationComparator()
metrics = SimilarityMetrics()

# Download reference documentation
ref_content = downloader.download_and_extract(
    "https://numpy.org/doc/stable/user/quickstart.html"
)

# Generate and compare
results = comparator.compare_with_url(
    topic="NumPy Quick Start Guide",
    reference_url="https://numpy.org/doc/stable/user/quickstart.html",
    generation_params={
        'model': 'gpt-4',
        'temperature': 0.3,
        'runs': 1
    }
)

# Access scores
print(f"Composite Score: {results['scores']['composite_score']:.2%}")
print(f"Content Similarity: {results['scores']['content_similarity']:.2%}")
print(f"Code Similarity: {results['scores']['code_similarity']:.2%}")

# Get recommendations
for rec in results['recommendations']:
    print(f"- {rec}")

# Generate report
report = comparator.generate_report(results, "numpy_comparison.md")

# Calculate custom similarity
custom_sim = metrics.jaccard_similarity(
    "Your generated text",
    "Reference text",
    use_words=True
)
print(f"Jaccard Similarity: {custom_sim:.2%}")
```

## Error Handling

The evaluator module handles common errors gracefully:

```python
try:
    results = comparator.compare_with_url(
        topic="My Topic",
        reference_url="https://invalid-url.com"
    )
except requests.RequestException as e:
    print(f"Failed to download: {e}")
except ValueError as e:
    print(f"Generation failed: {e}")
```

## Performance Considerations

### Caching
- Downloaded pages are cached to avoid repeated downloads
- Cache is stored in `.doc_cache/` directory
- Use `use_cache=False` to force fresh downloads

### Large Documents
- Very long documents are automatically truncated for API calls
- Default max length: 3000 characters per section
- Adjust in code if needed for specific use cases

### Rate Limiting
- Comparison with generation includes API rate limiting
- Batch comparisons should include delays between calls