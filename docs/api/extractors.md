# Extractors API

The extractors module provides content extraction capabilities for various document formats, enabling the standardization system to process different input types with a unified interface.

## Overview

Content extractors form the foundation of the document standardization pipeline, responsible for parsing source documents and extracting structured content that can be processed by the standardization engine.

## Architecture

The extractor system uses a plugin-based architecture with a common base interface:

```python
from doc_generator.extractors import ContentExtractor, HTMLContentExtractor

# Base extractor interface
class ContentExtractor:
    def extract(self, file_path: Path) -> dict:
        """Extract structured content from document"""
        pass
    
    def get_supported_formats(self) -> list:
        """Return list of supported file extensions"""
        pass

# HTML-specific implementation
extractor = HTMLContentExtractor()
content = extractor.extract("document.html")
```

## Base Extractor

::: doc_generator.extractors.base
    options:
      show_source: true
      show_root_heading: true
      show_root_members_full_path: false
      show_category_heading: true
      heading_level: 3

## HTML Content Extractor

The HTML extractor provides comprehensive parsing of HTML documents with intelligent content structure detection.

::: doc_generator.extractors.html_extractor
    options:
      show_source: true
      show_root_heading: true
      show_root_members_full_path: false
      show_category_heading: true
      heading_level: 3

## Usage Examples

### Basic HTML Extraction

```python
from doc_generator.extractors import HTMLContentExtractor
from pathlib import Path

# Initialize extractor
extractor = HTMLContentExtractor()

# Extract content from HTML file
content = extractor.extract(Path("document.html"))

# Access extracted data
print(f"Title: {content['title']}")
print(f"Sections: {len(content['sections'])}")
print(f"Format: {content['format']}")

# Iterate through sections
for section in content['sections']:
    print(f"- {section['title']}: {len(section['content'])} characters")
```

### Advanced Extraction with Configuration

```python
from doc_generator.extractors import HTMLContentExtractor

# Configure extractor with custom settings
extractor = HTMLContentExtractor(
    preserve_formatting=True,
    extract_metadata=True,
    include_images=True
)

# Extract with detailed parsing
content = extractor.extract("complex-document.html")

# Access metadata
metadata = content.get('metadata', {})
print(f"Author: {metadata.get('author', 'Unknown')}")
print(f"Created: {metadata.get('created', 'Unknown')}")

# Process images
images = content.get('images', [])
for image in images:
    print(f"Image: {image['src']} - Alt: {image['alt']}")
```

### Error Handling

```python
from doc_generator.extractors import HTMLContentExtractor
from doc_generator.exceptions import ExtractionError

extractor = HTMLContentExtractor()

try:
    content = extractor.extract("document.html")
    print("Extraction successful")
except ExtractionError as e:
    print(f"Extraction failed: {e}")
except FileNotFoundError:
    print("Document file not found")
```

## Content Structure

### Extracted Content Format

The extractor returns a standardized dictionary structure:

```python
{
    "title": "Document Title",
    "format": "html",
    "sections": [
        {
            "title": "Section Title",
            "level": 1,  # Heading level (1-6)
            "content": "Section content text...",
            "subsections": [
                {
                    "title": "Subsection Title", 
                    "level": 2,
                    "content": "Subsection content..."
                }
            ]
        }
    ],
    "metadata": {
        "author": "Document Author",
        "created": "2024-01-01",
        "modified": "2024-01-15",
        "language": "en"
    },
    "images": [
        {
            "src": "image.jpg",
            "alt": "Image description",
            "title": "Image title"
        }
    ],
    "links": [
        {
            "text": "Link text",
            "url": "https://example.com",
            "title": "Link title"
        }
    ]
}
```

### Section Hierarchy

Sections are organized hierarchically based on HTML heading levels:

- **H1**: Top-level sections (`level: 1`)
- **H2**: Major subsections (`level: 2`) 
- **H3-H6**: Nested subsections (`level: 3-6`)

```python
# Access section hierarchy
for section in content['sections']:
    print(f"{'  ' * (section['level'] - 1)}{section['title']}")
    
    # Process subsections
    for subsection in section.get('subsections', []):
        indent = '  ' * (subsection['level'] - 1)
        print(f"{indent}{subsection['title']}")
```

## Extending Extractors

### Creating Custom Extractors

Implement the `ContentExtractor` interface for new formats:

```python
from doc_generator.extractors.base import ContentExtractor
from pathlib import Path
import json

class JSONContentExtractor(ContentExtractor):
    """Extract content from JSON documentation files"""
    
    def get_supported_formats(self) -> list:
        return ['.json']
    
    def extract(self, file_path: Path) -> dict:
        """Extract structured content from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return {
            'title': data.get('title', file_path.stem),
            'format': 'json',
            'sections': self._extract_sections(data),
            'metadata': data.get('metadata', {}),
            'images': [],
            'links': []
        }
    
    def _extract_sections(self, data: dict) -> list:
        """Convert JSON structure to sections"""
        sections = []
        
        for key, value in data.items():
            if key not in ['title', 'metadata']:
                sections.append({
                    'title': key.replace('_', ' ').title(),
                    'level': 1,
                    'content': str(value),
                    'subsections': []
                })
        
        return sections

# Register custom extractor
from doc_generator.extractors import register_extractor
register_extractor(JSONContentExtractor())
```

### Plugin Registration

Register extractors through entry points in `pyproject.toml`:

```toml
[project.entry-points."doc_generator.extractors"]
json = "mypackage.extractors:JSONContentExtractor"
yaml = "mypackage.extractors:YAMLContentExtractor"
docx = "mypackage.extractors:DocxContentExtractor"
```

## Configuration Options

### HTML Extractor Configuration

```python
from doc_generator.extractors import HTMLContentExtractor

# Create with custom configuration
extractor = HTMLContentExtractor(
    # Content processing options
    preserve_formatting=True,      # Keep original HTML formatting
    clean_whitespace=True,         # Normalize whitespace
    extract_code_blocks=True,      # Parse <pre> and <code> blocks
    
    # Metadata extraction
    extract_metadata=True,         # Extract document metadata
    include_meta_tags=True,        # Include HTML meta tags
    
    # Media handling
    include_images=True,           # Extract image references
    include_links=True,            # Extract link references
    preserve_alt_text=True,        # Keep image alt text
    
    # Section parsing
    max_section_depth=6,           # Maximum heading level (H1-H6)
    merge_adjacent_text=True,      # Combine adjacent text nodes
    skip_navigation=True,          # Ignore nav elements
    skip_footer=True,              # Ignore footer elements
    
    # Custom selectors
    content_selector="main, article, .content",  # CSS selector for main content
    exclude_selectors=[".sidebar", ".ads"],      # CSS selectors to exclude
)
```

### Processing Options

Control how content is extracted and structured:

```python
# Minimal extraction (fastest)
minimal_extractor = HTMLContentExtractor(
    preserve_formatting=False,
    extract_metadata=False,
    include_images=False,
    include_links=False
)

# Comprehensive extraction (most complete)
comprehensive_extractor = HTMLContentExtractor(
    preserve_formatting=True,
    extract_metadata=True,
    include_images=True,
    include_links=True,
    extract_code_blocks=True,
    include_meta_tags=True
)
```

## Integration with Standardization

### Extractor Selection

The standardization system automatically selects appropriate extractors:

```python
from doc_generator.standardizers import DocumentStandardizer

# Standardizer automatically chooses extractor based on file extension
standardizer = DocumentStandardizer()

# Process HTML file -> uses HTMLContentExtractor
result = standardizer.standardize_file("document.html")

# Process Markdown file -> uses MarkdownContentExtractor (when available)
result = standardizer.standardize_file("document.md")
```

### Custom Extractor Integration

Use specific extractors within the standardization pipeline:

```python
from doc_generator.standardizers import DocumentStandardizer
from doc_generator.extractors import HTMLContentExtractor

# Initialize with custom extractor configuration
extractor = HTMLContentExtractor(
    preserve_formatting=True,
    extract_code_blocks=True
)

# Use custom extractor in standardizer
standardizer = DocumentStandardizer()
standardizer.set_extractor('html', extractor)

# Process file with custom extraction settings
result = standardizer.standardize_file("technical-doc.html")
```

## Error Handling

### Common Exceptions

The extractor system provides specific exception types:

```python
from doc_generator.exceptions import (
    ExtractionError,
    UnsupportedFormatError,
    DocumentParsingError
)

try:
    content = extractor.extract("document.html")
except UnsupportedFormatError:
    print("File format not supported by this extractor")
except DocumentParsingError as e:
    print(f"Error parsing document structure: {e}")
except ExtractionError as e:
    print(f"General extraction error: {e}")
```

### Validation and Recovery

Implement robust error handling with validation:

```python
from doc_generator.extractors import HTMLContentExtractor
import logging

logger = logging.getLogger(__name__)

def safe_extract(file_path: str) -> dict:
    """Safely extract content with error recovery"""
    extractor = HTMLContentExtractor()
    
    try:
        # Attempt extraction
        content = extractor.extract(file_path)
        
        # Validate extracted content
        if not content.get('sections'):
            logger.warning(f"No sections found in {file_path}")
            return create_minimal_content(file_path)
        
        return content
        
    except Exception as e:
        logger.error(f"Extraction failed for {file_path}: {e}")
        return create_fallback_content(file_path, str(e))

def create_minimal_content(file_path: str) -> dict:
    """Create minimal content structure for failed extractions"""
    return {
        'title': Path(file_path).stem,
        'format': 'unknown',
        'sections': [],
        'metadata': {},
        'images': [],
        'links': []
    }
```

## Performance Considerations

### Optimization Strategies

Improve extraction performance for large documents:

```python
from doc_generator.extractors import HTMLContentExtractor

# Configure for large documents
large_doc_extractor = HTMLContentExtractor(
    # Reduce processing overhead
    preserve_formatting=False,     # Skip formatting preservation
    extract_metadata=False,        # Skip metadata extraction
    include_images=False,          # Skip image processing
    
    # Optimize parsing
    max_section_depth=3,           # Limit section nesting
    merge_adjacent_text=True,      # Reduce text node count
    
    # Use efficient selectors
    content_selector="main",       # Target main content only
    exclude_selectors=[".sidebar", ".footer", ".nav"]  # Skip non-content
)
```

### Batch Processing

Efficiently process multiple documents:

```python
from doc_generator.extractors import HTMLContentExtractor
from pathlib import Path
import concurrent.futures

def batch_extract(document_paths: list) -> dict:
    """Extract content from multiple documents in parallel"""
    extractor = HTMLContentExtractor()
    results = {}
    
    # Process documents in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_path = {
            executor.submit(extractor.extract, path): path 
            for path in document_paths
        }
        
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                results[str(path)] = future.result()
            except Exception as e:
                print(f"Failed to extract {path}: {e}")
                results[str(path)] = None
    
    return results

# Usage
html_files = list(Path("docs").glob("*.html"))
extracted_content = batch_extract(html_files)
```

## Testing

### Unit Testing Extractors

Create comprehensive tests for extractor functionality:

```python
import unittest
from pathlib import Path
from doc_generator.extractors import HTMLContentExtractor

class TestHTMLExtractor(unittest.TestCase):
    
    def setUp(self):
        self.extractor = HTMLContentExtractor()
        self.sample_html = Path("tests/fixtures/sample.html")
    
    def test_basic_extraction(self):
        """Test basic HTML content extraction"""
        content = self.extractor.extract(self.sample_html)
        
        # Verify structure
        self.assertIn('title', content)
        self.assertIn('sections', content)
        self.assertIn('format', content)
        self.assertEqual(content['format'], 'html')
    
    def test_section_hierarchy(self):
        """Test proper section hierarchy extraction"""
        content = self.extractor.extract(self.sample_html)
        
        # Check section structure
        sections = content['sections']
        self.assertGreater(len(sections), 0)
        
        # Verify heading levels
        for section in sections:
            self.assertIn('level', section)
            self.assertIn('title', section)
            self.assertIn('content', section)
    
    def test_metadata_extraction(self):
        """Test metadata extraction from HTML"""
        extractor = HTMLContentExtractor(extract_metadata=True)
        content = extractor.extract(self.sample_html)
        
        # Verify metadata presence
        self.assertIn('metadata', content)
        metadata = content['metadata']
        
        # Check common metadata fields
        if 'title' in metadata:
            self.assertIsInstance(metadata['title'], str)

if __name__ == '__main__':
    unittest.main()
```

## Future Extensions

### Planned Extractors

Future versions will include additional extractors:

- **Markdown Extractor**: Parse Markdown files with frontmatter support
- **PDF Extractor**: Extract content from PDF documents
- **Word Extractor**: Process DOCX files
- **Wiki Extractor**: Parse MediaWiki markup
- **LaTeX Extractor**: Process LaTeX documents

### Enhancement Opportunities

Areas for improvement:

- **Machine Learning Integration**: AI-powered content structure detection
- **Format-Specific Optimization**: Specialized parsing for technical documentation
- **Real-time Processing**: Streaming extraction for large documents
- **Semantic Analysis**: Understanding content relationships and context

The extractors system provides a robust foundation for document content extraction, enabling the standardization pipeline to process diverse document formats with consistent, high-quality results.