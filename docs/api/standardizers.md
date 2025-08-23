# Standardizers API

The standardizers module provides the core document standardization engine, responsible for transforming extracted content according to organizational templates and formatting standards.

## Overview

The standardization system takes extracted content and applies intelligent transformations to ensure consistent structure, formatting, and presentation across documentation. It combines template-based organization with AI-powered content enhancement.

## Architecture

The standardizer system consists of interconnected components:

```python
from doc_generator.standardizers import DocumentStandardizer, SectionMapper

# Core standardization workflow
standardizer = DocumentStandardizer(
    provider='openai',
    model='gpt-4o-mini',
    output_format='markdown'
)

# Standardize a document
result = standardizer.standardize_file(
    file_path="legacy-doc.html",
    output_path="standardized-doc.md",
    target_format="markdown"
)
```

## Document Standardizer

The main standardization engine that orchestrates the transformation process.

::: doc_generator.standardizers.document_standardizer
    options:
      show_source: true
      show_root_heading: true
      show_root_members_full_path: false
      show_category_heading: true
      heading_level: 3

## Section Mapper

Intelligent content organization system that maps existing content to standardized section templates.

::: doc_generator.standardizers.section_mapper
    options:
      show_source: true
      show_root_heading: true
      show_root_members_full_path: false
      show_category_heading: true
      heading_level: 3

## Basic Usage

### Simple Standardization

```python
from doc_generator.standardizers import DocumentStandardizer
from pathlib import Path

# Initialize standardizer with default settings
standardizer = DocumentStandardizer()

# Standardize an HTML document to Markdown
result = standardizer.standardize_file(
    file_path=Path("legacy-docs.html"),
    output_path=Path("standardized-docs.md"),
    target_format="markdown"
)

# Check results
print(f"Standardized: {result['output_path']}")
print(f"Sections processed: {len(result['sections_processed'])}")
print(f"Format conversion: {result['original_format']} → {result['target_format']}")
```

### Advanced Configuration

```python
from doc_generator.standardizers import DocumentStandardizer

# Configure with specific provider and model
standardizer = DocumentStandardizer(
    provider='claude',
    model='claude-3-5-sonnet-20240620',
    temperature=0.2,
    output_format='html'
)

# Standardize with custom template
result = standardizer.standardize_file(
    file_path="api-reference.html",
    output_path="standardized-api.html", 
    target_format="html",
    template="api_documentation"
)
```

## Templates and Section Mapping

### Available Templates

The standardizer supports multiple organizational templates:

#### Technical Documentation Template

```python
# Template structure for technical documentation
TECHNICAL_TEMPLATE = {
    "overview": "Overview and Purpose",
    "prerequisites": "Prerequisites and Requirements", 
    "installation": "Installation and Setup",
    "configuration": "Configuration",
    "usage": "Usage Instructions",
    "examples": "Examples and Code Samples",
    "troubleshooting": "Troubleshooting",
    "best_practices": "Best Practices",
    "resources": "Additional Resources"
}

# Apply technical template
result = standardizer.standardize_file(
    "tech-guide.html",
    "standardized-guide.md",
    template="technical_documentation"
)
```

#### User Guide Template

```python
# Template structure for user guides
USER_GUIDE_TEMPLATE = {
    "introduction": "Introduction",
    "getting_started": "Getting Started",
    "basic_operations": "Basic Operations", 
    "advanced_features": "Advanced Features",
    "common_tasks": "Common Tasks",
    "faq": "Frequently Asked Questions",
    "support": "Support and Help",
    "glossary": "Glossary"
}

# Apply user guide template
result = standardizer.standardize_file(
    "user-manual.html",
    "standardized-manual.md",
    template="user_guide"
)
```

#### API Documentation Template

```python
# Template structure for API documentation
API_TEMPLATE = {
    "overview": "API Overview",
    "authentication": "Authentication", 
    "endpoints": "Endpoints and Methods",
    "formats": "Request/Response Formats",
    "errors": "Error Codes",
    "rate_limits": "Rate Limits",
    "examples": "Code Examples",
    "sdks": "SDKs and Libraries",
    "changelog": "Changelog"
}

# Apply API template
result = standardizer.standardize_file(
    "api-docs.html", 
    "standardized-api.md",
    template="api_documentation"
)
```

### Custom Template Creation

Create custom templates for specialized documentation:

```python
from doc_generator.standardizers import DocumentStandardizer

# Define custom template
RESEARCH_TEMPLATE = {
    "abstract": "Abstract",
    "introduction": "Introduction and Background",
    "methodology": "Research Methodology", 
    "implementation": "Implementation Details",
    "results": "Results and Analysis",
    "discussion": "Discussion",
    "conclusions": "Conclusions", 
    "future_work": "Future Work",
    "references": "References"
}

# Register custom template
standardizer = DocumentStandardizer()
standardizer.register_template("research_paper", RESEARCH_TEMPLATE)

# Use custom template
result = standardizer.standardize_file(
    "research-paper.html",
    "standardized-paper.md", 
    template="research_paper"
)
```

## Section Mapping Intelligence

### Automatic Content Organization

The section mapper uses AI to intelligently organize content:

```python
from doc_generator.standardizers import SectionMapper

# Initialize section mapper
mapper = SectionMapper(
    provider='openai',
    model='gpt-4o-mini',
    temperature=0.1  # Conservative for accurate mapping
)

# Map extracted content to template
original_content = {
    "sections": [
        {"title": "Getting Started", "content": "..."},
        {"title": "Basic Usage", "content": "..."},
        {"title": "Advanced Topics", "content": "..."}
    ]
}

mapped_content = mapper.map_to_template(
    content=original_content,
    template="technical_documentation"
)

# Review mapping decisions
for section_name, mapped_section in mapped_content.items():
    print(f"Mapped '{mapped_section['original_title']}' → '{section_name}'")
```

### Mapping Strategies

The section mapper employs multiple strategies:

#### Semantic Similarity Matching

```python
# Example: Map similar content based on semantic meaning
original_sections = [
    {"title": "How to Install", "content": "Installation steps..."},
    {"title": "Setup Guide", "content": "Configuration details..."},
    {"title": "Common Problems", "content": "Troubleshooting info..."}
]

# Maps to:
# "How to Install" → "installation" 
# "Setup Guide" → "configuration"
# "Common Problems" → "troubleshooting"
```

#### Keyword-Based Matching

```python
# Keywords associated with template sections
SECTION_KEYWORDS = {
    "installation": ["install", "setup", "deploy", "download"],
    "configuration": ["config", "settings", "options", "parameters"],
    "troubleshooting": ["error", "problem", "issue", "debug", "fix"],
    "examples": ["example", "sample", "demo", "tutorial"]
}
```

#### Content Analysis Matching

```python
# Analyze content structure and patterns
content_patterns = {
    "code_heavy": "examples",      # Sections with lots of code blocks
    "list_heavy": "requirements",  # Sections with many lists
    "question_format": "faq",      # Q&A style content
    "step_by_step": "usage"        # Numbered procedures
}
```

## Format Conversion

### HTML to Markdown

```python
from doc_generator.standardizers import DocumentStandardizer

# Configure for HTML → Markdown conversion
standardizer = DocumentStandardizer(
    output_format='markdown'
)

# Convert with content optimization
result = standardizer.standardize_file(
    file_path="complex-doc.html",
    output_path="clean-doc.md", 
    target_format="markdown"
)

# Review conversion quality
print(f"Original format: {result['original_format']}")
print(f"Target format: {result['target_format']}")
print(f"Content preserved: {result['content_fidelity']:.1%}")
```

### Markdown to HTML

```python
# Configure for Markdown → HTML conversion
standardizer = DocumentStandardizer(
    output_format='html'
)

# Convert with styling
result = standardizer.standardize_file(
    file_path="README.md",
    output_path="styled-readme.html",
    target_format="html"
)
```

### Format-Specific Optimizations

```python
# Markdown-specific optimizations
markdown_standardizer = DocumentStandardizer(
    output_format='markdown',
    markdown_options={
        'code_fence_style': '```',     # Use triple backticks
        'emphasis_style': '*',         # Use asterisks for emphasis
        'list_style': '-',             # Use dashes for lists
        'header_style': 'atx',         # Use # for headers
        'line_length': 80,             # Wrap at 80 characters
        'preserve_breaks': True        # Keep original line breaks
    }
)

# HTML-specific optimizations  
html_standardizer = DocumentStandardizer(
    output_format='html',
    html_options={
        'semantic_markup': True,       # Use semantic HTML5 tags
        'code_highlighting': True,     # Add syntax highlighting classes
        'responsive_images': True,     # Add responsive image attributes
        'accessibility': True,         # Add ARIA labels and alt text
        'minify': False,               # Keep readable formatting
        'include_toc': True           # Generate table of contents
    }
)
```

## Provider Integration

### Multi-Provider Support

```python
from doc_generator.standardizers import DocumentStandardizer

# OpenAI provider configuration
openai_standardizer = DocumentStandardizer(
    provider='openai',
    model='gpt-4',
    temperature=0.3
)

# Anthropic Claude provider configuration  
claude_standardizer = DocumentStandardizer(
    provider='claude',
    model='claude-3-5-sonnet-20240620',
    temperature=0.2
)

# Auto-selection based on available API keys
auto_standardizer = DocumentStandardizer(
    provider='auto'  # Automatically chooses best available provider
)
```

### Provider-Specific Optimizations

```python
# Configure based on provider strengths
def create_optimized_standardizer(provider: str):
    """Create standardizer optimized for specific provider"""
    
    if provider == 'openai':
        # OpenAI excels at structured output
        return DocumentStandardizer(
            provider='openai',
            model='gpt-4',
            temperature=0.2,  # Lower temperature for consistency
            max_tokens=4000   # Longer context for complex documents
        )
    
    elif provider == 'claude':
        # Claude excels at nuanced understanding
        return DocumentStandardizer(
            provider='claude', 
            model='claude-3-5-sonnet-20240620',
            temperature=0.3,  # Slightly higher for natural language
            max_tokens=8000   # Utilize Claude's longer context
        )
    
    else:
        return DocumentStandardizer(provider='auto')
```

## Batch Processing

### Multiple Document Standardization

```python
from doc_generator.standardizers import DocumentStandardizer
from pathlib import Path
import logging

def batch_standardize(input_dir: Path, output_dir: Path, template: str):
    """Standardize all documents in a directory"""
    
    standardizer = DocumentStandardizer(
        provider='auto',
        output_format='markdown'
    )
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all HTML files
    html_files = list(input_dir.glob("*.html"))
    results = {}
    
    for html_file in html_files:
        try:
            # Generate output path
            output_file = output_dir / f"{html_file.stem}_standardized.md"
            
            # Standardize document
            result = standardizer.standardize_file(
                file_path=html_file,
                output_path=output_file,
                target_format="markdown",
                template=template
            )
            
            results[str(html_file)] = result
            logging.info(f"Standardized: {html_file} → {output_file}")
            
        except Exception as e:
            logging.error(f"Failed to standardize {html_file}: {e}")
            results[str(html_file)] = {"error": str(e)}
    
    return results

# Usage
results = batch_standardize(
    input_dir=Path("legacy-docs"),
    output_dir=Path("standardized-docs"), 
    template="technical_documentation"
)
```

### Parallel Processing

```python
import concurrent.futures
from doc_generator.standardizers import DocumentStandardizer

def parallel_standardize(file_paths: list, template: str, max_workers: int = 3):
    """Standardize documents in parallel"""
    
    def standardize_single(file_path):
        standardizer = DocumentStandardizer()
        output_path = file_path.with_suffix('.md').with_name(
            f"{file_path.stem}_standardized.md"
        )
        return standardizer.standardize_file(file_path, output_path, template=template)
    
    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(standardize_single, path): path
            for path in file_paths
        }
        
        results = {}
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                results[str(path)] = future.result()
            except Exception as e:
                results[str(path)] = {"error": str(e)}
    
    return results
```

## Quality Assurance

### Content Validation

```python
from doc_generator.standardizers import DocumentStandardizer

def validate_standardization(result: dict) -> dict:
    """Validate standardization quality"""
    
    validation = {
        "sections_mapped": len(result.get('sections_processed', [])),
        "content_preserved": result.get('content_fidelity', 0),
        "format_valid": result.get('target_format') in ['html', 'markdown'],
        "template_applied": bool(result.get('template_used')),
        "errors": result.get('errors', [])
    }
    
    # Calculate quality score
    quality_factors = [
        validation["sections_mapped"] > 0,
        validation["content_preserved"] > 0.8,
        validation["format_valid"],
        validation["template_applied"],
        len(validation["errors"]) == 0
    ]
    
    validation["quality_score"] = sum(quality_factors) / len(quality_factors)
    
    return validation

# Usage
standardizer = DocumentStandardizer()
result = standardizer.standardize_file("document.html", "output.md")
quality_report = validate_standardization(result)

print(f"Quality Score: {quality_report['quality_score']:.1%}")
```

### A/B Testing Different Approaches

```python
def compare_standardization_approaches(file_path: Path, template: str):
    """Compare different standardization configurations"""
    
    approaches = {
        "conservative": DocumentStandardizer(temperature=0.1),
        "balanced": DocumentStandardizer(temperature=0.3), 
        "creative": DocumentStandardizer(temperature=0.7),
        "openai": DocumentStandardizer(provider='openai'),
        "claude": DocumentStandardizer(provider='claude')
    }
    
    results = {}
    
    for name, standardizer in approaches.items():
        try:
            output_path = file_path.with_name(f"{file_path.stem}_{name}.md")
            result = standardizer.standardize_file(
                file_path, output_path, template=template
            )
            results[name] = result
        except Exception as e:
            results[name] = {"error": str(e)}
    
    return results
```

## Error Handling

### Exception Types

```python
from doc_generator.exceptions import (
    DocumentStandardizerError,
    TemplateError,
    SectionMappingError,
    FormatConversionError
)

try:
    result = standardizer.standardize_file("document.html", "output.md")
    
except DocumentStandardizerError as e:
    print(f"General standardization error: {e}")
    
except TemplateError as e:
    print(f"Template application error: {e}")
    
except SectionMappingError as e:
    print(f"Section mapping failed: {e}")
    
except FormatConversionError as e:
    print(f"Format conversion error: {e}")
```

### Robust Error Recovery

```python
def robust_standardize(file_path: Path, output_path: Path, template: str = None):
    """Standardize with comprehensive error handling"""
    
    # Try multiple fallback strategies
    strategies = [
        {"template": template, "temperature": 0.2},
        {"template": "technical_documentation", "temperature": 0.1},
        {"template": None, "temperature": 0.1}  # No template fallback
    ]
    
    for strategy in strategies:
        try:
            standardizer = DocumentStandardizer(**strategy)
            return standardizer.standardize_file(file_path, output_path)
            
        except Exception as e:
            logging.warning(f"Strategy failed: {strategy}, Error: {e}")
            continue
    
    # Final fallback: simple format conversion
    try:
        from doc_generator.extractors import HTMLContentExtractor
        extractor = HTMLContentExtractor()
        content = extractor.extract(file_path)
        
        # Create minimal standardized output
        with open(output_path, 'w') as f:
            f.write(f"# {content.get('title', 'Document')}\n\n")
            for section in content.get('sections', []):
                f.write(f"## {section['title']}\n\n{section['content']}\n\n")
        
        return {
            "output_path": str(output_path),
            "original_format": "html",
            "target_format": "markdown",
            "fallback_used": True
        }
        
    except Exception as e:
        raise DocumentStandardizerError(f"All standardization strategies failed: {e}")
```

## Performance Optimization

### Caching

```python
from functools import lru_cache
from doc_generator.standardizers import DocumentStandardizer

class CachedStandardizer(DocumentStandardizer):
    """Standardizer with intelligent caching"""
    
    @lru_cache(maxsize=128)
    def _cached_section_mapping(self, content_hash: str, template: str):
        """Cache section mappings for repeated content"""
        return super()._map_sections(content_hash, template)
    
    def standardize_file(self, file_path, output_path, **kwargs):
        """Standardize with caching"""
        
        # Generate content hash for caching
        import hashlib
        content_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
        
        # Use cached mapping if available
        kwargs['content_hash'] = content_hash
        return super().standardize_file(file_path, output_path, **kwargs)
```

### Memory Optimization

```python
def memory_efficient_standardize(large_file: Path, output_path: Path):
    """Process large documents efficiently"""
    
    # Use streaming approach for large files
    standardizer = DocumentStandardizer(
        # Reduce memory usage
        max_tokens=2000,      # Smaller context window
        temperature=0.1,      # Consistent output
        
        # Optimize for large documents  
        chunk_size=1000,      # Process in chunks
        streaming=True        # Stream processing
    )
    
    return standardizer.standardize_file(large_file, output_path)
```

## Integration Examples

### CI/CD Pipeline Integration

```python
# ci_standardization.py
from doc_generator.standardizers import DocumentStandardizer
from pathlib import Path
import sys

def standardize_documentation_pipeline():
    """Standardize documentation in CI/CD pipeline"""
    
    docs_dir = Path("docs")
    output_dir = Path("standardized-docs")
    
    # Initialize standardizer
    standardizer = DocumentStandardizer(
        provider='auto',
        temperature=0.2,  # Consistent output for CI
        output_format='markdown'
    )
    
    # Track results
    success_count = 0
    error_count = 0
    
    # Process all HTML documentation
    for html_file in docs_dir.glob("**/*.html"):
        try:
            output_file = output_dir / html_file.relative_to(docs_dir).with_suffix('.md')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            result = standardizer.standardize_file(
                html_file, 
                output_file,
                template="technical_documentation"
            )
            
            print(f"✓ Standardized: {html_file.name}")
            success_count += 1
            
        except Exception as e:
            print(f"✗ Failed: {html_file.name} - {e}")
            error_count += 1
    
    # Report results
    total = success_count + error_count
    print(f"\nResults: {success_count}/{total} documents standardized")
    
    # Exit with appropriate code for CI
    if error_count > 0:
        sys.exit(1)  # CI failure
    else:
        sys.exit(0)  # CI success

if __name__ == "__main__":
    standardize_documentation_pipeline()
```

### Pre-commit Hook Integration

```python
#!/usr/bin/env python3
# .git/hooks/pre-commit

from doc_generator.standardizers import DocumentStandardizer
import subprocess
import sys

def check_documentation_standards():
    """Pre-commit hook to ensure documentation standards"""
    
    # Get list of changed HTML files
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=AM"],
        capture_output=True, text=True
    )
    
    changed_files = [
        f for f in result.stdout.strip().split('\n') 
        if f.endswith('.html')
    ]
    
    if not changed_files:
        return True  # No HTML files changed
    
    # Check if files meet standards
    standardizer = DocumentStandardizer()
    
    for file_path in changed_files:
        try:
            # Validate document structure
            result = standardizer.analyze_structure(file_path)
            
            if result['quality_score'] < 0.8:
                print(f"✗ {file_path}: Quality score too low ({result['quality_score']:.1%})")
                return False
                
            if not result['has_proper_sections']:
                print(f"✗ {file_path}: Missing proper section structure")
                return False
                
        except Exception as e:
            print(f"✗ {file_path}: Validation failed - {e}")
            return False
    
    print("✓ All documentation meets quality standards")
    return True

if __name__ == "__main__":
    if not check_documentation_standards():
        print("Commit rejected: Documentation standards not met")
        sys.exit(1)
    
    sys.exit(0)
```

## Testing

### Unit Testing

```python
import unittest
from pathlib import Path
from doc_generator.standardizers import DocumentStandardizer

class TestDocumentStandardizer(unittest.TestCase):
    
    def setUp(self):
        self.standardizer = DocumentStandardizer()
        self.test_html = Path("tests/fixtures/sample.html")
        self.output_dir = Path("tests/output")
        self.output_dir.mkdir(exist_ok=True)
    
    def test_html_to_markdown_conversion(self):
        """Test HTML to Markdown standardization"""
        output_file = self.output_dir / "test_output.md"
        
        result = self.standardizer.standardize_file(
            self.test_html,
            output_file,
            target_format="markdown"
        )
        
        # Verify result structure
        self.assertIn('output_path', result)
        self.assertIn('original_format', result)
        self.assertIn('target_format', result)
        self.assertEqual(result['target_format'], 'markdown')
        
        # Verify output file exists and has content
        self.assertTrue(output_file.exists())
        self.assertGreater(output_file.stat().st_size, 0)
    
    def test_template_application(self):
        """Test template application"""
        output_file = self.output_dir / "test_template.md"
        
        result = self.standardizer.standardize_file(
            self.test_html,
            output_file,
            template="technical_documentation"
        )
        
        # Verify template was applied
        self.assertIn('template_used', result)
        self.assertEqual(result['template_used'], 'technical_documentation')
        
        # Check sections were processed
        self.assertIn('sections_processed', result)
        self.assertGreater(len(result['sections_processed']), 0)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        with self.assertRaises(FileNotFoundError):
            self.standardizer.standardize_file(
                Path("nonexistent.html"),
                self.output_dir / "output.md"
            )
    
    def tearDown(self):
        # Clean up test outputs
        for file in self.output_dir.glob("test_*"):
            file.unlink()

if __name__ == '__main__':
    unittest.main()
```

The standardizers system provides powerful and flexible document transformation capabilities, enabling organizations to maintain consistent, high-quality documentation standards across diverse content sources and formats.