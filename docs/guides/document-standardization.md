# Document Standardization Guide

The document standardization feature allows you to transform existing documentation to meet organizational standards and templates. This powerful capability enables consistent formatting, structure, and presentation across your documentation ecosystem.

## Overview

Document standardization uses AI-powered content extraction and template-based transformation to:

- Convert between formats (HTML ↔ Markdown)
- Apply organizational templates and standards
- Maintain content fidelity while improving structure
- Ensure consistent documentation across projects

## Architecture

The standardization system consists of three main components:

### Content Extractors

Extract structured content from various document formats:

- **HTML Extractor**: Parses HTML documents, extracting sections, headers, and content blocks
- **Markdown Extractor**: Processes Markdown files (future enhancement)
- **Extensible Architecture**: Plugin system for additional format support

### Document Standardizer

Core engine that orchestrates the standardization process:

- **Template Management**: Applies predefined organizational templates
- **Section Mapping**: Intelligently maps content to standard sections
- **Format Conversion**: Transforms content between output formats
- **Provider Integration**: Supports multiple LLM providers for content processing

### Section Mapper

Smart content organization system:

- **Template-Based Mapping**: Uses organizational templates for consistent structure
- **Content Analysis**: AI-powered analysis of existing content structure
- **Intelligent Merging**: Combines similar sections appropriately
- **Hierarchical Organization**: Maintains logical document flow

## Quick Start

### Basic Standardization

Transform an HTML document to Markdown with default template:

```bash
# Standardize HTML to Markdown
doc-gen --standardize legacy-docs.html --target-format markdown

# Output: legacy-docs_standardized.md
```

### Using Specific Templates

Apply organizational templates for consistent structure:

```bash
# Technical documentation template
doc-gen --standardize api-docs.html --template technical_documentation

# User guide template  
doc-gen --standardize user-manual.html --template user_guide

# API documentation template
doc-gen --standardize reference.html --template api_documentation
```

### Custom Output Location

Control where standardized documents are saved:

```bash
# Specify output directory
doc-gen --standardize docs.html --output-dir ./standardized-docs

# Files saved to: ./standardized-docs/docs_standardized.html
```

## Available Templates

### Technical Documentation Template

Ideal for technical guides, tutorials, and implementation documentation:

**Structure:**
- Overview and Purpose
- Prerequisites and Requirements
- Installation and Setup
- Configuration
- Usage Instructions
- Examples and Code Samples
- Troubleshooting
- Best Practices
- Additional Resources

**Use Cases:**
- Software installation guides
- Technical tutorials
- Implementation documentation
- System administration guides

```bash
doc-gen --standardize tech-guide.html --template technical_documentation
```

### User Guide Template

Perfect for end-user documentation and how-to guides:

**Structure:**
- Introduction
- Getting Started
- Basic Operations
- Advanced Features
- Common Tasks
- Frequently Asked Questions
- Support and Help
- Glossary

**Use Cases:**
- Application user manuals
- Service documentation
- End-user tutorials
- Process documentation

```bash
doc-gen --standardize user-docs.html --template user_guide
```

### API Documentation Template

Designed for API references and developer documentation:

**Structure:**
- API Overview
- Authentication
- Endpoints and Methods
- Request/Response Formats
- Error Codes
- Rate Limits
- Code Examples
- SDKs and Libraries
- Changelog

**Use Cases:**
- REST API documentation
- GraphQL API guides
- SDK documentation
- Developer references

```bash
doc-gen --standardize api-ref.html --template api_documentation
```

## Format Conversion

### HTML to Markdown

Convert HTML documentation to clean, readable Markdown:

```bash
# Basic conversion
doc-gen --standardize document.html --target-format markdown

# With specific template
doc-gen --standardize complex-doc.html \
    --template technical_documentation \
    --target-format markdown
```

**Benefits:**
- Clean, readable format
- Version control friendly
- Platform independent
- Easy editing and maintenance

### Markdown to HTML

Transform Markdown files to formatted HTML:

```bash
# Convert to HTML
doc-gen --standardize README.md --target-format html

# With styling and template
doc-gen --standardize guide.md \
    --template user_guide \
    --target-format html
```

**Benefits:**
- Professional presentation
- Consistent styling
- Web-ready output
- Rich formatting support

### Auto-Detection

Let the system automatically determine the best format:

```bash
# Auto-detect input and choose appropriate output
doc-gen --standardize document.html --format auto

# System will convert HTML → Markdown by default
# Or choose based on organizational preferences
```

## Advanced Configuration

### Provider Selection

Choose your preferred LLM provider for content processing:

```bash
# Use OpenAI GPT models
doc-gen --standardize docs.html \
    --provider openai \
    --model gpt-4

# Use Anthropic Claude models
doc-gen --standardize docs.html \
    --provider claude \
    --model claude-3-5-sonnet-20240620

# Auto-select based on available keys
doc-gen --standardize docs.html --provider auto
```

### Temperature Control

Adjust creativity vs. consistency in content transformation:

```bash
# Conservative transformation (recommended for technical docs)
doc-gen --standardize technical-spec.html --temperature 0.1

# Balanced transformation (default)
doc-gen --standardize user-guide.html --temperature 0.3

# Creative transformation (for marketing materials)
doc-gen --standardize marketing-copy.html --temperature 0.7
```

### Verbose Output

Monitor the standardization process:

```bash
# Detailed process information
doc-gen --standardize document.html --verbose

# Shows:
# - Content extraction details
# - Section mapping decisions
# - Template application steps
# - Provider interactions
# - File operations
```

## Integration Workflows

### Documentation Migration

Migrate legacy documentation to modern standards:

```bash
#!/bin/bash
# migrate-docs.sh

# Create output directory
mkdir -p migrated-docs

# Process all HTML files
for file in legacy-docs/*.html; do
    echo "Processing: $file"
    doc-gen --standardize "$file" \
        --template technical_documentation \
        --target-format markdown \
        --output-dir migrated-docs
    sleep 2  # Rate limiting
done

echo "Migration completed. Check migrated-docs/ directory."
```

### Quality Assurance

Standardize and analyze documentation quality:

```bash
# Standardize with analysis
doc-gen --standardize old-docs.html \
    --template technical_documentation \
    --analyze \
    --report-format markdown

# Review generated reports:
# - old-docs_standardized.md (standardized content)
# - old-docs_analysis_report.md (quality analysis)
```

### Continuous Integration

Integrate standardization into CI/CD pipelines:

```yaml
# .github/workflows/docs-standardization.yml
name: Documentation Standardization

on:
  pull_request:
    paths: ['docs/**']

jobs:
  standardize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install doc-generator
        run: pip install doc-generator
        
      - name: Standardize documentation
        run: |
          for file in docs/**/*.html; do
            doc-gen --standardize "$file" \
                --template technical_documentation \
                --target-format markdown \
                --output-dir standardized/
          done
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          
      - name: Upload standardized docs
        uses: actions/upload-artifact@v3
        with:
          name: standardized-documentation
          path: standardized/
```

## Troubleshooting

### Common Issues

#### Issue: Content extraction fails

```bash
# Check input file exists and is readable
ls -la document.html
file document.html

# Try with verbose output
doc-gen --standardize document.html --verbose

# Check file format and encoding
file -i document.html
```

#### Issue: Poor section mapping

```bash
# Try different template
doc-gen --standardize document.html --template user_guide

# Use lower temperature for more conservative mapping
doc-gen --standardize document.html --temperature 0.1

# Review with analysis
doc-gen --standardize document.html --analyze
```

#### Issue: Format conversion problems

```bash
# Explicitly specify target format
doc-gen --standardize document.html --target-format markdown

# Check output format support
doc-gen --info | grep -A 10 "Supported Formats"

# Try with different provider
doc-gen --standardize document.html --provider claude
```

### Error Messages

#### `DocumentStandardizerError`

Indicates issues with the standardization process:

```bash
# Common causes:
# - Invalid input file
# - Unsupported format
# - API rate limiting
# - Missing provider credentials

# Solutions:
# 1. Verify input file
ls -la input-file.html

# 2. Check API keys
echo $OPENAI_API_KEY | head -c 10
echo $ANTHROPIC_API_KEY | head -c 10

# 3. Try with verbose output
doc-gen --standardize input.html --verbose
```

#### File not found errors

```bash
# Ensure file path is correct
doc-gen --standardize ./docs/existing-file.html

# Use absolute paths if needed
doc-gen --standardize /full/path/to/document.html

# Check working directory
pwd
ls -la *.html
```

### Performance Optimization

#### Rate Limiting

Avoid API rate limits during batch processing:

```bash
# Add delays between requests
for file in *.html; do
    doc-gen --standardize "$file"
    sleep 5  # 5-second delay
done

# Use batch processing script
cat > batch-standardize.sh << 'EOF'
#!/bin/bash
for file in "$@"; do
    echo "Processing: $file"
    doc-gen --standardize "$file" \
        --template technical_documentation
    sleep 3
done
EOF

chmod +x batch-standardize.sh
./batch-standardize.sh docs/*.html
```

#### Token Usage Optimization

Minimize API costs:

```bash
# Use efficient models for standardization
doc-gen --standardize large-doc.html --model gpt-3.5-turbo

# Process smaller sections
# (Split large documents before standardization)

# Use caching effectively
# (System automatically caches similar content)
```

## Best Practices

### Template Selection

Choose templates based on document type:

- **Technical Documentation**: Implementation guides, tutorials, technical specifications
- **User Guide**: End-user manuals, how-to documentation, process guides  
- **API Documentation**: Developer references, API guides, SDK documentation

### Content Preparation

Prepare source documents for optimal standardization:

1. **Clean HTML**: Remove unnecessary styling and formatting
2. **Consistent Headers**: Use proper heading hierarchy (H1, H2, H3)
3. **Semantic Markup**: Use appropriate HTML tags for content structure
4. **Clear Sections**: Organize content into logical sections

### Quality Assurance

Ensure standardization quality:

```bash
# Always review standardized output
doc-gen --standardize important-doc.html --analyze

# Compare with original
diff -u original.html standardized.html

# Test with multiple templates
doc-gen --standardize doc.html --template technical_documentation
doc-gen --standardize doc.html --template user_guide
```

### Version Control

Track standardization changes:

```bash
# Commit original and standardized versions
git add original-docs/ standardized-docs/
git commit -m "Add standardized documentation versions"

# Use descriptive branch names
git checkout -b docs/standardization-migration
```

## Next Steps

- **[Advanced Examples](../examples/advanced.md)**: Complex standardization workflows
- **[API Reference](../api/)**: Technical implementation details  
- **[Contributing](contributing.md)**: Help improve standardization features
- **[Plugin Development](creating-plugins.md)**: Extend extractor and template capabilities

The document standardization system provides powerful capabilities for maintaining consistent, high-quality documentation across your organization. Regular use of standardization workflows ensures documentation remains current, accessible, and professionally presented.