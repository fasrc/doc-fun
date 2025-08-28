# Basic Usage Examples

This page demonstrates basic usage patterns for doc-generator across all operation modes with practical examples.

## Getting Started Examples

### Operation Modes Overview

Doc-generator supports four main operation modes:

1. **Topic Mode** (`--topic`): Generate documentation for specific topics
2. **README Mode** (`--readme`): Generate README files for code projects
3. **Standardization Mode** (`--standardize`): Transform existing documentation
4. **Code Scanning Mode** (`--scan-code`): Legacy code discovery

## Topic Mode Examples

### Simple Documentation Generation

```bash
# Generate documentation for a Python topic
doc-gen --topic "Python Programming" --output-dir ./docs

# What this creates:
# ./docs/python_programming_gpt4omini_temp03.html
```

**Generated content includes:**
- Overview of Python programming
- Installation instructions with HPC modules
- Basic usage examples and syntax
- Code examples and best practices
- Relevant references and links

### Multiple Runs for Quality

```bash
# Generate 3 variants to compare quality
doc-gen --topic "Machine Learning" --runs 3 --output-dir ./ml-docs

# Creates:
# ./ml-docs/machine_learning_gpt4omini_temp03_v1.html
# ./ml-docs/machine_learning_gpt4omini_temp03_v2.html  
# ./ml-docs/machine_learning_gpt4omini_temp03_v3.html
```

### Different Models and Temperatures

```bash
# Use recommended models for optimal quality
doc-gen --topic "Database Design" --model gpt-4o-mini --temperature 0.2

# Use premium model for complex topics
doc-gen --topic "Advanced Algorithms" --model gpt-4o --temperature 0.3

# Use Claude for code-heavy documentation
doc-gen --topic "API Development" --model claude-3-5-sonnet-20241022 --temperature 0.3

# Use higher temperature for more creative content
doc-gen --topic "Creative Coding" --model gpt-4o --temperature 0.8
```

## Plugin Examples

### View Available Plugins

```bash
# List all loaded plugins
doc-gen --list-plugins
```

**Expected output:**
```
Available Recommendation Engine Plugins:
==================================================

Plugin: modules
  Class: ModuleRecommender  
  Module: doc_generator.plugins.modules
  Supported Types: hpc_modules, software, compilers, libraries
  Priority: 100
  Enabled: True
```

### Plugin-Enhanced Documentation

```bash
# Generate with module recommendations
doc-gen --topic "NumPy Array Processing"

# The ModuleRecommender plugin automatically adds:
# - module load python/3.12.8-fasrc01
# - module load gcc/12.2.0-fasrc01  
# - Relevant module descriptions and usage
```

### Managing Plugins

```bash
# Disable all plugins for plain documentation
doc-gen --topic "General Topic" --disable-plugins modules

# Enable only specific plugins (when you have multiple)
doc-gen --topic "Data Science" --enable-only modules,datasets
```

## Output Management

### Organized Output Structure

```bash
# Create organized documentation structure
mkdir -p docs/{tutorials,references,guides}

# Generate different types of documentation
doc-gen --topic "Getting Started with Python" --output-dir docs/tutorials
doc-gen --topic "Python API Reference" --output-dir docs/references  
doc-gen --topic "Python Best Practices" --output-dir docs/guides
```

### Custom File Naming

The tool automatically generates descriptive filenames:

```bash
# Topic: "Python Machine Learning"
# Output: python_machine_learning_gpt4omini_temp03.html

# Topic: "SLURM Job Arrays" 
# Output: slurm_job_arrays_gpt4omini_temp03.html

# With multiple runs:
# Output: topic_name_model_tempX_v1.html, topic_name_model_tempX_v2.html, etc.
```

## Configuration Examples

### Custom Prompt Templates

```bash
# Use Markdown output format
doc-gen --topic "GitHub Documentation" \
  --prompt-yaml prompts/generator/markdown.yaml \
  --output-dir ./markdown-docs

# Use API documentation template
doc-gen --topic "REST API Guide" \
  --prompt-yaml prompts/generator/api-docs.yaml
```

### Custom Terminology

```bash
# Use custom HPC modules and terminology
doc-gen --topic "Specialized Computing" \
  --terminology-path ./config/custom-terminology.yaml \
  --output-dir ./custom-docs
```

## Quality and Analysis

### Basic Analysis

```bash
# Generate with document analysis
doc-gen --topic "Important Documentation" --analyze

# Creates additional files:
# - topic_name_analysis_report.md (structural analysis)
# - Section scores and recommendations
```

### Full Quality Pipeline

```bash
# Complete quality evaluation
doc-gen --topic "Critical Guide" \
  --runs 3 \
  --analyze \
  --quality-eval \
  --verbose

# Creates:
# - Multiple documentation variants
# - Analysis report with scores
# - GPT quality evaluation report
# - Detailed logging output
```

## README Mode Examples

### Basic README Generation

```bash
# Generate README for a single project directory
doc-gen --readme /path/to/my-project --output-dir ./output

# Result: my_project_readme_v1.md with:
# Project structure analysis
# Code example discovery
# Installation and usage instructions
```

### Recursive README Generation

```bash
# Generate READMEs for all subdirectories
doc-gen --readme /path/to/projects --recursive --output-dir ./readmes

# Processes:
# /path/to/projects/project1/ → project1_readme_v1.md
# /path/to/projects/project2/ → project2_readme_v2.md
# /path/to/projects/project3/ → project3_readme_v1.md
```

### README with Analysis

```bash
# Generate multiple variants with quality analysis
doc-gen --readme /path/to/project --runs 3 --analyze --output-dir ./analyzed

# Creates:
# - project_readme_v1.md (best quality variant)
# - project_readme_v2.md
# - project_readme_v3.md
# - project_analysis_report.md (quality assessment)
```

### README with Different Providers

```bash
# Use Claude for README generation
doc-gen --readme /path/to/project --provider claude --model claude-3-5-sonnet-20240620

# Use OpenAI GPT-4
doc-gen --readme /path/to/project --provider openai --model gpt-4
```

## Document Standardization Examples

### Basic HTML to Markdown Conversion

```bash
# Convert HTML documentation to Markdown
doc-gen --standardize legacy-docs.html --target-format markdown

# Result: legacy-docs_standardized.md
# Clean Markdown formatting
# Preserved content structure
# Consistent section organization
```

### Template-Based Standardization

```bash
# Apply technical documentation template
doc-gen --standardize api-docs.html --template technical_documentation

# Apply user guide template
doc-gen --standardize user-manual.html --template user_guide

# Apply API documentation template
doc-gen --standardize reference.html --template api_documentation
```

### Batch Document Standardization

```bash
# Standardize all HTML files in a directory
for file in legacy-docs/*.html; do
    echo "Processing: $file"
    doc-gen --standardize "$file" \
        --template technical_documentation \
        --target-format markdown \
        --output-dir standardized/
    sleep 2  # Rate limiting
done
```

### Standardization with Custom Output

```bash
# Specify output location and format
doc-gen --standardize complex-doc.html \
    --template technical_documentation \
    --target-format markdown \
    --output-dir ./standardized-docs/

# With verbose output for debugging
doc-gen --standardize problem-doc.html \
    --verbose \
    --temperature 0.1 \
    --provider claude
```

## Code Scanning Examples (Legacy Mode)

### Basic Code Scanning

```bash
# Scan current directory for code examples
doc-gen --scan-code . --max-scan-files 50

# Updates terminology.yaml with discovered code patterns
# Creates code-examples-report.txt
```

### Code Scanning with README Generation

```bash
# Legacy mode: scan and generate READMEs
doc-gen --scan-code /path/to/project \
    --generate-readme \
    --ai-enhance \
    --overwrite

# Scans code, generates enhanced README with AI descriptions
```

### Recursive Code Scanning

```bash
# Scan entire project tree
doc-gen --scan-code /large/project \
    --recursive \
    --max-scan-files 200 \
    --suffix "_generated"
```

## Common Workflows

### Documentation Migration Workflow

```bash
# 1. Scan existing codebase
doc-gen --scan-code ./legacy-project --max-scan-files 100

# 2. Generate new READMEs
doc-gen --readme ./legacy-project --recursive --runs 2

# 3. Standardize existing HTML docs
for html in docs/*.html; do
    doc-gen --standardize "$html" --template technical_documentation
done

# 4. Generate topic-based documentation for missing areas
doc-gen --topic "API Migration Guide" --analyze
```

### Multi-Format Documentation Workflow  

```bash
# 1. Generate HTML documentation
doc-gen --topic "User Guide" --format html --output-dir ./html-docs

# 2. Generate Markdown version
doc-gen --topic "User Guide" --format markdown --output-dir ./md-docs

# 3. Standardize existing docs to match
doc-gen --standardize existing.html --target-format markdown --template user_guide

# 4. Create project READMEs
doc-gen --readme . --recursive
```

### Research Documentation Workflow

```bash
# 1. Generate methodology documentation
doc-gen --topic "Computational Biology Pipeline" \
  --output-dir research/methods \
  --model gpt-4 \
  --temperature 0.2

# 2. Create software documentation  
doc-gen --topic "Bioinformatics Tools Setup" \
  --output-dir research/software \
  --runs 2

# 3. Generate troubleshooting guide
doc-gen --topic "Common Pipeline Errors" \
  --output-dir research/troubleshooting
```

### Software Project Workflow

```bash
# 1. API documentation
doc-gen --topic "REST API Reference" \
  --prompt-yaml prompts/generator/api-docs.yaml \
  --output-dir docs/api

# 2. Installation guide
doc-gen --topic "Development Environment Setup" \
  --output-dir docs/setup \
  --runs 2 \
  --analyze

# 3. User tutorials
doc-gen --topic "Getting Started Tutorial" \
  --output-dir docs/tutorials \
  --temperature 0.4
```

### HPC User Support Workflow

```bash
# 1. New user onboarding
doc-gen --topic "Cluster Access and Setup" \
  --output-dir support/onboarding

# 2. Software tutorials
doc-gen --topic "SLURM Job Submission Guide" \
  --output-dir support/tutorials \
  --runs 2

# 3. Troubleshooting documentation
doc-gen --topic "Common SLURM Issues" \
  --output-dir support/troubleshooting \
  --analyze \
  --quality-eval
```

## Topic Formulation Tips

### Effective Topics

```bash
# Good: Specific and descriptive
doc-gen --topic "Python pandas DataFrame operations for time series analysis"
doc-gen --topic "SLURM job arrays for Monte Carlo simulations"
doc-gen --topic "GPU-accelerated machine learning with PyTorch"

# Poor: Too vague or generic
doc-gen --topic "Python"
doc-gen --topic "Help"
doc-gen --topic "Programming"
```

### Context-Rich Topics

```bash
# Include relevant context for better results
doc-gen --topic "Setting up Jupyter notebooks on FASRC cluster with conda environments"

# Specify use cases
doc-gen --topic "Parallel processing with MPI for computational chemistry simulations"

# Include technology stack
doc-gen --topic "Django web application deployment with PostgreSQL and Redis"
```

## Common Issues and Solutions

### No Module Recommendations

```bash
# Issue: Generated documentation lacks module suggestions
# Solution: Check plugin status
doc-gen --list-plugins

# Ensure terminology.yaml has HPC modules
head -20 terminology.yaml

# Test plugin directly
python -c "
from doc_generator.plugins.modules import ModuleRecommender
import yaml
with open('terminology.yaml') as f:
    term = yaml.safe_load(f)
rec = ModuleRecommender(terminology=term)
print(rec.get_recommendations('Python'))
"
```

### Empty or Low Quality Output

```bash
# Issue: Generated documentation is empty, incomplete, or low quality
# Solutions:

# 1. Check for GPT-5 model usage (known issue)
# Avoid these problematic models:
# doc-gen --topic "Any Topic" --model gpt-5        # Often returns empty content
# doc-gen --topic "Any Topic" --model gpt-5-mini   # API bugs

# Use recommended models instead:
doc-gen --topic "Same Topic" --model gpt-4o-mini    # Reliable and cost-effective
doc-gen --topic "Same Topic" --model claude-3-5-sonnet-20241022  # Excellent quality

# 2. Use more specific topics
doc-gen --topic "Python scikit-learn model training for image classification with GPU acceleration"

# 3. Choose optimal model for task complexity
doc-gen --topic "Simple Guide" --model gpt-4o-mini      # Cost-effective for basic topics
doc-gen --topic "Complex Analysis" --model gpt-4o       # Higher capability for advanced topics
doc-gen --topic "Code Documentation" --model claude-3-5-sonnet-20241022  # Excellent for code

# 4. Lower temperature for more focused output
doc-gen --topic "Technical Documentation" --temperature 0.1

# 5. Generate multiple variants and select best
doc-gen --topic "Important Documentation" --runs 5 --analyze

# 6. Use quality evaluation to identify issues
doc-gen --topic "Critical Documentation" --runs 3 --analyze --quality-eval
```

### API Rate Limiting

```bash
# Issue: OpenAI API rate limits
# Solutions:

# 1. Add delays between generations
doc-gen --topic "Topic 1"
sleep 5
doc-gen --topic "Topic 2"

# 2. Use cheaper model for testing
doc-gen --topic "Test Topic" --model gpt-3.5-turbo

# 3. Reduce number of runs during development
doc-gen --topic "Dev Topic" --runs 1
```

---

Ready for more advanced usage? Check out the [Advanced Workflows](advanced.md) examples or learn about [Plugin Examples](plugins.md)!