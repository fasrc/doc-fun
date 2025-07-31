# Basic Usage Examples

This page demonstrates basic usage patterns for doc-generator with practical examples.

## 🚀 Getting Started Examples

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
# Use GPT-4 for higher quality
doc-gen --topic "Database Design" --model gpt-4 --temperature 0.2

# Use higher temperature for more creative output
doc-gen --topic "Creative Coding" --model gpt-4 --temperature 0.8
```

## 🔌 Plugin Examples

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

## 📁 Output Management

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

## 🎛️ Configuration Examples

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

## 🔍 Quality and Analysis

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

## 📊 Common Workflows

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

## 🎯 Topic Formulation Tips

### Effective Topics

```bash
# ✅ Good: Specific and descriptive
doc-gen --topic "Python pandas DataFrame operations for time series analysis"
doc-gen --topic "SLURM job arrays for Monte Carlo simulations"
doc-gen --topic "GPU-accelerated machine learning with PyTorch"

# ❌ Poor: Too vague or generic
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

## 🚨 Common Issues and Solutions

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

### Low Quality Output

```bash
# Issue: Generated documentation is too generic or low quality
# Solutions:

# 1. Use more specific topics
doc-gen --topic "Python scikit-learn model training for image classification"

# 2. Use better model
doc-gen --topic "Same Topic" --model gpt-4

# 3. Lower temperature for more focused output
doc-gen --topic "Same Topic" --temperature 0.1

# 4. Generate multiple variants and compare
doc-gen --topic "Same Topic" --runs 5 --analyze
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