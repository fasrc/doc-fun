# Getting Started Guide

This guide will walk you through your first steps with doc-generator, from basic usage to advanced features.

## Overview

Doc-generator is an AI-powered tool that creates high-quality technical documentation using multiple LLM providers (OpenAI GPT and Anthropic Claude). It supports three main capabilities:

1. **Topic Documentation** - Generate comprehensive documentation from simple topic descriptions
2. **README Generation** - Create README.md files for code projects with automatic structure analysis  
3. **Document Standardization** - Transform existing documentation to organizational templates

The system features an extensible plugin architecture that provides intelligent recommendations for HPC modules, code examples, and more.

## Your First Documentation

### Step 1: Basic Generation

Let's create your first piece of documentation:

```bash
# Generate documentation for a simple topic
doc-gen --topic "Python Programming" --output-dir ./my-docs

# Check the output
ls -la my-docs/
cat my-docs/python_programming_gpt4omini_temp03.html
```

**What happened?**
- Doc-generator analyzed the topic "Python Programming"
- The ModuleRecommender plugin suggested relevant HPC modules
- GPT generated structured documentation with recommended modules
- Output saved as HTML file with descriptive filename

### Step 2: Multiple Variants

Generate multiple versions to compare quality:

```bash
# Generate 3 variants with different temperature settings
doc-gen --topic "Machine Learning with GPU" --runs 3 --temperature 0.7

# Compare the generated files
ls -la output/machine_learning_with_gpu_*
```

**Why multiple runs?**
- Different temperature settings produce varied outputs
- You can select the best version manually
- Future versions will include automatic best-variant selection

### Step 3: Custom Output Location

Organize your documentation:

```bash
# Create organized output structure
mkdir -p docs/{tutorials,references,examples}

# Generate documentation in specific location
doc-gen --topic "SLURM Job Scheduling" --output-dir docs/tutorials
doc-gen --topic "MPI Programming" --output-dir docs/references
doc-gen --topic "Python Examples" --output-dir docs/examples
```

## Exploring Different Operation Modes

### README Generation Mode

Generate README files for your code projects:

```bash
# Generate README for a single project
doc-gen --readme /path/to/my-project --output-dir ./readmes

# Recursive generation for multiple projects
doc-gen --readme /path/to/projects --recursive --runs 2

# With analysis and quality evaluation
doc-gen --readme /path/to/project --analyze --output-dir ./output
```

**What happens during README generation:**
- Analyzes project directory structure and file types
- Discovers code examples and configuration files  
- Generates installation and usage instructions
- Creates comprehensive project documentation
- Applies AI enhancement for better descriptions

### Document Standardization Mode

Transform existing documentation to organizational standards:

```bash
# Convert HTML to standardized Markdown
doc-gen --standardize legacy-docs.html --target-format markdown

# Apply organizational template
doc-gen --standardize api-docs.html --template technical_documentation

# Batch standardization
for file in docs/*.html; do
    doc-gen --standardize "$file" --template user_guide --output-dir standardized/
done
```

**What happens during standardization:**
- Extracts content from existing documents (HTML, Markdown)
- Maps content to standardized organizational templates
- Applies consistent formatting and structure
- Preserves original content while improving organization
- Converts between different document formats

### Provider and Model Selection

Choose your preferred AI provider and model for optimal results:

#### Recommended Models (Production-Ready)

```bash
# OpenAI GPT models - Recommended for most use cases
doc-gen --topic "Python Guide" --provider openai --model gpt-4o-mini     # Cost-effective, high quality
doc-gen --topic "Complex Topic" --provider openai --model gpt-4o         # Higher capability
doc-gen --topic "Technical Docs" --provider openai --model gpt-4         # Most reliable

# Anthropic Claude models - Excellent for code analysis and README generation
doc-gen --readme /path/to/project --provider claude --model claude-3-5-sonnet-20241022      # Latest, best overall
doc-gen --readme /path/to/project --provider claude --model claude-3-5-sonnet-20240620      # Stable alternative
doc-gen --topic "Advanced Analysis" --provider claude --model claude-opus-4-1-20250805      # Most capable

# Auto-selection based on available API keys and task type
doc-gen --standardize document.html --provider auto
```

!!! warning "Critical: GPT-5 Model Limitations"
    **Avoid GPT-5 models for production use:**
    ```bash
    # These models have known issues:
    # gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-chat-latest
    # - Often return empty content (API bug)
    # - Only support temperature=1.0 (no customization)
    # - Inconsistent availability
    ```

#### Available Models by Provider

**OpenAI Models:**

| Model | Status | Use Case | Notes |
|-------|--------|----------|-------|
| `gpt-4o-mini` | **Recommended** | General documentation, cost-effective | Best value for most tasks |
| `gpt-4o` | **Recommended** | Complex documentation, analysis | Enhanced capabilities |
| `gpt-4` | **Recommended** | Production documentation | Most stable and reliable |
| `gpt-4-turbo` | Supported | Large document processing | Good for complex tasks |
| `gpt-3.5-turbo` | Supported | Simple documentation, testing | Budget option |
| `gpt-5*` | **Avoid** | Not suitable for production | Known API bugs, limited customization |

**Anthropic Claude Models:**

| Model | Status | Use Case | Notes |
|-------|--------|----------|-------|
| `claude-opus-4-1-20250805` | **Premium** | Most complex tasks | Latest, most capable |
| `claude-3-5-sonnet-20241022` | **Recommended** | General documentation, code analysis | Latest Sonnet, excellent balance |
| `claude-3-5-sonnet-20240620` | **Stable** | Production documentation | Well-tested, reliable |
| `claude-3-5-haiku-20241022` | Supported | Fast, simple tasks | Cost-effective for basic docs |
| `claude-3-haiku-20240307` | Supported | Legacy compatibility | Older but stable |
| `claude-3-sonnet-20240229` | Supported | Basic documentation | Previous generation |

#### Provider Selection Strategy

**Auto-Selection Logic:**
- **Topic Documentation**: Defaults to OpenAI GPT models (most reliable for general topics)
- **README Generation**: Prefers Claude Sonnet models (excellent code analysis)  
- **Document Standardization**: Chooses based on content complexity
- **Fallback**: Uses any available provider if preferred is unavailable

## Understanding Plugins

### View Available Plugins

```bash
# List all loaded plugins
doc-gen --list-plugins
```

**Expected Output:**
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

### Plugin Integration

See how plugins enhance your documentation:

```bash
# Generate documentation and observe module recommendations
doc-gen --topic "Parallel Python with NumPy" --runs 1 --verbose

# The output will include:
# - Recommended HPC modules (python/3.12.8-fasrc01, etc.)
# - Load commands (module load python/3.12.8-fasrc01)
# - Relevant descriptions and categories
```

## Command Line Options

### Essential Options

| Option | Purpose | Example |
|--------|---------|---------|
| `--topic` | Topic documentation | `--topic "CUDA Programming"` |
| `--readme` | README generation | `--readme /path/to/project` |
| `--standardize` | Document standardization | `--standardize docs.html` |
| `--output-dir` | Where to save | `--output-dir ./output` |
| `--runs` | Number of variants | `--runs 3` |
| `--provider` | AI provider | `--provider claude` |
| `--model` | Specific model | `--model gpt-4o-mini` |
| `--template` | Standardization template | `--template technical_documentation` |
| `--target-format` | Output format | `--target-format markdown` |

### Advanced Options

```bash
# Analysis and quality evaluation
doc-gen --topic "Deep Learning" --analyze --quality-eval

# Plugin management
doc-gen --topic "Statistics" --disable-plugins modules
doc-gen --topic "Data Science" --enable-only datasets workflows

# Verbose output for debugging
doc-gen --topic "Debug Test" --verbose
```

## Environment Setup

### API Key Configuration

Doc-generator supports multiple AI providers. Set up API keys for the providers you want to use:

```bash
# For OpenAI GPT models
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic Claude models (optional)
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Or create .env file
cat > .env << 'EOF'
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
EOF
```

### Provider Auto-Detection

If you have multiple API keys configured, doc-generator will:

1. **Auto-detect** available providers based on environment variables
2. **Prioritize** based on the specific operation mode:
   - Topic documentation: OpenAI GPT (default)
   - README generation: Claude Sonnet (preferred for code analysis)  
   - Document standardization: Auto-select based on content type
3. **Fall back** to available providers if the preferred one is unavailable

### Test Your Setup

```bash
# List available models and providers
doc-gen --list-models

# Expected output shows your configured providers:
# Available Providers:
# OpenAI: gpt-3.5-turbo, gpt-4, gpt-4o-mini
# Claude: claude-3-haiku-20240307, claude-3-5-sonnet-20240620

# Test with a simple generation
doc-gen --topic "Test Documentation" --runs 1 --output-dir ./test
```

## Configuration Files

### Understanding the Structure

```
doc-fun/
├── prompts/
│   ├── generator/
│   │   ├── default.yaml      # Main prompt template
│   │   ├── markdown.yaml     # Markdown format template
│   │   ├── readme.yaml       # README generation prompts
│   │   └── custom.yaml       # Your custom templates
│   ├── standardization/
│   │   └── default.yaml      # Document standardization prompts
│   └── analysis/
│       └── default.yaml      # Quality evaluation prompts
├── terminology.yaml          # HPC modules and commands
├── shots/                    # Few-shot learning examples
│   ├── user_docs/           # HTML documentation examples
│   └── user_codes/          # Code-based examples
└── .env                     # API keys and secrets
```

### Customizing Prompts

Create a custom prompt template:

```bash
# Create custom prompt
cat > prompts/generator/my-custom.yaml << 'EOF'
system_prompt: |
  You are a technical writer creating {format} documentation for {topic}.
  Focus on practical examples and clear step-by-step instructions.
  Include troubleshooting sections and best practices.

placeholders:
  format: "HTML"
  organization: "My Organization"

user_prompt: |
  Create comprehensive documentation for: {topic}
  
  Structure:
  1. Overview and Purpose
  2. Prerequisites and Setup
  3. Step-by-Step Instructions
  4. Common Issues and Solutions
  5. Best Practices
  6. Additional Resources
EOF

# Use custom prompt
doc-gen --topic "Custom Topic" --prompt-yaml prompts/generator/my-custom.yaml
```

### Modifying Terminology

Add your own HPC modules:

```bash
# Edit terminology.yaml
cat >> terminology.yaml << 'EOF'
hpc_modules:
  - name: "my-custom-tool/1.0.0"
    description: "My organization's custom tool"
    category: "custom"
  - name: "internal-library/2.1.0"  
    description: "Internal computational library"
    category: "library"
EOF
```

## 🔬 Advanced Features

### Code Scanning

Automatically discover code examples in your project:

```bash
# Scan current directory for code examples
doc-gen --scan-code . --max-scan-files 100

# Generate documentation with discovered examples
doc-gen --topic "Project Overview" --runs 1
```

### Quality Analysis

Get detailed quality reports:

```bash
# Generate with analysis
doc-gen --topic "Algorithm Implementation" --runs 3 --analyze --quality-eval

# Check generated analysis files
ls -la output/*analysis*
ls -la output/*evaluation*
```

### Batch Processing

Process multiple topics efficiently:

```bash
# Create topics file
cat > topics.txt << 'EOF'
Python Data Analysis
R Statistical Computing
MATLAB Numerical Methods
Julia High Performance
EOF

# Process all topics
while read topic; do
    echo "Processing: $topic"
    doc-gen --topic "$topic" --output-dir "batch-output"
    sleep 2  # Rate limiting
done < topics.txt
```

## Output Formats

### HTML (Default)

```bash
# Generate HTML documentation
doc-gen --topic "Web Documentation" --output-dir html-docs

# View in browser
open html-docs/*.html
```

### Markdown

```bash
# Use markdown template
doc-gen --topic "GitHub Documentation" \
  --prompt-yaml prompts/generator/markdown.yaml \
  --output-dir markdown-docs
```

### Custom Formatting

Modify prompt templates to control output format:

```yaml
# In your prompt template
system_prompt: |
  Generate documentation in {format} format.
  Use proper {format} syntax and structure.
  
placeholders:
  format: "reStructuredText"  # or "LaTeX", "AsciiDoc", etc.
```

## Quality Comparison

Compare your generated documentation with existing references to ensure quality:

```bash
# Compare with online documentation
doc-gen --topic "Python Data Analysis" \
        --compare-url https://pandas.pydata.org/docs/getting_started/intro_tutorials/ \
        --comparison-report quality_report.md

# View similarity scores
cat quality_report.md | grep "Composite Score"
# Output: Composite Score: 75.3%
```

The comparison feature helps you:
- Benchmark against gold-standard documentation
- Identify missing sections or content
- Optimize generation parameters
- Track quality improvements

[Learn more about comparison →](comparison.md)

## 🔄 Workflow Examples

### Academic Research Workflow

```bash
# 1. Document research methodology
doc-gen --topic "Computational Biology Pipeline" --output-dir research/methods

# 2. Create software documentation
doc-gen --topic "Bioinformatics Tools Setup" --output-dir research/software

# 3. Generate user guides
doc-gen --topic "Running Genomic Analysis" --output-dir research/guides

# 4. Create troubleshooting docs
doc-gen --topic "Common Pipeline Errors" --output-dir research/troubleshooting
```

### Software Development Workflow

```bash
# 1. API documentation
doc-gen --topic "REST API Reference" --output-dir docs/api

# 2. Installation guides
doc-gen --topic "Development Environment Setup" --output-dir docs/setup

# 3. User tutorials
doc-gen --topic "Getting Started Tutorial" --output-dir docs/tutorials

# 4. Deployment guides
doc-gen --topic "Production Deployment" --output-dir docs/deployment
```

### HPC User Support Workflow

```bash
# 1. User onboarding
doc-gen --topic "New User Guide" --output-dir support/onboarding

# 2. Software tutorials
doc-gen --topic "SLURM Job Submission" --output-dir support/tutorials

# 3. Troubleshooting guides
doc-gen --topic "Common SLURM Issues" --output-dir support/troubleshooting

# 4. Best practices
doc-gen --topic "Cluster Resource Optimization" --output-dir support/best-practices
```

## 🐛 Debugging and Troubleshooting

### Common Issues

#### Issue: No module recommendations appearing

```bash
# Check if modules plugin is loaded
doc-gen --list-plugins

# Verify terminology file has modules
head -20 terminology.yaml

# Test module recommender directly
python -c "
from doc_generator.plugins.modules import ModuleRecommender
import yaml
with open('terminology.yaml') as f:
    term = yaml.safe_load(f)
rec = ModuleRecommender(terminology=term)
print(rec.get_recommendations('Python'))
"
```

#### Issue: Poor quality output

```bash
# Try different temperature settings
doc-gen --topic "Same Topic" --temperature 0.1  # More focused
doc-gen --topic "Same Topic" --temperature 0.7  # More creative

# Use more specific topics
doc-gen --topic "Python pandas DataFrame operations for time series analysis"
# instead of
doc-gen --topic "Python"

# Add context to topics
doc-gen --topic "SLURM job arrays for Monte Carlo simulations on FASRC cluster"
```

#### Issue: API rate limiting

```bash
# Add delays between requests
doc-gen --topic "Topic 1" && sleep 5 && doc-gen --topic "Topic 2"

# Use lower-cost models for testing
doc-gen --topic "Test Topic" --model gpt-3.5-turbo

# Monitor API usage
# Check your OpenAI dashboard for current usage
```

### Verbose Mode

Get detailed information about what's happening:

```bash
# Enable verbose logging
doc-gen --topic "Debug Topic" --verbose

# This shows:
# - Plugin loading process
# - API requests and responses
# - File operations
# - Error details
```

## Performance Tips

### Optimize Generation Speed

```bash
# Use faster model for drafts
doc-gen --topic "Draft Content" --model gpt-3.5-turbo

# Reduce runs for testing
doc-gen --topic "Test Topic" --runs 1

# Use specific topics to reduce API token usage
doc-gen --topic "Specific numpy array slicing techniques"
```

### Optimize Quality

```bash
# Use higher-quality model for final versions
doc-gen --topic "Production Documentation" --model gpt-4

# Generate multiple variants and compare
doc-gen --topic "Important Guide" --runs 5 --analyze

# Use lower temperature for technical accuracy
doc-gen --topic "API Reference" --temperature 0.2
```

## 🎓 Best Practices

### Topic Formulation

**Good Topics:**
- "SLURM job arrays for parameter sweeps on FASRC"
- "PyTorch distributed training with multiple GPUs"
- "R parallel processing with foreach and doParallel"

**Poor Topics:**
- "Help"
- "Programming"
- "Cluster"

### Documentation Organization

```bash
# Use consistent directory structure
mkdir -p docs/{user-guides,admin-guides,tutorials,reference,troubleshooting}

# Use descriptive output directories
doc-gen --topic "User Guide" --output-dir docs/user-guides
doc-gen --topic "Admin Tasks" --output-dir docs/admin-guides
```

### Quality Assurance

```bash
# Always review generated content
doc-gen --topic "Critical Documentation" --runs 3 --analyze

# Test recommendations manually
# If doc suggests "module load python/3.12.8-fasrc01"
# Verify: module avail python

# Version control your configs
git add prompts/ terminology.yaml
git commit -m "Update documentation templates"
```

## Next Steps

Now that you understand the basics:

1. **[Learn Testing](testing.md)** - Run and create tests
2. **[Create Plugins](creating-plugins.md)** - Extend functionality  
3. **[Advanced Configuration](configuration.md)** - Customize templates
4. 🤝 **[Contributing](contributing.md)** - Help improve doc-generator

## You're Ready!

You now have the knowledge to:
- Generate high-quality documentation
- Customize prompts and templates
- Use plugins effectively
- Optimize for your workflow
- Debug common issues

**Happy documenting!**