# Configuration Guide

This guide covers how to customize doc-generator's behavior through configuration files, environment variables, and runtime parameters.

## 🎛️ Configuration Overview

Doc-generator uses multiple configuration layers:

1. **Prompt Templates** (`prompts/`) - Control AI generation behavior
2. **Terminology Files** (`terminology.yaml`) - Define HPC modules and commands  
3. **Environment Variables** (`.env`) - API keys and runtime settings
4. **Command-line Options** - Override defaults per execution

## 📝 Prompt Templates

### Default Template Structure

```yaml
# prompts/generator/default.yaml
system_prompt: |
  You are creating {format} documentation for {topic} at {organization}.
  Focus on practical examples and clear step-by-step instructions.
  
  Structure your response with these sections:
  1. Description - Brief overview and purpose
  2. Installation - Setup instructions with module commands
  3. Usage - Basic usage examples and syntax
  4. Examples - Practical code examples and workflows
  5. References - Relevant links and documentation

placeholders:
  format: "HTML"
  organization: "FASRC"
  
user_prompt: |
  Create comprehensive documentation for: {topic}
  
  Include relevant HPC modules, cluster commands, and best practices.
  Use clear code examples and provide troubleshooting tips.
```

### Custom Templates

Create specialized templates for different use cases:

```yaml
# prompts/generator/api-docs.yaml
system_prompt: |
  You are creating API documentation for {topic}.
  Focus on endpoints, parameters, and response examples.
  
  Required sections:
  1. Overview - API purpose and authentication
  2. Endpoints - Available endpoints with HTTP methods
  3. Parameters - Request/response parameters
  4. Examples - Code examples in multiple languages
  5. Error Handling - Common errors and solutions

placeholders:
  format: "HTML"
  api_version: "v1"
  base_url: "https://api.example.com"

user_prompt: |
  Document the {topic} API with complete endpoint reference.
  Include curl examples and response schemas.
```

### Template Parameters

Templates support dynamic placeholders:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `{topic}` | Documentation topic | Required |
| `{format}` | Output format | `"HTML"` |
| `{organization}` | Organization name | `"FASRC"` |
| `{date}` | Current date | Auto-generated |
| `{model}` | AI model used | Runtime value |

## 🏗️ Terminology Configuration

### HPC Modules

Define available software modules:

```yaml
# terminology.yaml
hpc_modules:
  - name: "python/3.12.8-fasrc01"
    description: "Python 3.12 with Anaconda distribution"
    category: "programming"
    keywords: ["python", "anaconda", "data science"]
    
  - name: "gcc/12.2.0-fasrc01"  
    description: "GNU Compiler Collection 12.2"
    category: "compiler"
    keywords: ["gcc", "compiler", "c", "cpp", "fortran"]
    
  - name: "cuda/12.9.1-fasrc01"
    description: "NVIDIA CUDA Toolkit 12.9"
    category: "gpu"
    keywords: ["cuda", "gpu", "nvidia", "machine learning"]
```

### Cluster Commands

Define common cluster operations:

```yaml
cluster_commands:
  - name: "sbatch"
    description: "Submit a batch job to SLURM"
    usage: "sbatch script.sh"
    category: "job_management"
    
  - name: "squeue"
    description: "View job queue status"
    usage: "squeue -u $USER"
    category: "monitoring"
    
  - name: "module load"
    description: "Load software module"
    usage: "module load python/3.12.8-fasrc01"
    category: "environment"
```

### Code Examples

Pre-define reusable code snippets:

```yaml
code_examples:
  python_import:
    language: "python"
    code: |
      import numpy as np
      import pandas as pd
      import matplotlib.pyplot as plt
    description: "Standard Python data science imports"
    
  slurm_header:
    language: "bash"
    code: |
      #!/bin/bash
      #SBATCH -J my_job
      #SBATCH -p shared
      #SBATCH -t 1:00:00
      #SBATCH --mem=4G
    description: "Basic SLURM job script header"
```

## 🔧 Environment Configuration

### API Keys

```bash
# .env file
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_ORGANIZATION=your-org-id  # Optional
```

### Runtime Settings

```bash
# Default model settings
DOC_GEN_DEFAULT_MODEL=gpt-4o-mini
DOC_GEN_DEFAULT_TEMPERATURE=0.3
DOC_GEN_DEFAULT_RUNS=1

# Output settings
DOC_GEN_OUTPUT_DIR=output
DOC_GEN_MAX_TOKENS=4000

# Plugin settings
DOC_GEN_DISABLE_PLUGINS=  # Comma-separated list
DOC_GEN_PLUGIN_PRIORITY=modules:100,datasets:90

# Debug settings
DOC_GEN_VERBOSE=false
DOC_GEN_LOG_LEVEL=INFO
```

## 🎯 Command-Line Configuration

### Global Options

```bash
# Model and generation settings
doc-gen --model gpt-4 --temperature 0.5 --runs 3

# Output configuration  
doc-gen --output-dir custom-docs --format html

# Plugin management
doc-gen --disable-plugins modules --enable-only datasets workflows

# Quality and analysis
doc-gen --analyze --quality-eval --verbose
```

### Configuration Files Override

```bash
# Use custom prompt template
doc-gen --prompt-yaml prompts/generator/custom.yaml

# Use custom terminology
doc-gen --terminology-path config/custom-terminology.yaml

# Use custom examples directory
doc-gen --examples-dir examples/specialized/
```

## 🔌 Plugin Configuration

### Plugin-Specific Settings

```yaml
# terminology.yaml - Plugin configuration section
plugin_config:
  modules:
    priority: 100
    enabled: true
    max_recommendations: 5
    categories: ["programming", "compiler", "library"]
    
  datasets:
    priority: 90
    enabled: false
    api_timeout: 10
    cache_duration: 3600
    sources: ["zenodo", "datahub"]
```

### Environment-Based Plugin Config

```bash
# ModuleRecommender settings
MODULE_RECOMMENDER_MAX_RESULTS=5
MODULE_RECOMMENDER_MIN_SCORE=3.0
MODULE_RECOMMENDER_CATEGORIES=programming,compiler

# DatasetRecommender settings (TBD - Plugin not yet implemented)
DATASET_RECOMMENDER_TIMEOUT=10
DATASET_RECOMMENDER_CACHE_TTL=3600
DATASET_RECOMMENDER_SOURCES=zenodo,datahub,nasa
```

## 📁 Directory Structure

### Standard Configuration Layout

```
doc-fun/
├── .env                      # Environment variables and API keys
├── terminology.yaml          # HPC modules and cluster commands
├── prompts/
│   ├── generator/
│   │   ├── default.yaml     # Standard documentation template
│   │   ├── markdown.yaml    # Markdown output template
│   │   ├── api-docs.yaml    # API documentation template
│   │   └── tutorial.yaml    # Tutorial-focused template
│   └── analysis/
│       ├── default.yaml     # Quality evaluation prompts
│       └── technical.yaml   # Technical accuracy evaluation
├── examples/                 # Few-shot learning examples
│   ├── matlab.html
│   ├── mpi.html
│   └── python.html
└── config/                   # Optional: additional config files
    ├── development.yaml
    ├── production.yaml
    └── custom-terminology.yaml
```

### Custom Configuration Paths

```bash
# Override default paths
export DOC_GEN_CONFIG_DIR=/path/to/custom/config
export DOC_GEN_PROMPTS_DIR=/path/to/custom/prompts
export DOC_GEN_TERMINOLOGY_PATH=/path/to/custom/terminology.yaml
```

## 🎨 Output Customization

### HTML Styling

Create custom CSS for generated HTML:

```css
/* styles/custom.css */
.doc-generator-output {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto;
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

.code-block {
    background: #f8f9fa;
    border-left: 4px solid #007bff;
    padding: 1rem;
    margin: 1rem 0;
}

.module-recommendation {
    background: #e7f3ff;
    border: 1px solid #b8daff;
    border-radius: 4px;
    padding: 0.75rem;
    margin: 0.5rem 0;
}
```

### Template Customization

```yaml
# prompts/generator/styled.yaml
system_prompt: |
  Create HTML documentation with custom styling.
  Include this CSS in the <head> section:
  <link rel="stylesheet" href="styles/custom.css">
  
  Use these CSS classes:
  - .doc-generator-output for main container
  - .code-block for code examples
  - .module-recommendation for HPC module suggestions

placeholders:
  css_framework: "custom"
  include_toc: true
```

## 🔄 Configuration Profiles

### Profile-Based Configuration

Create different configuration profiles:

```yaml
# config/profiles.yaml
profiles:
  development:
    model: "gpt-3.5-turbo"
    temperature: 0.7
    runs: 1
    verbose: true
    
  production:
    model: "gpt-4"
    temperature: 0.3
    runs: 3
    analyze: true
    quality_eval: true
    
  research:
    model: "gpt-4"
    temperature: 0.2
    runs: 5
    plugins: ["modules", "datasets", "papers"]
    output_format: "academic"
```

Usage:

```bash
# Use specific profile
doc-gen --profile production --topic "Critical Documentation"

# Override profile settings
doc-gen --profile development --model gpt-4 --topic "Test Topic"
```

## 🧪 Advanced Configuration

### Conditional Configuration

```yaml
# prompts/generator/conditional.yaml
system_prompt: |
  {% if topic contains "python" %}
  Focus on Python-specific examples and best practices.
  Include pip installation commands and virtual environments.
  {% elif topic contains "gpu" %}
  Emphasize GPU computing concepts and CUDA examples.
  Include module load commands for CUDA toolkit.
  {% else %}
  Provide general HPC guidance and cluster usage examples.
  {% endif %}
  
  Create documentation for: {topic}

placeholders:
  topic_type: "auto-detected"
```

### Multi-Organization Support

```yaml
# config/organizations.yaml
organizations:
  fasrc:
    name: "Faculty Arts and Sciences Research Computing"
    modules_prefix: "fasrc01"
    scheduler: "slurm"
    documentation_url: "https://docs.rc.fas.harvard.edu"
    
  mit:
    name: "MIT SuperCloud"
    modules_prefix: "mit"
    scheduler: "slurm"  
    documentation_url: "https://supercloud.mit.edu"
    
  nsf:
    name: "XSEDE/ACCESS"
    modules_prefix: "xsede"
    scheduler: "pbs"
    documentation_url: "https://access-ci.org"
```

## 📊 Configuration Validation

### Validate Configuration

```bash
# Check configuration syntax
doc-gen --validate-config

# Test with dry run
doc-gen --topic "Test" --dry-run --verbose

# Validate specific files
doc-gen --validate-prompts prompts/generator/
doc-gen --validate-terminology terminology.yaml
```

### Configuration Schema

```python
# config/schema.py
from typing import Dict, List, Optional
from pydantic import BaseModel

class PromptConfig(BaseModel):
    system_prompt: str
    user_prompt: str
    placeholders: Dict[str, str] = {}

class ModuleConfig(BaseModel):
    name: str
    description: str
    category: str
    keywords: List[str] = []

class TerminologyConfig(BaseModel):
    hpc_modules: List[ModuleConfig]
    cluster_commands: List[Dict[str, str]] = []
    code_examples: Dict[str, Dict] = {}
```

## ✅ Configuration Best Practices

### Security
- **Never commit API keys** to version control
- **Use environment variables** for sensitive data
- **Rotate API keys** regularly
- **Limit API key permissions** when possible

### Organization
- **Use descriptive file names** for templates
- **Group related configurations** in directories
- **Document custom placeholders** and their usage
- **Version control configuration files** (except .env)

### Performance
- **Cache frequently used configurations** in memory
- **Use specific prompts** rather than generic ones
- **Optimize token usage** with concise templates
- **Monitor API costs** with usage tracking

### Maintenance
- **Test configuration changes** before deploying
- **Keep templates up to date** with model capabilities
- **Document custom configurations** for team members
- **Regular backup** of configuration files

---

Ready to customize doc-generator for your needs? Start with the [Getting Started Guide](getting-started.md) to learn basic usage, then return here to tailor the system to your requirements.