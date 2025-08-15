# doc-generator

> AI-powered documentation generator for any topic with extensible plugin architecture

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-57%20passed-green.svg)](#testing)

## Overview

**doc-generator** is a sophisticated Python package that automates the creation of high-quality technical documentation using OpenAI's GPT models. Originally designed for Faculty Arts and Sciences Research Computing (FASRC), it features an extensible plugin architecture that allows intelligent recommendations for HPC modules, datasets, code examples, and more.

## ✨ Key Features

### 🧠 **AI-Powered Generation**
- OpenAI GPT integration with customizable models and parameters
- Few-shot prompting with curated examples for consistent output
- Parameterized prompt templates with runtime customization
- Multiple format support (HTML, Markdown)

### 🔌 **Plugin Architecture**
- **Extensible recommendation engines** via entry points
- **Built-in ModuleRecommender** for HPC module suggestions
- **Third-party plugin support** with automatic discovery
- **Priority-based ordering** and plugin management

### 📊 **Quality Assurance**
- **Algorithmic analysis** with section scoring and structure validation
- **GPT-based quality evaluation** for technical accuracy assessment
- **Multi-run generation** with best-variant compilation
- **Comprehensive testing** with 57 test cases

### 🛠️ **Professional Tooling**
- **CLI interface** with comprehensive options
- **Code scanning** for automatic example discovery
- **Batch processing** capabilities
- **Development-ready** package structure

## 🚀 Quick Start

### Installation

```bash
# Install the package
pip install -e .

# For development and testing (includes pytest-cov)
pip install -e ".[test]"

# Verify installation
doc-gen --version
```

### Basic Usage

```bash
# Generate documentation
doc-gen --topic "Python Programming" --output-dir ./output

# List available plugins
doc-gen --list-plugins

# Multiple runs with analysis
doc-gen --topic "Machine Learning" --runs 3 --analyze --quality-eval
```

### API Usage

```python
from doc_generator import DocumentationGenerator

# Initialize generator
generator = DocumentationGenerator(
    prompt_yaml_path="./prompts/generator/default.yaml",
    terminology_path="./terminology.yaml"
)

# Generate documentation
results = generator.generate_documentation(
    query="Parallel Computing with MPI",
    runs=2,
    model="gpt-4o-mini"
)
```

## 🏗️ Architecture

### Plugin System

The extensible plugin architecture allows third-party developers to create specialized recommendation engines:

```python
from doc_generator.plugins import RecommendationEngine

class DatasetRecommender(RecommendationEngine):
    def get_name(self) -> str:
        return "datasets"
    
    def get_recommendations(self, topic: str, context=None):
        # Your recommendation logic here
        return [{"title": "Relevant Dataset", "score": 8.5}]
```

### Package Structure

```
src/doc_generator/
├── __init__.py              # Package exports
├── core.py                  # Main DocumentationGenerator
├── cli.py                   # Command-line interface
├── plugin_manager.py        # Plugin discovery and management
└── plugins/
    ├── __init__.py         # Plugin interface exports
    ├── base.py             # RecommendationEngine base class
    └── modules.py          # Built-in ModuleRecommender
```

## 📖 Documentation Structure

Generated documentation follows a consistent structure:
- **Description**: Overview and purpose
- **Installation**: Setup instructions with recommended modules
- **Usage**: Step-by-step usage examples
- **Examples**: Practical code examples and workflows
- **References**: Relevant links and documentation

## 🔧 Configuration

### Prompt Templates

Create customized prompt templates with parameterization:

```yaml
# prompts/generator/custom.yaml
system_prompt: |
  You are creating {format} documentation for {topic} at {organization}.
  Focus on practical examples and clear instructions.

placeholders:
  format: "HTML"
  organization: "FASRC"
```

### Terminology Configuration

Define HPC modules, cluster commands, and code examples:

```yaml
# terminology.yaml
hpc_modules:
  - name: "python/3.12.8-fasrc01"
    description: "Python 3.12 with Anaconda distribution"
    category: "programming"

cluster_commands:
  - name: "sbatch"
    description: "Submit a batch job"
    usage: "sbatch script.sh"
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Install test dependencies first
pip install -e ".[test]"

# Run all tests
python -m pytest -v

# Run with coverage
python -m pytest --cov=src/doc_generator --cov-report=html

# Run specific test categories
python -m pytest tests/test_plugin_manager.py -v
```

**Test Coverage:**
- 57 tests covering all major functionality
- Plugin system integration tests
- Error handling and edge case validation
- Mock systems for external dependencies

## 🔌 Creating Plugins

### Third-Party Plugin Example

**1. Create your plugin package:**

```python
# my_plugin/dataset_recommender.py
from doc_generator.plugins import RecommendationEngine

class DatasetRecommender(RecommendationEngine):
    def get_name(self) -> str:
        return "datasets"
    
    def get_recommendations(self, topic: str, context=None):
        # Search relevant datasets
        return [
            {
                "title": "Climate Dataset XYZ",
                "url": "https://data.example.com/climate",
                "relevance_score": 9.2
            }
        ]
```

**2. Configure entry points:**

```toml
# pyproject.toml
[project.entry-points."doc_generator.plugins"]
datasets = "my_plugin.dataset_recommender:DatasetRecommender"
```

**3. Install and use:**

```bash
pip install my-dataset-plugin
doc-gen --topic "Climate Science" --output-dir ./output
# Dataset recommendations automatically included!
```

## 📋 Command Reference

### Core Commands

```bash
# Basic generation
doc-gen --topic "TOPIC" [options]

# Plugin management
doc-gen --list-plugins
doc-gen --disable-plugins modules
doc-gen --enable-only datasets workflows

# Analysis and evaluation
doc-gen --topic "TOPIC" --analyze --quality-eval
doc-gen --topic "TOPIC" --runs 5 --model gpt-4

# Code scanning
doc-gen --scan-code ./src --max-scan-files 100
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--topic` | Topic for documentation generation | Required |
| `--output-dir` | Output directory | `output` |
| `--runs` | Number of variants to generate | `1` |
| `--model` | OpenAI model to use | `gpt-4o-mini` |
| `--temperature` | Generation temperature | `0.3` |
| `--analyze` | Run document analysis | `False` |
| `--quality-eval` | Run GPT quality evaluation | `False` |

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-plugin`
3. **Run tests**: `python -m pytest -v`
4. **Submit a pull request**

### Development Setup

```bash
# Clone repository
git clone https://github.com/fasrc/doc-fun.git
cd doc-fun

# Install in development mode
pip install -e ".[dev]"

# Install documentation dependencies
pip install -e ".[docs]"

# Run tests
python -m pytest -v
```

### Documentation

The project uses MkDocs with Material theme for documentation:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Serve documentation locally (http://127.0.0.1:8000)
mkdocs serve

# Build static documentation site
mkdocs build

# Deploy to GitHub Pages (requires permissions)
mkdocs gh-deploy
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: [GitHub Repository](https://github.com/fasrc/doc-fun)
- **Issues**: [Bug Tracker](https://github.com/fasrc/doc-fun/issues)
- **FASRC**: [Research Computing](https://www.rc.fas.harvard.edu/)

## 📊 Version History

### v2.0.0 (Current)
- ✅ **Professional Documentation**: Comprehensive MkDocs site with Material theme
- ✅ **Auto-Generated API Docs**: mkdocstrings integration with source code
- ✅ **GitHub Pages Deployment**: Automated documentation hosting
- ✅ **Complete User Guides**: Installation, getting started, testing, plugins, contributing
- ✅ **Advanced Examples**: Basic usage, workflows, plugin development, troubleshooting
- ✅ **Plugin Architecture**: Extensible recommendation engine system

### v1.1.0
- ✅ **Plugin Architecture**: Extensible recommendation engine system
- ✅ **CLI Interface**: Professional command-line tool
- ✅ **Enhanced Testing**: 57 comprehensive test cases
- ✅ **ModuleRecommender**: Intelligent HPC module suggestions
- ✅ **Quality Pipeline**: Multi-run generation with analysis
- ✅ **Package Structure**: Professional Python package layout

### v1.0.0 (Legacy)
- Basic documentation generation script
- OpenAI integration
- Few-shot prompting system
- Jupyter notebook interface

---

**Built with ❤️ by FASRC Research Computing**
