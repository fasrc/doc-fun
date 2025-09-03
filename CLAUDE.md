# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

doc-generator is an AI-powered documentation generation system that supports multiple LLM providers (OpenAI GPT and Anthropic Claude) to create high-quality technical documentation. Originally designed for Faculty Arts and Sciences Research Computing (FASRC), it features an extensible plugin architecture, multi-format support, and comprehensive quality assurance capabilities.

## Key Commands

### CLI Interface

**New Command Pattern (Recommended)**
```bash
# Install the package
pip install -e .

# Basic documentation generation
doc-gen generate "Python Programming" --runs 3 --analyze

# README generation for directories  
doc-gen readme /path/to/directory --recursive --output-dir ./output

# Document standardization
doc-gen standardize document.html --target-format markdown

# List available models and providers
doc-gen list-models

# List and manage plugins
doc-gen list-plugins
doc-gen generate "Topic" --disable-plugins module_recommender

# Clean up output directory
doc-gen cleanup

# Display detailed help information
doc-gen info
```

**Legacy Interface (Still Supported)**
```bash
# All original flag-based commands continue to work
doc-gen --topic "Python Programming" --runs 3 --analyze
doc-gen --readme /path/to/directory --recursive --output-dir ./output
doc-gen --list-models
doc-gen --cleanup
doc-gen --info
```

### Testing
```bash
# Run all tests (recommended - uses parallel execution for better isolation)
pytest -n auto

# Run all tests (sequential - may have isolation issues)
pytest

# Run with coverage
pytest --cov=doc_generator tests/

# Run specific test categories
pytest -m unit
pytest -m integration

# Run specific test files
pytest tests/test_doc_generator.py
pytest tests/providers/

# Note: Using pytest-xdist (-n auto) is recommended as it runs tests in 
# isolated processes, preventing mock state leakage between tests.
```

## Architecture

### CLI Architecture (Phase 2 Refactoring)

**Command Pattern Structure** (`src/doc_generator/cli_commands/`)
- **BaseCommand**: Abstract base class for all CLI commands with validation and error handling
- **CommandRegistry**: Registration system supporting command lookup by name and aliases
- **CommandDispatcher**: Argument parsing, routing, and execution with consistent error handling
- **Bootstrap System**: Automatic command discovery and registration
- **Backward Compatibility**: Intelligent detection routes old `--flag` vs new `command` arguments

**Available Commands**:
- `generate` (aliases: `gen`, `g`) - Technical documentation generation (replaces `--topic`)
- `readme` (alias: `r`) - README.md generation for directories (replaces `--readme`) 
- `standardize` (aliases: `std`, `s`) - Document standardization (replaces `--standardize`)
- `list-models` (aliases: `lm`, `models`) - List available AI models and providers
- `cleanup` (alias: `clean`) - Clean output directory with confirmation
- `info` (alias: `help-detailed`) - Display comprehensive help information
- `list-plugins` (aliases: `lp`, `plugins`) - List available plugins

### Core Components

**DocumentationGenerator** (`src/doc_generator/core.py`)
- Main class supporting multiple LLM providers (OpenAI, Claude)
- Provider management through ProviderManager
- Plugin system integration for extensible recommendations
- Few-shot prompting with examples from `shots/` directory
- Support for HTML and Markdown output formats

**ReadmeGenerator** (`src/doc_generator/readme_generator.py`)
- Specialized generator for README.md files
- Directory structure analysis and context building
- Recursive generation for subdirectories
- AI-enhanced descriptions with code example scanning

**Provider System** (`src/doc_generator/providers/`)
- `ProviderManager`: Automatic provider detection and management
- `OpenAIProvider`: GPT-3.5, GPT-4, GPT-4o, GPT-5 model families
- `ClaudeProvider`: Claude 3 (Haiku, Sonnet, Opus) models
- Unified API through `CompletionRequest/Response` classes

**Plugin Architecture** (`src/doc_generator/plugins/`)
- `PluginManager`: Dynamic plugin loading via entry points
- `RecommendationEngine`: Base class for recommendation plugins
- `AnalysisPlugin`: Base class for analysis extensions
- Built-in plugins: ModuleRecommender, DocumentCompiler, LinkValidator

**Analysis Components**
- `DocumentAnalyzer`: Algorithmic scoring and structure validation
- `GPTQualityEvaluator`: AI-based quality assessment
- `DocumentationComparator`: URL and file comparison capabilities
- `AnalysisReporter`: Multi-format report generation

### File Structure

```
doc-generator/
├── src/doc_generator/
│   ├── __init__.py              # Package initialization
│   ├── cli.py                   # Command-line interface
│   ├── core.py                  # Core generation classes
│   ├── plugin_manager.py        # Plugin system
│   ├── readme_generator.py      # README generation
│   ├── providers/               # LLM provider implementations
│   │   ├── base.py             # Abstract provider interface
│   │   ├── manager.py          # Provider management
│   │   ├── openai_provider.py  # OpenAI implementation
│   │   └── claude_provider.py  # Anthropic implementation
│   ├── plugins/                 # Plugin implementations
│   │   ├── base.py             # Plugin base classes
│   │   └── modules.py          # Module recommender
│   ├── analysis/               # Analysis plugins
│   │   ├── compiler.py         # Document compilation
│   │   ├── reporter.py         # Report generation
│   │   └── link_validator.py   # Link validation
│   └── evaluator/              # Evaluation components
│       ├── comparator.py       # Document comparison
│       ├── metrics.py          # Similarity metrics
│       └── downloader.py       # URL content fetching
├── prompts/                     # Prompt configurations
│   ├── generator/              # Generation prompts
│   │   ├── default.yaml        # Default HTML generation
│   │   ├── markdown.yaml       # Markdown generation
│   │   └── readme.yaml         # README generation
│   └── analysis/               # Analysis prompts
│       └── default.yaml        # Default analysis
├── shots/                       # Few-shot examples
│   ├── user_docs/              # HTML documentation examples
│   └── user_codes/             # Code-based examples
├── tests/                       # Test suite
│   ├── providers/              # Provider tests
│   ├── test_*.py              # Unit tests
│   └── conftest.py            # Test configuration
├── pyproject.toml              # Package configuration
├── pytest.ini                  # Test configuration
└── terminology.yaml            # Domain terminology

```

### Configuration System

**Prompt Templates** (`prompts/`)
- YAML-based configuration for system and user prompts
- Parameterized templates with runtime substitution
- Format-specific configurations (HTML, Markdown, README)

**Terminology** (`terminology.yaml`)
- Domain-specific terms and definitions
- Module mappings and categorization
- Custom vocabulary for specialized topics

**Entry Points** (`pyproject.toml`)
- Plugin discovery mechanism
- Analysis extensions registration
- Third-party integration support

### Quality Evaluation Pipeline

1. **Multi-Run Generation**: Generate multiple documentation variants
2. **Algorithmic Analysis**: Section scoring, structure validation, metrics
3. **AI Evaluation**: GPT-based quality assessment and scoring
4. **Comparison**: Compare with existing documentation (URL/file)
5. **Compilation**: Select best sections across variants
6. **Reporting**: Generate comprehensive analysis reports

## Environment Setup

### Required Environment Variables
```bash
# For OpenAI provider
OPENAI_API_KEY=your-openai-key

# For Claude provider (optional)
ANTHROPIC_API_KEY=your-anthropic-key
```

### Installation
```bash
# Development installation
pip install -e .[dev]

# Testing dependencies
pip install -e .[test]
```

## Output Structure

### Documentation Files
- Topic docs: `{topic}_{provider}_{model}_temp{temperature}_v{number}.{format}`
- README files: `{directory}_readme_v{number}.md`
- Compilations: `{topic}_best_compilation.{format}`

### Analysis Reports
- Analysis: `{topic}_analysis_report.{format}`
- GPT evaluation: `{topic}_gpt_evaluation_report.{format}`
- Comparison: `{topic}_comparison_report.{format}`

## Development Guidelines

### Adding a New Plugin
1. Create plugin class inheriting from `RecommendationEngine` or `AnalysisPlugin`
2. Implement required methods (`get_name()`, `get_recommendations()`, etc.)
3. Register in `pyproject.toml` under appropriate entry point
4. Plugin will be auto-discovered on next run

### Adding a New Provider
1. Create provider class inheriting from `LLMProvider`
2. Implement `complete()` method for API interaction
3. Add to `ProviderManager` for automatic detection
4. Set appropriate environment variable for API key

### Testing
- **Recommended**: Use `pytest -n auto` for isolated parallel execution
- Unit tests: Focus on individual components
- Integration tests: Test provider and plugin interactions
- Use pytest fixtures for mock data and configurations
- Mark slow tests with `@pytest.mark.slow`
- **Mock Setup**: Provider tests require proper tuple returns from `validate_model_provider_combination()`
- **Test Isolation**: pytest-xdist prevents mock state leakage between tests

## CLI Operation Modes

### Topic Mode (HTML/Markdown Generation)
```bash
doc-gen --topic "Topic Name" [options]
```
Generates technical documentation in HTML or Markdown format.

### README Mode (Directory Documentation)
```bash
doc-gen --readme /path/to/directory [options]
```
Generates README.md files for code directories with optional recursion.

### Code Scanning Mode (Legacy)
```bash
doc-gen --scan-code /path/to/directory [options]
```
Scans directories for code examples and updates terminology.

### Utility Commands
- `--list-models`: Show available models and providers
- `--list-plugins`: Display loaded plugins
- `--cleanup`: Remove all files and directories in ./output/ (with confirmation)
- `--info`: Display comprehensive help with detailed option descriptions and examples
- `--version`: Show package version