# Project Structure

## Root Directory
```
doc-fun/
├── docs/                     # Documentation and design documents
├── prompts/                  # YAML prompt templates for generation
├── shots/                    # Few-shot examples for AI training
├── src/doc_generator/        # Main source code package
├── tests/                    # Test suite
├── tasks/                    # Task-related files
├── reference/                # Reference materials
├── README.md                 # Project documentation
├── CLAUDE.md                 # Claude Code guidance
├── pyproject.toml           # Project configuration and dependencies
├── pytest.ini              # Testing configuration
├── mkdocs.yml              # Documentation site configuration
└── terminology.yaml         # HPC modules and terminology definitions
```

## Source Code Structure (src/doc_generator/)
```
src/doc_generator/
├── __init__.py              # Package initialization and exports
├── core.py                  # Main DocumentationGenerator class
├── cli.py                   # Command-line interface
├── plugin_manager.py        # Plugin discovery and management
├── cache.py                 # Caching layer (Phase 1)
├── exceptions.py            # Custom exception hierarchy (Phase 1)
├── error_handler.py         # Error handling framework (Phase 1)
├── readme_generator.py      # README-specific generation
├── readme_documentation_generator.py  # Unified README pipeline
├── config/                  # Configuration management (Phase 1)
│   ├── __init__.py         # Configuration exports
│   ├── settings.py         # Pydantic-based settings
│   └── validators.py       # Custom validation logic
├── providers/              # LLM provider implementations
│   ├── __init__.py         # Provider interface exports
│   ├── base.py             # Abstract provider interface
│   ├── manager.py          # Provider management
│   ├── openai_provider.py  # OpenAI implementation
│   └── claude_provider.py  # Anthropic implementation
├── plugins/                # Plugin system
│   ├── __init__.py         # Plugin interface exports
│   ├── base.py             # RecommendationEngine base class
│   ├── analysis_base.py    # Analysis plugin base
│   └── modules.py          # Module recommender plugin
├── analysis/               # Analysis plugins
│   ├── __init__.py         # Analysis exports
│   ├── compiler.py         # Document compilation
│   ├── reporter.py         # Report generation
│   └── link_validator.py   # Link validation
└── evaluator/              # Evaluation components
    ├── __init__.py         # Evaluator exports
    ├── comparator.py       # Document comparison
    ├── metrics.py          # Similarity metrics
    └── downloader.py       # URL content fetching
```

## Key Directories

### prompts/
Contains YAML configuration files for different generation modes:
- `generator/` - Documentation generation prompts
- `analysis/` - Analysis and evaluation prompts

### tests/
Comprehensive test suite with fixtures:
- `fixtures/` - Test data and mock objects
- `providers/` - Provider-specific tests
- Individual test files for each module

### docs/
Documentation and design materials:
- `design/` - Architecture documentation and refactoring plans

## Entry Points
- **CLI Tool**: `doc-gen` command (defined in pyproject.toml)
- **Plugin Discovery**: Entry points for plugins and analysis tools
- **Package Import**: Main classes available via `from doc_generator import ...`

## Configuration Files
- **pyproject.toml**: Project metadata, dependencies, tool configurations
- **pytest.ini**: Testing framework configuration
- **mkdocs.yml**: Documentation site configuration
- **terminology.yaml**: HPC modules and cluster command definitions

## Architecture Notes
- **Modular Design**: Clear separation between core, CLI, plugins, and analysis
- **Plugin System**: Extensible via entry points defined in pyproject.toml
- **Provider Pattern**: Unified interface for multiple LLM providers
- **Recent Phase 1 Updates**: New config/, cache.py, exceptions.py, error_handler.py