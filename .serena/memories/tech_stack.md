# Technology Stack

## Core Language & Version
- **Python**: 3.8+ (supports 3.8, 3.9, 3.10, 3.11, 3.12)
- **Build System**: setuptools with pyproject.toml

## Main Dependencies
- **LLM Providers**:
  - `openai>=1.0.0` - OpenAI GPT integration
  - `anthropic>=0.25.0` - Anthropic Claude integration
- **Configuration & Data**:
  - `pyyaml>=6.0` - YAML configuration files
  - `python-dotenv>=0.19.0` - Environment variable management
  - `pydantic>=2.0.0` - Data validation and settings management
  - `pydantic-settings>=2.0.0` - Settings with environment variables
- **Web & Parsing**:
  - `beautifulsoup4>=4.11.0` - HTML parsing and manipulation
  - `requests>=2.28.0` - HTTP requests
- **Data Processing**:
  - `pandas>=1.5.0` - Data analysis and manipulation
  - `numpy>=1.21.0` - Numerical computing
  - `tabulate>=0.9.0` - Table formatting

## Development Dependencies
- **Testing**:
  - `pytest>=7.0.0` - Testing framework
  - `pytest-cov>=4.0.0` - Coverage reporting
  - `pytest-mock>=3.10.0` - Mocking utilities
- **Code Quality**:
  - `black>=22.0.0` - Code formatting (line-length: 100)
  - `flake8>=5.0.0` - Linting
  - `mypy>=1.0.0` - Type checking (strict mode enabled)
- **Documentation**:
  - `mkdocs>=1.5.0` - Documentation generator
  - `mkdocs-material>=9.0.0` - Material theme
  - `mkdocstrings[python]>=0.24.0` - API documentation

## Architecture Patterns
- **Plugin System**: Entry points-based extensible architecture
- **Provider Pattern**: Multiple LLM provider support with unified interface
- **Configuration Management**: Pydantic-based settings with validation
- **Error Handling**: Structured exceptions with context and retry logic
- **Caching**: Intelligent caching with TTL and LRU eviction

## Package Structure
- **Entry Points**: CLI tools and plugin discovery
- **Modular Design**: Separate modules for core, CLI, plugins, analysis
- **Type Safety**: Comprehensive type hints and mypy validation