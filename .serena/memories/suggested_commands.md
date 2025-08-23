# Essential Development Commands

## Installation and Setup
```bash
# Install package in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with test dependencies
pip install -e ".[test]"

# Install with documentation dependencies
pip install -e ".[docs]"
```

## Testing Commands
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=doc_generator tests/

# Run with coverage and HTML report
pytest --cov=doc_generator --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Exclude slow tests

# Run specific test file
pytest tests/test_plugin_manager.py -v
```

## Code Quality Commands
```bash
# Format code with black
black src/ tests/

# Check formatting without making changes
black --check src/ tests/

# Run linting with flake8
flake8 src/ tests/

# Run type checking with mypy
mypy src/doc_generator

# Run all quality checks together
black src/ tests/ && flake8 src/ tests/ && mypy src/doc_generator
```

## Application Commands
```bash
# Basic documentation generation
doc-gen --topic "Python Programming" --runs 3 --analyze

# README generation for directories
doc-gen --readme /path/to/directory --recursive --output-dir ./output

# List available models and providers
doc-gen --list-models

# List and manage plugins
doc-gen --list-plugins
doc-gen --disable-plugins module_recommender

# Clean up output directory
doc-gen --cleanup

# Display detailed help information
doc-gen --info
```

## Documentation Commands
```bash
# Serve documentation locally (http://127.0.0.1:8000)
mkdocs serve

# Build static documentation site
mkdocs build

# Deploy to GitHub Pages (requires permissions)
mkdocs gh-deploy
```

## Git and Development Workflow
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Check git status
git status

# Add and commit changes
git add -A
git commit -m "feat: description of changes"

# Push branch
git push -u origin feature/your-feature-name
```

## System Utilities (Linux)
```bash
# File operations
ls -la                    # List files with details
find . -name "*.py"       # Find Python files
grep -r "pattern" src/    # Search for pattern in source

# Process management
ps aux | grep python      # Find running Python processes
htop                      # System monitor

# Package management
pip list                  # List installed packages
pip show package-name     # Show package details
```

## Debugging and Development
```bash
# Python REPL with project context
python -c "from src.doc_generator.core import DocumentationGenerator; print('Ready')"

# Check Python path and imports
python -c "import sys; print(sys.path)"

# Test specific functionality
python -c "from src.doc_generator.config import get_settings; print(get_settings().app_name)"
```