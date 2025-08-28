# Contributing Guide

Welcome to the doc-generator project! This guide will help you contribute effectively, whether you're fixing bugs, adding features, or creating new plugins.

## Ways to Contribute

### Code Contributions
- **Bug fixes** - Fix issues and improve stability
- **New features** - Add functionality to core system
- **Plugin development** - Create new recommendation engines
- **Test improvements** - Expand test coverage and quality
- **Documentation** - Improve guides and API docs

### Non-Code Contributions  
- **Issue reporting** - Report bugs and suggest features
- **Documentation** - Fix typos, clarify instructions
- **Community support** - Help other users in discussions
- **Examples** - Contribute usage examples and tutorials

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/doc-fun.git
cd doc-fun

# Add upstream remote
git remote add upstream https://github.com/fasrc/doc-fun.git
```

### 2. Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks (if available)
pre-commit install

# Verify setup
python -m pytest -v
doc-gen --version
```

### 3. Create Feature Branch

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/amazing-new-feature
# or
git checkout -b fix/important-bug-fix
```

## Development Workflow

### Code Standards

#### Python Style
- **PEP 8** compliance with Black formatting
- **Type hints** for all function signatures
- **Docstrings** in Google style for all public functions
- **Maximum line length** of 88 characters (Black default)

```python
from typing import List, Dict, Optional

def generate_documentation(
    topic: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    context: Optional[Dict] = None
) -> List[Dict]:
    """Generate documentation for the given topic.
    
    Args:
        topic: The topic to document
        model: OpenAI model to use
        temperature: Generation temperature (0.0-1.0)
        context: Optional context dictionary
        
    Returns:
        List of generated documentation variants
        
    Raises:
        ValueError: If topic is empty or invalid
        OpenAIError: If API request fails
    """
    if not topic.strip():
        raise ValueError("Topic cannot be empty")
    
    # Implementation here...
    return results
```

#### Code Organization
- **Single responsibility** principle for classes and functions
- **Clear separation** between core logic and plugin system
- **Dependency injection** for testability
- **Error handling** with specific exception types

### Testing Requirements

#### Test Coverage
- **New features** must have 90%+ test coverage
- **Bug fixes** must include regression tests
- **Public API** changes require integration tests
- **Plugin interfaces** need comprehensive test suites

#### Test Categories

```bash
# Unit tests - Fast, isolated tests
python -m pytest tests/unit/ -v

# Integration tests - Plugin and system integration
python -m pytest tests/integration/ -v

# All tests with coverage
python -m pytest --cov=src/doc_generator --cov-report=html
```

#### Writing Tests

```python
# Test file: tests/test_new_feature.py
import pytest
from unittest.mock import Mock, patch
from doc_generator.new_feature import NewFeatureClass

class TestNewFeature:
    """Test suite for NewFeature functionality."""
    
    def test_basic_functionality(self):
        """Test basic NewFeature operation."""
        feature = NewFeatureClass()
        result = feature.process_input("test input")
        
        assert result is not None
        assert isinstance(result, dict)
        assert "status" in result
    
    @patch('doc_generator.new_feature.external_api_call')
    def test_external_dependency(self, mock_api):
        """Test NewFeature with mocked external dependencies."""
        mock_api.return_value = {"data": "test"}
        
        feature = NewFeatureClass()
        result = feature.process_with_api("input")
        
        mock_api.assert_called_once_with("input")
        assert result["data"] == "test"
    
    def test_error_handling(self):
        """Test NewFeature error handling."""
        feature = NewFeatureClass()
        
        with pytest.raises(ValueError, match="Invalid input"):
            feature.process_input("")
```

### Code Review Process

#### Pre-submission Checklist
- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] New features have documentation
- [ ] Changes are covered by tests
- [ ] No security vulnerabilities introduced
- [ ] Performance impact considered

#### Formatting and Linting

```bash
# Format code with Black
python -m black src/ tests/

# Check formatting
python -m black --check src/ tests/

# Lint with flake8
python -m flake8 src/ tests/

# Type checking with mypy (if configured)
python -m mypy src/doc_generator/
```

## Plugin Development

### Creating New Plugins

#### 1. Plugin Interface

```python
# src/doc_generator/plugins/my_plugin.py
from typing import List, Dict, Optional
from doc_generator.plugins.base import RecommendationEngine

class MyRecommender(RecommendationEngine):
    """Recommends relevant resources for documentation topics."""
    
    def __init__(self, terminology: Optional[Dict] = None, **kwargs):
        super().__init__(terminology, **kwargs)
        # Plugin-specific initialization
    
    def get_name(self) -> str:
        """Return unique plugin identifier."""
        return "my_plugin"
    
    def get_supported_types(self) -> List[str]:
        """Return types of recommendations this plugin provides."""
        return ["resources", "tools", "references"]
    
    def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
        """Generate recommendations for the given topic."""
        # Implementation here
        return [
            {
                "title": "Relevant Resource",
                "description": "Helpful resource for " + topic,
                "url": "https://example.com/resource",
                "relevance_score": 8.5,
                "type": "resource"
            }
        ]
    
    def is_enabled(self) -> bool:
        """Check if plugin should be active."""
        return True  # Or check environment/config
    
    def get_priority(self) -> int:
        """Get plugin execution priority (higher = earlier)."""
        return 50  # Default priority
```

#### 2. Plugin Testing

```python
# tests/test_my_plugin.py
import pytest
from doc_generator.plugins.my_plugin import MyRecommender

class TestMyRecommender:
    """Test suite for MyRecommender plugin."""
    
    def test_plugin_interface(self):
        """Test plugin implements required interface."""
        plugin = MyRecommender()
        
        assert plugin.get_name() == "my_plugin"
        assert "resources" in plugin.get_supported_types()
        assert plugin.is_enabled() is True
        assert isinstance(plugin.get_priority(), int)
    
    def test_get_recommendations(self):
        """Test recommendation generation."""
        plugin = MyRecommender()
        results = plugin.get_recommendations("Python Programming")
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for rec in results:
            assert "title" in rec
            assert "description" in rec
            assert "relevance_score" in rec
            assert isinstance(rec["relevance_score"], (int, float))
    
    def test_empty_topic_handling(self):
        """Test handling of empty/invalid topics."""
        plugin = MyRecommender()
        results = plugin.get_recommendations("")
        
        # Should return empty list, not crash
        assert isinstance(results, list)
```

#### 3. Plugin Registration

```toml
# For external plugin packages - pyproject.toml
[project.entry-points."doc_generator.plugins"]
my_plugin = "my_package.plugins:MyRecommender"
```

### Built-in Plugin Guidelines

- **Follow existing patterns** in `src/doc_generator/plugins/modules.py`
- **Handle errors gracefully** - never crash the main generation
- **Log appropriately** using `self.logger`
- **Respect rate limits** for external APIs
- **Cache results** when appropriate
- **Document configuration options** clearly

## Issue Guidelines

### Reporting Bugs

Use the bug report template:

```markdown
**Bug Description**
A clear description of the bug.

**Steps to Reproduce**
1. Run command: `doc-gen --topic "Test" --model gpt-4`
2. Observe error in output
3. Check logs for details

**Expected Behavior**
Documentation should generate successfully.

**Actual Behavior**
Process crashes with KeyError.

**Environment**
- OS: macOS 13.2
- Python: 3.10.8
- doc-generator: 1.1.0
- OpenAI: 1.35.3

**Additional Context**
Error occurs only with gpt-4 model, works fine with gpt-3.5-turbo.

**Logs**
```
Traceback (most recent call last):
  File "src/doc_generator/core.py", line 123, in generate
KeyError: 'response_format'
```

### Feature Requests

Use the feature request template:

```markdown
**Feature Description**
Add support for generating LaTeX documentation format.

**Use Case**
Academic users need LaTeX output for integration with research papers and thesis documents.

**Proposed Solution**
1. Create LaTeX prompt template in `prompts/generator/latex.yaml`
2. Add LaTeX formatting options to CLI
3. Include LaTeX-specific section structure

**Alternative Solutions**
- Markdown output with Pandoc conversion
- HTML with LaTeX CSS styling

**Additional Context**
This would complement existing HTML and Markdown formats.
```

## Pull Request Process

### 1. Prepare Your Changes

```bash
# Ensure tests pass
python -m pytest -v

# Format code
python -m black src/ tests/

# Check for issues
python -m flake8 src/ tests/

# Update documentation if needed
# Edit relevant files in docs/
```

### 2. Commit Guidelines

Use conventional commit format:

```bash
# Feature commits
git commit -m "feat: add LaTeX output format support"
git commit -m "feat(plugins): add DatasetRecommender plugin (TBD - conceptual example)"

# Bug fix commits  
git commit -m "fix: handle empty topic input gracefully"
git commit -m "fix(cli): correct argument parsing for --disable-plugins"

# Documentation commits
git commit -m "docs: add LaTeX format usage examples"
git commit -m "docs(api): update plugin interface documentation"

# Test commits
git commit -m "test: add integration tests for new plugin system"
```

### 3. Pull Request Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] New tests added for changes
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Code is commented where needed
- [ ] Documentation updated where needed
- [ ] Changes generate no new warnings

## Screenshots (if applicable)
Add screenshots to help explain your changes.

## Additional Notes
Any additional information about the changes.
```

### 4. Review Process

#### Reviewer Guidelines
- **Check functionality** - Does the code work as intended?
- **Review tests** - Are changes adequately tested?
- **Style compliance** - Does code follow project standards?
- **Documentation** - Are changes documented appropriately?
- **Performance** - Are there any performance implications?
- **Security** - Are there any security concerns?

#### Addressing Feedback
```bash
# Make requested changes
git add .
git commit -m "fix: address review feedback for error handling"

# Update pull request (no need to create new PR)
git push origin feature/amazing-new-feature
```

## Recognition

### Contributors
- All contributors are recognized in the project README
- Significant contributions may be highlighted in release notes
- Plugin authors are credited in plugin documentation

### Community
- Help other contributors in GitHub discussions
- Share your plugins and extensions with the community
- Contribute to documentation and examples

## Resources

### Development Resources
- **[Python Style Guide](https://pep8.org/)** - Official Python style guidelines
- **[pytest Documentation](https://docs.pytest.org/)** - Testing framework docs
- **[Black Formatter](https://black.readthedocs.io/)** - Code formatting tool
- **[Type Hints](https://docs.python.org/3/library/typing.html)** - Python typing system

### Project Resources
- **[Plugin Development Guide](creating-plugins.md)** - Comprehensive plugin creation guide
- **[Testing Guide](testing.md)** - Understanding the test suite
- **[API Documentation](../api/index.md)** - Auto-generated API docs
- **[Architecture Overview](../index.md#architecture-overview)** - System design patterns

## ❓ Getting Help

### Questions and Support
- **GitHub Discussions** - Ask questions and share ideas
- **GitHub Issues** - Report bugs and request features
- **FASRC Support** - Contact the original development team

### Best Practices
- **Search existing issues** before creating new ones
- **Use descriptive titles** for issues and PRs
- **Provide minimal reproducible examples** for bugs
- **Be respectful** and constructive in all interactions

---

**Ready to contribute?** Start by exploring the codebase, running the tests, and looking for issues labeled `good first issue` or `help wanted`.

Thank you for contributing to doc-generator!