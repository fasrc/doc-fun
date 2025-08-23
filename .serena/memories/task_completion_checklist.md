# Task Completion Checklist

## Before Committing Code

### 1. Code Quality Checks
```bash
# Format code with black
black src/ tests/

# Run linting
flake8 src/ tests/

# Run type checking
mypy src/doc_generator
```

### 2. Testing Requirements
```bash
# Run full test suite
pytest -v

# Run with coverage (should maintain high coverage)
pytest --cov=doc_generator tests/

# Test specific changes
pytest tests/test_[relevant_module].py -v
```

### 3. Functional Testing
```bash
# Test CLI functionality
doc-gen --help
doc-gen --list-models
doc-gen --list-plugins

# Test core functionality (if applicable)
python -c "from src.doc_generator.core import DocumentationGenerator; gen = DocumentationGenerator(); print('✅ Core working')"
```

## Git Workflow

### 1. Branch Management
```bash
# Create feature branch from main
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

### 2. Commit Standards
```bash
# Stage changes
git add -A

# Commit with conventional format
git commit -m "feat: description of changes

Details about the implementation.

🤖 Generated with Claude Code"

# Push branch
git push -u origin feature/your-feature-name
```

### 3. Pull Request Preparation
- Ensure all tests pass
- Update documentation if needed
- Check that CLI still works as expected
- Verify backward compatibility

## Documentation Updates

### When to Update Documentation
- New features or CLI options
- API changes or new classes
- Configuration changes
- New dependencies

### Documentation Locations
- **README.md**: For user-facing changes
- **CLAUDE.md**: For development guidance
- **docs/**: For detailed documentation
- **Docstrings**: For code-level documentation

## Quality Standards

### Code Requirements
- ✅ All functions have type hints
- ✅ All public methods have docstrings
- ✅ Code formatted with black (100 chars)
- ✅ No flake8 violations
- ✅ No mypy errors
- ✅ Tests pass with good coverage

### Architecture Requirements
- ✅ Follow existing patterns
- ✅ Use configuration system for settings
- ✅ Apply error handling framework
- ✅ Add caching where appropriate
- ✅ Maintain plugin compatibility

## Post-Commit Actions

### 1. Monitor CI/CD
- Check that automated tests pass
- Verify documentation builds correctly
- Ensure no breaking changes

### 2. Communication
- Update relevant team members
- Document any breaking changes
- Update version numbers if needed

## Emergency Rollback
```bash
# If issues are found after merge
git revert [commit-hash]
git push origin main
```

## Notes
- Phase 1 foundation components (config, error handling, caching) should be used in new code
- Maintain backward compatibility for existing CLI interface
- Follow the established plugin architecture patterns
- Consider impact on both OpenAI and Claude providers