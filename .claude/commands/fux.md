---
name: fux
description: Run all tests and automatically fix any errors found
---

# Fix Tests Command

Automatically run all tests, identify failures, and fix errors to ensure a passing test suite.

## Overview

The `fux` command provides an intelligent test runner that:
- Executes the complete test suite
- Analyzes any failures or errors
- Automatically fixes identified issues
- Re-runs tests to verify fixes
- Reports on changes made

## Process

### 1. Initial Test Run
- Identify test framework (pytest, unittest, jest, etc.)
- Run full test suite with verbose output
- Capture all errors, failures, and warnings

### 2. Error Analysis
- Parse test output for failure patterns
- Categorize errors:
  - Import errors
  - Type errors
  - Assertion failures
  - Missing dependencies
  - Syntax errors
  - Configuration issues

### 3. Automated Fixes
Apply appropriate fixes based on error type:

#### Import Errors
- Add missing imports
- Fix import paths
- Install missing packages

#### Type Errors
- Add type annotations
- Fix type mismatches
- Update function signatures

#### Assertion Failures
- Analyze expected vs actual values
- Update test expectations if appropriate
- Fix implementation bugs

#### Missing Dependencies
- Install required packages
- Update requirements files
- Configure test environment

#### Syntax Errors
- Fix syntax issues
- Update deprecated syntax
- Resolve formatting problems

### 4. Validation
- Re-run affected tests
- Verify fixes resolved issues
- Check for new failures introduced
- Run linting and type checking

### 5. Reporting
Generate summary of:
- Tests fixed
- Changes made
- Remaining issues (if any)
- Recommendations for manual review

## Test Frameworks Support

### Python (pytest)
```bash
# Run tests with coverage
pytest -v --tb=short

# Run specific test file
pytest tests/test_specific.py

# Run with markers
pytest -m "not slow"
```

### Python (unittest)
```bash
python -m unittest discover
```

### JavaScript/TypeScript
```bash
npm test
npm run test:coverage
jest --watch
```

### Other Frameworks
- Go: `go test ./...`
- Rust: `cargo test`
- Ruby: `rspec`
- Java: `mvn test` or `gradle test`

## Fix Strategies

### Common Patterns

#### Missing Module
```python
# Error: ModuleNotFoundError: No module named 'requests'
# Fix: pip install requests
```

#### Import Path Issues
```python
# Error: ImportError: cannot import name 'Component'
# Fix: Update import path or add __init__.py
```

#### Type Mismatches
```python
# Error: TypeError: expected str, got int
# Fix: Add type conversion or update type hints
```

#### Assertion Updates
```python
# Error: AssertionError: expected 5, got 6
# Fix: Analyze if test or implementation needs update
```

## Configuration

### Test Configuration Files
- `pytest.ini` / `pyproject.toml` - pytest configuration
- `jest.config.js` - Jest configuration
- `.coveragerc` - Coverage settings
- `tox.ini` - Tox environments

### Environment Variables
```bash
# Set test environment
export TESTING=true
export TEST_DATABASE_URL=sqlite:///:memory:
```

## Safety Measures

### Protected Operations
- Never modify test data fixtures without analysis
- Preserve test isolation
- Maintain backwards compatibility
- Keep test coverage above threshold

### Review Requirements
Some fixes require manual review:
- Changes to core test logic
- Modifications to fixtures
- Updates to mocked behavior
- Changes affecting multiple test files

## Example Usage

```
User: fux

Assistant will:
1. Detect pytest as test framework
2. Run: pytest -v
3. Find 3 failures:
   - ImportError in test_utils.py
   - TypeError in test_core.py
   - AssertionError in test_api.py
4. Fix issues:
   - Add missing import
   - Update type conversion
   - Correct assertion value
5. Re-run tests: All pass ✓
6. Report: "Fixed 3 test failures. All tests now passing."
```

## Advanced Options

### Selective Fixing
- Fix only specific test files
- Target particular error types
- Skip certain test categories

### Integration with CI/CD
- Pre-commit hook integration
- GitHub Actions compatibility
- Pipeline failure recovery

### Performance Optimization
- Parallel test execution
- Test result caching
- Incremental test runs

## Notes
- Always backup code before major fixes
- Some fixes may require domain knowledge
- Complex failures might need manual intervention
- Consider test flakiness vs actual failures
- Maintain test quality while fixing