# Testing Guide

This comprehensive guide covers how to run tests, understand test results, write new tests, and maintain test quality for doc-generator.

## 🧪 Running Tests

### Quick Test Commands

```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run with short traceback format
python -m pytest -v --tb=short

# Run tests and show coverage
python -m pytest --cov=src/doc_generator --cov-report=html
```

### Test Categories

The test suite is organized into several categories:

```bash
# Core functionality tests
python -m pytest tests/test_doc_generator.py -v

# Plugin system tests
python -m pytest tests/test_plugin_manager.py -v
python -m pytest tests/test_plugin_interface.py -v

# Integration tests
python -m pytest tests/test_plugin_integration.py -v

# Run specific test class
python -m pytest tests/test_plugin_manager.py::TestPluginManager -v

# Run specific test method
python -m pytest tests/test_plugin_interface.py::TestRecommendationEngineInterface::test_recommendation_engine_is_abstract -v
```

### Test Results Understanding

**Successful Test Run:**
```
============================= test session starts ==============================
platform darwin -- Python 3.10.14, pytest-8.4.1, pluggy-1.6.0
collecting ... collected 57 items

tests/test_doc_generator.py::TestDocumentationGenerator::test_init_with_defaults PASSED [  1%]
tests/test_plugin_manager.py::TestPluginManager::test_initialization PASSED [100%]

========================= 57 passed in 0.41s =========================
```

**Test Failure Example:**
```
FAILED tests/test_plugin_manager.py::TestPluginManager::test_plugin_discovery_success
_________________________ TestPluginManager.test_plugin_discovery_success _________________________

    def test_plugin_discovery_success(self, sample_terminology):
        mock_entry_point = Mock()
        mock_entry_point.name = "datasets"
>       mock_entry_point.load.return_value = MockDatasetRecommender
E       NameError: name 'MockDatasetRecommender' is not defined

tests/test_plugin_manager.py:156: NameError
```

## 📊 Test Coverage

### Generate Coverage Reports

```bash
# HTML coverage report
python -m pytest --cov=src/doc_generator --cov-report=html

# View in browser
open htmlcov/index.html

# Terminal coverage report
python -m pytest --cov=src/doc_generator --cov-report=term-missing

# Coverage with specific threshold
python -m pytest --cov=src/doc_generator --cov-fail-under=90
```

### Current Test Coverage

**Test Statistics (as of v1.1.0):**
- **Total Tests:** 57
- **Test Files:** 4
- **Coverage:** ~95% of core functionality
- **Plugin Tests:** 29 tests covering plugin architecture
- **Integration Tests:** 6 tests covering system integration

**Coverage Breakdown:**
```
Name                                 Stmts   Miss  Cover   Missing
----------------------------------------------------------------
src/doc_generator/__init__.py           8      0   100%
src/doc_generator/core.py             235     12    95%   45-47, 234-236
src/doc_generator/plugin_manager.py    89      3    97%   67-69
src/doc_generator/plugins/base.py      67      2    97%   89-91
src/doc_generator/plugins/modules.py  156      8    95%   234-241
src/doc_generator/cli.py              198     45    77%   167-189, 234-267
----------------------------------------------------------------
TOTAL                                 753     70    91%
```

## 🏗️ Test Architecture

### Test Structure

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures and configuration
├── fixtures/
│   └── mock_plugins.py        # Mock plugin implementations
├── test_doc_generator.py      # Core functionality tests
├── test_plugin_manager.py     # Plugin management tests
├── test_plugin_interface.py   # Plugin interface tests
└── test_plugin_integration.py # Full system integration tests
```

### Test Categories Explained

#### 1. Core Functionality Tests (`test_doc_generator.py`)
```python
class TestDocumentationGenerator:
    """Tests for main DocumentationGenerator class"""
    
class TestDocumentAnalyzer:
    """Tests for HTML analysis functionality"""
    
class TestGPTQualityEvaluator:
    """Tests for GPT-based quality assessment"""
    
class TestCodeExampleScanner:
    """Tests for code scanning and analysis"""
    
class TestModuleRecommender:
    """Tests for HPC module recommendation plugin"""
```

#### 2. Plugin System Tests (`test_plugin_manager.py`)
```python
class TestPluginManager:
    """Tests for plugin discovery, loading, and management"""
    
    def test_plugin_discovery_success(self):
        """Test successful plugin discovery via entry points"""
        
    def test_plugin_loading_failure(self):
        """Test graceful handling of plugin load failures"""
```

#### 3. Plugin Interface Tests (`test_plugin_interface.py`)
```python
class TestRecommendationEngineInterface:
    """Tests for plugin base class and interface requirements"""
    
    def test_recommendation_engine_is_abstract(self):
        """Test that base class cannot be instantiated"""
        
    def test_valid_plugin_implementation(self):
        """Test that proper plugins work correctly"""
```

#### 4. Integration Tests (`test_plugin_integration.py`)
```python
class TestPluginIntegration:
    """Tests for full system integration with plugins"""
    
    def test_multiple_plugins_in_context(self):
        """Test multiple plugins working together"""
        
    def test_plugin_failure_doesnt_break_generation(self):
        """Test system resilience to plugin failures"""
```

### Fixtures and Mocks

#### Key Fixtures (`conftest.py`)

```python
@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    
@pytest.fixture
def sample_terminology():
    """Sample HPC modules and terminology for testing"""
    
@pytest.fixture
def mock_plugin_discovery():
    """Mock plugin discovery process"""
    
@pytest.fixture
def mock_openai_client():
    """Mock OpenAI API client"""
```

#### Mock Plugins (`fixtures/mock_plugins.py`)

```python
class MockRecommendationEngine(RecommendationEngine):
    """Base mock plugin for testing"""
    
class MockDatasetRecommender(MockRecommendationEngine):
    """Mock dataset recommender plugin"""
    
class FailingPlugin(MockRecommendationEngine):
    """Plugin that fails for error testing"""
```

## ✍️ Writing New Tests

### Test Writing Guidelines

#### 1. Follow Naming Conventions

```python
# Good test names
def test_plugin_loads_successfully_with_valid_config():
def test_recommendation_engine_raises_error_with_invalid_input():
def test_module_recommender_returns_empty_list_for_unknown_topic():

# Poor test names  
def test_plugin():
def test_error():
def test_modules():
```

#### 2. Use Descriptive Docstrings

```python
def test_plugin_discovery_with_multiple_plugins(self, sample_terminology, mock_plugin_discovery):
    """
    Test that multiple plugins can be discovered and loaded simultaneously.
    
    This test verifies:
    - Multiple entry points are processed correctly
    - Each plugin gets its own instance
    - Plugin manager maintains separate references
    - No interference between plugin loading
    """
```

#### 3. Structure Tests with AAA Pattern

```python
def test_module_recommender_scores_python_modules_higher_for_python_topics(self, sample_terminology):
    """Test that Python modules get higher relevance scores for Python topics."""
    
    # Arrange
    recommender = ModuleRecommender(terminology=sample_terminology)
    python_topic = "Python Data Analysis with Pandas"
    
    # Act
    recommendations = recommender.get_recommendations(python_topic)
    
    # Assert
    assert len(recommendations) > 0
    top_recommendation = recommendations[0]
    assert 'python' in top_recommendation['name'].lower()
    assert top_recommendation['relevance_score'] > 5.0
```

### Adding Tests for New Features

#### Example: Adding Tests for a New Plugin

**1. Create Mock Plugin**

```python
# In tests/fixtures/mock_plugins.py
class MockWorkflowRecommender(MockRecommendationEngine):
    """Mock workflow recommender for testing"""
    
    def __init__(self, **kwargs):
        super().__init__(name='workflows', **kwargs)
        self._recommendations = [
            {
                "name": "SLURM Array Job Template",
                "type": "job_script",
                "relevance_score": 8.5
            }
        ]
    
    def get_supported_types(self):
        return ["workflows", "job_scripts", "templates"]
```

**2. Add Plugin Manager Tests**

```python
# In tests/test_plugin_manager.py
def test_workflow_plugin_discovery(self, sample_terminology, mock_plugin_discovery):
    """Test discovery of workflow recommendation plugin."""
    plugins = {"workflows": MockWorkflowRecommender}
    
    with mock_plugin_discovery(plugins):
        manager = PluginManager(terminology=sample_terminology)
        manager.load_plugins()
        
        assert "workflows" in manager.engines
        assert isinstance(manager.engines["workflows"], MockWorkflowRecommender)
```

**3. Add Integration Tests**

```python
# In tests/test_plugin_integration.py
def test_workflow_recommendations_in_context(self, temp_dir, sample_terminology, mock_plugin_discovery):
    """Test that workflow recommendations appear in documentation context."""
    plugins = {"workflows": MockWorkflowRecommender}
    
    with mock_plugin_discovery(plugins):
        # ... setup generator
        context = generator._build_terminology_context("Parallel Processing")
        
        assert "SLURM Array Job Template" in context
        assert "job_script" in context
```

### Testing Error Conditions

#### Test Plugin Failures

```python
def test_plugin_with_invalid_recommendations_handled_gracefully(self, sample_terminology, mock_plugin_discovery):
    """Test that plugins returning invalid data don't crash the system."""
    
    class BadPlugin(RecommendationEngine):
        def get_name(self):
            return "bad"
        
        def get_recommendations(self, topic, context=None):
            return "invalid_return_type"  # Should return list
    
    plugins = {"bad": BadPlugin}
    
    with mock_plugin_discovery(plugins):
        manager = PluginManager(terminology=sample_terminology)
        manager.load_plugins()
        
        # Should not crash, should handle gracefully
        results = manager.get_recommendations("test topic")
        assert isinstance(results, dict)
```

#### Test Configuration Errors

```python
def test_documentation_generator_handles_missing_config_files(self, temp_dir):
    """Test graceful handling of missing configuration files."""
    
    # Don't create config files - test with missing files
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        # Should not crash, should use defaults
        generator = DocumentationGenerator(
            prompt_yaml_path="nonexistent.yaml",
            terminology_path="missing.yaml"
        )
        
        assert generator.prompt_config is not None
        assert generator.terminology is not None
```

## 🔧 Test Utilities and Helpers

### Custom Assertions

```python
# In tests/test_helpers.py (create if needed)
def assert_valid_html(html_content):
    """Assert that content is valid HTML."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    assert soup.find('title') is not None
    assert soup.find('body') is not None

def assert_contains_module_recommendations(content):
    """Assert that content contains HPC module recommendations."""
    assert "module load" in content
    assert any(module in content for module in ["python/", "gcc/", "cuda/"])

def assert_plugin_interface_compliance(plugin_class):
    """Assert that a plugin class properly implements the interface."""
    from doc_generator.plugins.base import RecommendationEngine
    assert issubclass(plugin_class, RecommendationEngine)
    
    # Test instantiation
    plugin = plugin_class()
    assert hasattr(plugin, 'get_name')
    assert hasattr(plugin, 'get_recommendations')
    assert callable(plugin.get_name)
    assert callable(plugin.get_recommendations)
```

### Test Data Factories

```python
# In tests/factories.py (create if needed)
def create_sample_hpc_module(name="python/3.12.8-fasrc01", category="programming"):
    """Factory for creating test HPC module data."""
    return {
        'name': name,
        'description': f'Test module {name}',
        'category': category
    }

def create_sample_terminology(num_modules=5):
    """Factory for creating test terminology data."""
    return {
        'hpc_modules': [
            create_sample_hpc_module(f"test-module-{i}/1.0.0") 
            for i in range(num_modules)
        ],
        'cluster_commands': [
            {'name': 'sbatch', 'description': 'Submit batch job'}
        ]
    }
```

## 📈 Performance Testing

### Timing Tests

```python
import time

def test_plugin_loading_performance(self, sample_terminology, mock_plugin_discovery):
    """Test that plugin loading completes within reasonable time."""
    plugins = {f"plugin_{i}": MockRecommendationEngine for i in range(10)}
    
    with mock_plugin_discovery(plugins):
        start_time = time.time()
        
        manager = PluginManager(terminology=sample_terminology)
        manager.load_plugins()
        
        load_time = time.time() - start_time
        
        assert load_time < 1.0  # Should load 10 plugins in under 1 second
        assert len(manager.engines) == 10
```

### Memory Usage Tests

```python
import psutil
import os

def test_plugin_memory_usage_reasonable(self, sample_terminology, mock_plugin_discovery):
    """Test that plugin loading doesn't consume excessive memory."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    plugins = {f"plugin_{i}": MockRecommendationEngine for i in range(100)}
    
    with mock_plugin_discovery(plugins):
        manager = PluginManager(terminology=sample_terminology)
        manager.load_plugins()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Should not increase memory by more than 50MB for 100 plugins
        assert memory_increase < 50 * 1024 * 1024
```

## 🚨 Debugging Test Failures

### Common Test Failure Patterns

#### 1. Import Errors

```bash
# Error
ModuleNotFoundError: No module named 'doc_generator'

# Solution
cd /path/to/doc-fun
pip install -e .
python -m pytest
```

#### 2. Fixture Not Found

```bash
# Error
fixture 'sample_terminology' not found

# Check conftest.py exists and contains the fixture
cat tests/conftest.py | grep "def sample_terminology"
```

#### 3. Mock Errors

```bash
# Error
AttributeError: <module 'doc_generator'> does not have the attribute 'OpenAI'

# Fix patch path
# Wrong: patch('doc_generator.OpenAI')
# Right: patch('doc_generator.core.OpenAI')
```

### Debugging Techniques

#### 1. Isolate Failing Tests

```bash
# Run only failing test
python -m pytest tests/test_plugin_manager.py::TestPluginManager::test_failing_method -v -s

# Run with pdb debugger
python -m pytest tests/test_plugin_manager.py::TestPluginManager::test_failing_method --pdb
```

#### 2. Add Debug Output

```python
def test_debug_example(self, sample_terminology):
    """Test with debug output."""
    import pprint
    
    recommender = ModuleRecommender(terminology=sample_terminology)
    results = recommender.get_recommendations("Python")
    
    # Debug output
    print("\n=== DEBUG OUTPUT ===")
    print(f"Results type: {type(results)}")
    print(f"Results length: {len(results)}")
    pprint.pprint(results)
    print("=== END DEBUG ===\n")
    
    assert len(results) > 0
```

#### 3. Use Temporary Debug Tests

```python
def test_debug_plugin_loading(self, sample_terminology, mock_plugin_discovery):
    """Temporary test to debug plugin loading."""
    plugins = {"test": MockRecommendationEngine}
    
    with mock_plugin_discovery(plugins):
        manager = PluginManager(terminology=sample_terminology)
        
        print(f"Before loading: {len(manager.engines)}")
        manager.load_plugins()
        print(f"After loading: {len(manager.engines)}")
        print(f"Loaded plugins: {list(manager.engines.keys())}")
        
        assert "test" in manager.engines
```

## 🎯 Test Quality Guidelines

### Test Quality Checklist

- [ ] **Clear test names** describing what is being tested
- [ ] **Descriptive docstrings** explaining purpose and expectations
- [ ] **Single responsibility** - each test tests one thing
- [ ] **Proper setup/teardown** using fixtures
- [ ] **No external dependencies** - use mocks for APIs, file systems
- [ ] **Deterministic** - tests produce same results every time
- [ ] **Fast execution** - tests complete quickly
- [ ] **Error condition coverage** - test failure scenarios

### Code Coverage Goals

- **Core modules:** 95%+ coverage
- **Plugin system:** 90%+ coverage  
- **CLI interface:** 80%+ coverage
- **Integration tests:** Cover major workflows
- **Error handling:** Test all exception paths

## 🔄 Continuous Integration

### GitHub Actions Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev,test]"
    
    - name: Run tests
      run: |
        python -m pytest -v --cov=src/doc_generator --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8

  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: python -m pytest
        language: python
        pass_filenames: false
        always_run: true
```

## ✅ Next Steps

After mastering testing:

1. 🔌 **[Create Plugins Guide](creating-plugins.md)** - Build your own plugins
2. 📚 **[Advanced Configuration](configuration.md)** - Customize the system
3. 🤝 **[Contributing Guide](contributing.md)** - Contribute to the project

## 🎉 Testing Mastery

You now understand:
- How to run and interpret tests
- Test architecture and organization  
- Writing effective new tests
- Debugging test failures
- Maintaining test quality
- Performance and integration testing

**Keep testing, keep improving!** 🧪✨