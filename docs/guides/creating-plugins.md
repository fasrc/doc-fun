# Creating Plugins Guide

This comprehensive guide shows you how to create powerful plugins for doc-generator, from simple recommendation engines to complex third-party packages.

## 🔌 Plugin Architecture Overview

### What are Plugins?

Plugins in doc-generator are **recommendation engines** that analyze a documentation topic and provide relevant suggestions. They automatically integrate into the documentation generation process to enhance the final output.

**Built-in Plugin Example:**
- **ModuleRecommender**: Suggests relevant HPC modules based on topic keywords
- Input: "Python Machine Learning"
- Output: `module load python/3.12.8-fasrc01`, `module load cuda/12.9.1-fasrc01`

### Plugin Capabilities

Plugins can recommend:
- **Software modules and libraries**
- **Datasets and data sources**
- **Code examples and templates**
- **Workflows and job scripts**
- **Documentation links and references**
- **Best practices and troubleshooting tips**

## 🏗️ Plugin Architecture

### Base Class: RecommendationEngine

All plugins inherit from the `RecommendationEngine` abstract base class:

```python
from doc_generator.plugins import RecommendationEngine
from typing import List, Dict, Optional

class MyPlugin(RecommendationEngine):
    def get_name(self) -> str:
        """Return unique plugin identifier"""
        return "my_plugin"
    
    def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
        """Return list of recommendations for the topic"""
        return [
            {
                "title": "My Recommendation",
                "description": "Helpful suggestion",
                "relevance_score": 8.5
            }
        ]
```

### Required Methods

| Method | Purpose | Return Type |
|--------|---------|-------------|
| `get_name()` | Unique plugin identifier | `str` |
| `get_recommendations()` | Main recommendation logic | `List[Dict]` |

### Optional Methods

| Method | Purpose | Default |
|--------|---------|---------|
| `get_supported_types()` | Types of recommendations | `[get_name()]` |
| `get_formatted_recommendations()` | Custom formatting | Auto-formatted |
| `is_enabled()` | Enable/disable plugin | `True` |
| `get_priority()` | Plugin ordering priority | `50` |

## 🚀 Creating Your First Plugin

### Step 1: Simple Dataset Recommender

Let's create a plugin that recommends research datasets:

```python
# my_plugin/dataset_recommender.py
from doc_generator.plugins import RecommendationEngine
from typing import List, Dict, Optional
import requests

class DatasetRecommender(RecommendationEngine):
    """Recommends research datasets based on topic keywords."""
    
    def __init__(self, terminology: Optional[Dict] = None, **kwargs):
        super().__init__(terminology, **kwargs)
        
        # Dataset sources to search
        self.dataset_sources = [
            {
                "name": "Zenodo",
                "api_url": "https://zenodo.org/api/records",
                "search_param": "q"
            },
            {
                "name": "DataHub", 
                "api_url": "https://datahub.io/api/search",
                "search_param": "query"
            }
        ]
    
    def get_name(self) -> str:
        return "datasets"
    
    def get_supported_types(self) -> List[str]:
        return ["datasets", "research_data", "data_repositories"]
    
    def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
        """Find relevant datasets for the topic."""
        keywords = self._extract_keywords(topic)
        datasets = []
        
        for source in self.dataset_sources:
            try:
                source_datasets = self._search_source(source, keywords)
                datasets.extend(source_datasets)
            except Exception as e:
                self.logger.warning(f"Error searching {source['name']}: {e}")
        
        # Sort by relevance and return top 5
        datasets.sort(key=lambda x: x['relevance_score'], reverse=True)
        return datasets[:5]
    
    def _extract_keywords(self, topic: str) -> List[str]:
        """Extract searchable keywords from topic."""
        import re
        words = re.findall(r'\w+', topic.lower())
        # Filter common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def _search_source(self, source: Dict, keywords: List[str]) -> List[Dict]:
        """Search a specific data source."""
        query = " ".join(keywords)
        
        response = requests.get(
            source["api_url"],
            params={source["search_param"]: query, "size": 10},
            timeout=10
        )
        response.raise_for_status()
        
        results = response.json()
        datasets = []
        
        # Parse response (format varies by source)
        if source["name"] == "Zenodo":
            for hit in results.get("hits", {}).get("hits", []):
                metadata = hit.get("metadata", {})
                dataset = {
                    "title": metadata.get("title", "Unknown Dataset"),
                    "description": metadata.get("description", "")[:200] + "...",
                    "url": hit.get("links", {}).get("html", ""),
                    "source": source["name"],
                    "relevance_score": self._calculate_relevance(metadata.get("title", ""), keywords)
                }
                datasets.append(dataset)
        
        return datasets
    
    def _calculate_relevance(self, title: str, keywords: List[str]) -> float:
        """Calculate relevance score based on keyword matches."""
        title_lower = title.lower()
        score = 0.0
        
        for keyword in keywords:
            if keyword in title_lower:
                score += 2.0  # Title match worth more
            
        # Bonus for multiple keyword matches
        if score >= 4.0:
            score *= 1.2
            
        return min(score, 10.0)  # Cap at 10
    
    def get_formatted_recommendations(self, topic: str, context: Optional[Dict] = None) -> str:
        """Format recommendations for documentation context."""
        datasets = self.get_recommendations(topic, context)
        
        if not datasets:
            return ""
        
        formatted = "\n## Relevant Research Datasets:\n\n"
        for dataset in datasets:
            formatted += f"**{dataset['title']}**\n"
            formatted += f"- Source: {dataset['source']}\n"
            formatted += f"- URL: {dataset['url']}\n"
            formatted += f"- Description: {dataset['description']}\n"
            formatted += f"- Relevance Score: {dataset['relevance_score']:.1f}/10\n\n"
        
        return formatted
```

### Step 2: Package Structure

Create a proper Python package for your plugin:

```
my-dataset-plugin/
├── pyproject.toml
├── README.md
├── src/
│   └── my_dataset_plugin/
│       ├── __init__.py
│       ├── dataset_recommender.py
│       └── config/
│           └── default_sources.yaml
├── tests/
│   ├── __init__.py
│   ├── test_dataset_recommender.py
│   └── fixtures/
│       └── sample_data.py
└── docs/
    └── usage.md
```

### Step 3: Package Configuration

**pyproject.toml:**
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-dataset-plugin"
version = "0.1.0"
description = "Dataset recommendation plugin for doc-generator"
authors = [{name = "Your Name", email = "your.email@example.com"}]
readme = "README.md"
license = {text = "MIT"}
keywords = ["documentation", "datasets", "research", "plugin"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Researchers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
requires-python = ">=3.8"
dependencies = [
    "doc-generator>=1.1.0",
    "requests>=2.28.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "flake8>=5.0.0",
]

# This is the key part - register your plugin!
[project.entry-points."doc_generator.plugins"]
datasets = "my_dataset_plugin.dataset_recommender:DatasetRecommender"

[project.urls]
Homepage = "https://github.com/yourusername/my-dataset-plugin"
Repository = "https://github.com/yourusername/my-dataset-plugin"
```

### Step 4: Installation and Testing

```bash
# Install your plugin in development mode
cd my-dataset-plugin
pip install -e .

# Test that it's discovered
doc-gen --list-plugins

# Expected output should include:
# Plugin: datasets
#   Class: DatasetRecommender
#   Module: my_dataset_plugin.dataset_recommender

# Test with documentation generation
doc-gen --topic "Climate Change Research" --output-dir test-output
```

## 🎯 Advanced Plugin Examples

### Example 1: Code Template Recommender

```python
class CodeTemplateRecommender(RecommendationEngine):
    """Recommends code templates and examples based on programming languages."""
    
    def __init__(self, terminology: Optional[Dict] = None, **kwargs):
        super().__init__(terminology, **kwargs)
        
        self.template_mappings = {
            'python': {
                'patterns': ['python', 'py', 'pandas', 'numpy', 'scipy'],
                'templates': [
                    {
                        'name': 'Python Data Analysis Template',
                        'description': 'Template for data analysis with pandas',
                        'url': 'https://github.com/templates/python-data-analysis',
                        'type': 'jupyter_notebook'
                    }
                ]
            },
            'r': {
                'patterns': ['r', 'rstudio', 'statistics', 'statistical'],
                'templates': [
                    {
                        'name': 'R Statistical Analysis Template',
                        'description': 'Template for statistical analysis in R',
                        'url': 'https://github.com/templates/r-stats',
                        'type': 'r_script'
                    }
                ]
            }
        }
    
    def get_name(self) -> str:
        return "code_templates"
    
    def get_supported_types(self) -> List[str]:
        return ["code_templates", "examples", "boilerplate"]
    
    def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
        topic_lower = topic.lower()
        recommendations = []
        
        for language, config in self.template_mappings.items():
            # Check if topic matches language patterns
            if any(pattern in topic_lower for pattern in config['patterns']):
                for template in config['templates']:
                    template_copy = template.copy()
                    template_copy['language'] = language
                    template_copy['relevance_score'] = self._calculate_template_relevance(topic, template)
                    recommendations.append(template_copy)
        
        return sorted(recommendations, key=lambda x: x['relevance_score'], reverse=True)
    
    def _calculate_template_relevance(self, topic: str, template: Dict) -> float:
        # Score based on keyword matches in template description
        topic_words = set(topic.lower().split())
        template_words = set(template['description'].lower().split())
        
        common_words = topic_words.intersection(template_words)
        return len(common_words) * 2.0
```

### Example 2: Workflow Recommender

```python
class WorkflowRecommender(RecommendationEngine):
    """Recommends SLURM job scripts and workflow templates."""
    
    def __init__(self, terminology: Optional[Dict] = None, **kwargs):
        super().__init__(terminology, **kwargs)
        
        self.workflow_templates = {
            'parallel': [
                {
                    'name': 'MPI Job Array Template',
                    'description': 'Template for MPI-based parallel jobs with job arrays',
                    'script_type': 'slurm',
                    'use_case': 'parallel_computing'
                }
            ],
            'gpu': [
                {
                    'name': 'GPU Training Job Template',
                    'description': 'Template for GPU-accelerated machine learning training',
                    'script_type': 'slurm_gpu',
                    'use_case': 'machine_learning'
                }
            ]
        }
    
    def get_name(self) -> str:
        return "workflows"
    
    def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
        topic_lower = topic.lower()
        recommendations = []
        
        # Detect workflow type from topic
        if any(word in topic_lower for word in ['parallel', 'mpi', 'distributed']):
            recommendations.extend(self._score_templates(self.workflow_templates['parallel'], topic))
        
        if any(word in topic_lower for word in ['gpu', 'cuda', 'machine learning', 'deep learning']):
            recommendations.extend(self._score_templates(self.workflow_templates['gpu'], topic))
        
        return recommendations
    
    def _score_templates(self, templates: List[Dict], topic: str) -> List[Dict]:
        scored_templates = []
        for template in templates:
            template_copy = template.copy()
            template_copy['relevance_score'] = 7.0  # Base score for matching category
            scored_templates.append(template_copy)
        return scored_templates
    
    def get_formatted_recommendations(self, topic: str, context: Optional[Dict] = None) -> str:
        workflows = self.get_recommendations(topic, context)
        
        if not workflows:
            return ""
        
        formatted = "\n## Recommended Workflow Templates:\n\n"
        for workflow in workflows:
            formatted += f"**{workflow['name']}**\n"
            formatted += f"- Type: {workflow['script_type']}\n"
            formatted += f"- Use Case: {workflow['use_case']}\n"
            formatted += f"- Description: {workflow['description']}\n\n"
        
        return formatted
```

## 🧪 Testing Your Plugin

### Basic Plugin Tests

```python
# tests/test_dataset_recommender.py
import pytest
from unittest.mock import Mock, patch
from my_dataset_plugin.dataset_recommender import DatasetRecommender

class TestDatasetRecommender:
    """Test suite for DatasetRecommender plugin."""
    
    def test_plugin_initialization(self):
        """Test plugin initializes correctly."""
        recommender = DatasetRecommender()
        
        assert recommender.get_name() == "datasets"
        assert "datasets" in recommender.get_supported_types()
        assert recommender.is_enabled() == True
    
    def test_keyword_extraction(self):
        """Test keyword extraction from topics."""
        recommender = DatasetRecommender()
        keywords = recommender._extract_keywords("Climate Change Research Data")
        
        assert "climate" in keywords
        assert "change" in keywords
        assert "research" in keywords
        assert "data" in keywords
        assert "the" not in keywords  # Stop word filtered
    
    @patch('requests.get')
    def test_dataset_search_success(self, mock_get):
        """Test successful dataset search."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "hits": {
                "hits": [
                    {
                        "metadata": {
                            "title": "Climate Dataset 2023",
                            "description": "Comprehensive climate data"
                        },
                        "links": {"html": "https://zenodo.org/record/123"}
                    }
                ]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        recommender = DatasetRecommender()
        recommendations = recommender.get_recommendations("Climate Research")
        
        assert len(recommendations) > 0
        assert recommendations[0]["title"] == "Climate Dataset 2023"
        assert recommendations[0]["source"] == "Zenodo"
    
    @patch('requests.get')
    def test_api_error_handling(self, mock_get):
        """Test graceful handling of API errors."""
        mock_get.side_effect = requests.RequestException("API Error")
        
        recommender = DatasetRecommender()
        recommendations = recommender.get_recommendations("Any Topic")
        
        # Should return empty list, not crash
        assert recommendations == []
    
    def test_formatted_output(self):
        """Test formatted recommendation output."""
        recommender = DatasetRecommender()
        
        # Mock recommendations
        with patch.object(recommender, 'get_recommendations') as mock_get_rec:
            mock_get_rec.return_value = [
                {
                    "title": "Test Dataset",
                    "description": "Test description",
                    "url": "https://example.com",
                    "source": "Test Source",
                    "relevance_score": 8.5
                }
            ]
            
            formatted = recommender.get_formatted_recommendations("Test Topic")
            
            assert "## Relevant Research Datasets:" in formatted
            assert "**Test Dataset**" in formatted
            assert "Source: Test Source" in formatted
            assert "8.5/10" in formatted
```

### Integration Tests

```python
# tests/test_integration.py
import pytest
from doc_generator import DocumentationGenerator
from my_dataset_plugin.dataset_recommender import DatasetRecommender

def test_plugin_integrates_with_doc_generator(temp_dir):
    """Test that plugin integrates properly with doc-generator."""
    
    # Create minimal config files
    prompt_file = temp_dir / "prompt.yaml"
    terminology_file = temp_dir / "terminology.yaml"
    examples_dir = temp_dir / "examples"
    examples_dir.mkdir()
    
    prompt_file.write_text("system_prompt: 'Test prompt'")
    terminology_file.write_text("hpc_modules: []")
    
    # Mock plugin discovery to include our plugin
    from unittest.mock import patch
    mock_entry_point = Mock()
    mock_entry_point.name = "datasets"
    mock_entry_point.load.return_value = DatasetRecommender
    
    with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
        mock_eps.return_value = [mock_entry_point]
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            generator = DocumentationGenerator(
                prompt_yaml_path=str(prompt_file),
                terminology_path=str(terminology_file),
                examples_dir=str(examples_dir)
            )
            
            # Plugin should be loaded
            assert "datasets" in generator.plugin_manager.engines
            
            # Plugin should contribute to context
            context = generator._build_terminology_context("Research Data Analysis")
            # Context should contain plugin contributions (depends on mocking API calls)
```

## 🎛️ Plugin Configuration

### Configuration Files

```python
# my_dataset_plugin/config.py
import yaml
from pathlib import Path

class PluginConfig:
    """Configuration management for dataset plugin."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "default_sources.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        return {
            "dataset_sources": [
                {
                    "name": "Zenodo",
                    "api_url": "https://zenodo.org/api/records",
                    "enabled": True
                }
            ],
            "search_limits": {
                "max_results_per_source": 10,
                "total_max_results": 5
            }
        }
    
    def get_sources(self) -> list:
        return [s for s in self.config["dataset_sources"] if s.get("enabled", True)]
```

**config/default_sources.yaml:**
```yaml
dataset_sources:
  - name: "Zenodo"
    api_url: "https://zenodo.org/api/records"
    search_param: "q"
    enabled: true
    
  - name: "DataHub"
    api_url: "https://datahub.io/api/search"
    search_param: "query"
    enabled: true
    
  - name: "NASA Data"
    api_url: "https://data.nasa.gov/api/search"
    search_param: "q"
    enabled: false  # Disabled by default

search_limits:
  max_results_per_source: 10
  total_max_results: 5
  timeout_seconds: 10

relevance_scoring:
  title_match_weight: 3.0
  description_match_weight: 1.0
  keyword_bonus_threshold: 2
  keyword_bonus_multiplier: 1.2
```

### Environment Variables

```python
# Support environment-based configuration
import os

class DatasetRecommender(RecommendationEngine):
    def __init__(self, terminology: Optional[Dict] = None, **kwargs):
        super().__init__(terminology, **kwargs)
        
        # Allow environment variable overrides
        self.api_timeout = int(os.getenv('DATASET_API_TIMEOUT', '10'))
        self.max_results = int(os.getenv('DATASET_MAX_RESULTS', '5'))
        self.enable_caching = os.getenv('DATASET_ENABLE_CACHE', 'true').lower() == 'true'
```

## 🚀 Publishing Your Plugin

### Step 1: Prepare for Distribution

```bash
# Ensure your package is ready
cd my-dataset-plugin

# Run tests
python -m pytest

# Check package structure
python -m build --check

# Build distribution packages
python -m build
```

### Step 2: Publish to PyPI

```bash
# Install twine for uploading
pip install twine

# Upload to Test PyPI first
twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ my-dataset-plugin

# If everything works, upload to real PyPI
twine upload dist/*
```

### Step 3: Documentation

**README.md for your plugin:**
```markdown
# Dataset Recommender Plugin for doc-generator

Automatically discover and recommend relevant research datasets for your documentation.

## Installation

```bash
pip install my-dataset-plugin
```

## Usage

The plugin automatically integrates with doc-generator:

```bash
doc-gen --topic "Climate Research" --output-dir ./docs
```

Generated documentation will include relevant dataset recommendations.

## Configuration

Create `dataset_config.yaml` to customize data sources:

```yaml
dataset_sources:
  - name: "My Custom API"
    api_url: "https://my-api.com/search"
    enabled: true
```

## Supported Data Sources

- Zenodo
- DataHub  
- NASA Open Data
- Custom APIs (configurable)
```

## 🎯 Best Practices

### Plugin Design Principles

1. **Single Responsibility**: Each plugin should focus on one type of recommendation
2. **Fail Gracefully**: Handle API errors, network issues, and missing data
3. **Configurable**: Allow users to customize behavior through configuration
4. **Performant**: Cache results, limit API calls, handle timeouts
5. **Well-Tested**: Comprehensive test coverage including error conditions

### Error Handling

```python
def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
    """Get recommendations with robust error handling."""
    try:
        # Main recommendation logic
        return self._get_recommendations_impl(topic, context)
    except requests.RequestException as e:
        self.logger.warning(f"Network error in {self.get_name()}: {e}")
        return []
    except Exception as e:
        self.logger.error(f"Unexpected error in {self.get_name()}: {e}")
        return []  # Never crash the main generation process

def _get_recommendations_impl(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
    """Internal implementation with specific error handling."""
    # Your actual logic here
    pass
```

### Performance Optimization

```python
import functools
import time

class DatasetRecommender(RecommendationEngine):
    def __init__(self, terminology: Optional[Dict] = None, **kwargs):
        super().__init__(terminology, **kwargs)
        self._cache = {}
        self._cache_ttl = 3600  # 1 hour
    
    @functools.lru_cache(maxsize=100)
    def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
        """Cached recommendations to avoid repeated API calls."""
        cache_key = f"{topic}:{hash(str(context))}"
        
        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached_data
        
        # Get fresh data
        recommendations = self._fetch_recommendations(topic, context)
        
        # Cache results
        self._cache[cache_key] = (recommendations, time.time())
        
        return recommendations
```

### Logging and Debugging

```python
import logging

class DatasetRecommender(RecommendationEngine):
    def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
        self.logger.debug(f"Getting dataset recommendations for topic: {topic}")
        
        keywords = self._extract_keywords(topic)
        self.logger.debug(f"Extracted keywords: {keywords}")
        
        recommendations = []
        for source in self.dataset_sources:
            try:
                results = self._search_source(source, keywords)
                self.logger.debug(f"Found {len(results)} results from {source['name']}")
                recommendations.extend(results)
            except Exception as e:
                self.logger.warning(f"Error searching {source['name']}: {e}")
        
        self.logger.info(f"Returning {len(recommendations)} total recommendations")
        return recommendations
```

## 🔧 Advanced Plugin Features

### Custom Priority and Ordering

```python
class HighPriorityPlugin(RecommendationEngine):
    def get_priority(self) -> int:
        return 200  # Higher than default (50), will appear first
    
    def is_enabled(self) -> bool:
        # Could check environment variables, config files, etc.
        return os.getenv('ENABLE_HIGH_PRIORITY_PLUGIN', 'true').lower() == 'true'
```

### Context-Aware Recommendations

```python
def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
    """Provide context-aware recommendations."""
    recommendations = self._get_base_recommendations(topic)
    
    if context:
        # Adjust based on user preferences
        if context.get('user_level') == 'beginner':
            recommendations = [r for r in recommendations if r.get('difficulty', 'medium') == 'easy']
        
        # Filter by organization
        if context.get('organization') == 'FASRC':
            recommendations = [r for r in recommendations if 'fasrc' in r.get('tags', [])]
        
        # Limit results based on context
        max_results = context.get('max_results', 5)
        recommendations = recommendations[:max_results]
    
    return recommendations
```

### Plugin Dependencies

```python
# In pyproject.toml
dependencies = [
    "doc-generator>=1.1.0",
    "requests>=2.28.0",
    "beautifulsoup4>=4.11.0",  # For HTML parsing
    "pandas>=1.5.0",           # For data manipulation
    "scikit-learn>=1.1.0",     # For text similarity
]

# In plugin code
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class AdvancedDatasetRecommender(RecommendationEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        if HAS_SKLEARN:
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.use_advanced_matching = True
        else:
            self.logger.warning("scikit-learn not available, using basic matching")
            self.use_advanced_matching = False
```

## 📚 Plugin Examples Repository

### Community Plugin Ideas

Here are ideas for plugins that would benefit the community:

1. **Academic Paper Recommender**: Suggest relevant research papers from arXiv, PubMed
2. **Software License Checker**: Recommend appropriate licenses for software projects
3. **Container Image Recommender**: Suggest Docker/Singularity images for workflows
4. **Best Practices Recommender**: Suggest coding standards and best practices
5. **Security Scanner**: Recommend security best practices and vulnerability checks
6. **Performance Optimizer**: Suggest performance improvements and profiling tools
7. **Collaboration Tools**: Recommend git workflows, issue templates, documentation templates

### Example: Academic Paper Recommender

```python
class PaperRecommender(RecommendationEngine):
    """Recommends relevant academic papers from arXiv and PubMed."""
    
    def get_name(self) -> str:
        return "papers"
    
    def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
        papers = []
        
        # Search arXiv for CS/physics papers
        if self._is_cs_topic(topic):
            papers.extend(self._search_arxiv(topic))
        
        # Search PubMed for bio/medical papers  
        if self._is_bio_topic(topic):
            papers.extend(self._search_pubmed(topic))
        
        return sorted(papers, key=lambda x: x['relevance_score'], reverse=True)[:5]
    
    def _search_arxiv(self, topic: str) -> List[Dict]:
        # Implementation for arXiv API
        pass
    
    def _search_pubmed(self, topic: str) -> List[Dict]:
        # Implementation for PubMed API  
        pass
```

## ✅ Next Steps

After creating your plugin:

1. 📝 **Document Your Plugin**: Write clear README, usage examples, configuration docs
2. 🧪 **Test Thoroughly**: Unit tests, integration tests, error condition tests
3. 🚀 **Publish**: Share on PyPI, GitHub, announce to the community
4. 🤝 **Contribute**: Consider contributing your plugin to the main doc-generator repository
5. 🔄 **Maintain**: Keep dependencies updated, fix bugs, add features based on user feedback

## 🎉 Plugin Creation Mastery

You now have the knowledge to:
- Create powerful recommendation engine plugins
- Handle errors gracefully and perform efficiently
- Test plugins thoroughly with proper mocking
- Configure plugins for flexibility and customization
- Publish and distribute plugins to the community
- Follow best practices for maintainable code

**Start building amazing plugins!** 🔌✨

The doc-generator ecosystem awaits your contributions. Whether it's dataset recommendations, code templates, workflow suggestions, or entirely new types of assistance, your plugins will help researchers and developers create better documentation faster.

**Happy plugin developing!** 🎯🚀