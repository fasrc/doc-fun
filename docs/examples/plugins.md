# Plugin Examples

This page provides practical examples of creating, testing, and using plugins with doc-generator.

## Built-in Plugin Usage

### ModuleRecommender in Action

The built-in ModuleRecommender demonstrates how plugins enhance documentation:

```bash
# Generate documentation with module recommendations
doc-gen --topic "Python Machine Learning with GPU acceleration"
```

**Generated content includes:**
```html
<h2>Installation</h2>
<p>To get started with Python machine learning on the cluster, load the required modules:</p>

<pre><code>module load python/3.12.8-fasrc01
module load gcc/12.2.0-fasrc01  
module load cuda/12.9.1-fasrc01
</code></pre>

<p>These modules provide:</p>
<ul>
<li><strong>python/3.12.8-fasrc01</strong>: Python 3.12 with Anaconda distribution</li>
<li><strong>gcc/12.2.0-fasrc01</strong>: GNU Compiler Collection for building extensions</li>
<li><strong>cuda/12.9.1-fasrc01</strong>: NVIDIA CUDA toolkit for GPU acceleration</li>
</ul>
```

### Plugin Management

```bash
# View all available plugins
doc-gen --list-plugins

# Disable the modules plugin
doc-gen --topic "General Topic" --disable-plugins modules

# Enable only specific plugins (useful when you have multiple)
doc-gen --topic "Data Science" --enable-only modules,datasets
```

## Creating Simple Plugins

### 1. Basic Resource Recommender

```python
# my_plugins/resource_recommender.py
from typing import List, Dict, Optional
from doc_generator.plugins.base import RecommendationEngine

class ResourceRecommender(RecommendationEngine):
    """Recommends online resources and documentation links."""
    
    def __init__(self, terminology: Optional[Dict] = None, **kwargs):
        super().__init__(terminology, **kwargs)
        
        # Define resource database
        self.resources = {
            'python': [
                {
                    'title': 'Python Official Documentation',
                    'url': 'https://docs.python.org/',
                    'description': 'Comprehensive Python documentation',
                    'type': 'documentation'
                },
                {
                    'title': 'Real Python Tutorials',
                    'url': 'https://realpython.com/',
                    'description': 'Practical Python tutorials and guides',
                    'type': 'tutorial'
                }
            ],
            'machine learning': [
                {
                    'title': 'scikit-learn User Guide',
                    'url': 'https://scikit-learn.org/stable/user_guide.html',
                    'description': 'Machine learning library documentation',
                    'type': 'documentation'
                }
            ]
        }
    
    def get_name(self) -> str:
        return "resources"
    
    def get_supported_types(self) -> List[str]:
        return ["documentation", "tutorials", "references"]
    
    def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
        """Find relevant resources for the topic."""
        topic_lower = topic.lower()
        recommendations = []
        
        for category, resources in self.resources.items():
            if category in topic_lower:
                for resource in resources:
                    recommendation = resource.copy()
                    recommendation['relevance_score'] = self._calculate_relevance(topic_lower, category)
                    recommendations.append(recommendation)
        
        return sorted(recommendations, key=lambda x: x['relevance_score'], reverse=True)
    
    def _calculate_relevance(self, topic: str, category: str) -> float:
        """Calculate relevance score based on keyword matches."""
        # Simple keyword matching
        words = topic.split()
        category_words = category.split()
        
        matches = sum(1 for word in words if word in category_words)
        return min(10.0, matches * 3.0 + 5.0)  # Base score of 5, bonus for matches

# Test the plugin
if __name__ == "__main__":
    plugin = ResourceRecommender()
    results = plugin.get_recommendations("Python machine learning tutorial")
    
    for rec in results:
        print(f"Title: {rec['title']}")
        print(f"URL: {rec['url']}")
        print(f"Score: {rec['relevance_score']}")
        print("---")
```

### 2. Code Template Recommender

```python
# my_plugins/template_recommender.py
from typing import List, Dict, Optional
from doc_generator.plugins.base import RecommendationEngine

class TemplateRecommender(RecommendationEngine):
    """Recommends code templates and boilerplate for different languages."""
    
    def __init__(self, terminology: Optional[Dict] = None, **kwargs):
        super().__init__(terminology, **kwargs)
        
        self.templates = {
            'python': {
                'data_analysis': {
                    'name': 'Python Data Analysis Template',
                    'description': 'Template for data analysis with pandas and matplotlib',
                    'code': '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data.csv')

# Basic analysis
print(df.describe())
print(df.info())

# Visualization
plt.figure(figsize=(10, 6))
df.plot()
plt.show()''',
                    'language': 'python'
                },
                'machine_learning': {
                    'name': 'scikit-learn ML Template',
                    'description': 'Basic machine learning workflow template',
                    'code': '''from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))''',
                    'language': 'python'
                }
            },
            'slurm': {
                'basic_job': {
                    'name': 'Basic SLURM Job Script',
                    'description': 'Template for submitting jobs to SLURM scheduler',
                    'code': '''#!/bin/bash
#SBATCH -J my_job
#SBATCH -p shared
#SBATCH -t 1:00:00
#SBATCH --mem=4G
#SBATCH -o output_%j.out
#SBATCH -e error_%j.err

# Load required modules  
module load python/3.12.8-fasrc01

# Run your code
python my_script.py''',
                    'language': 'bash'
                }
            }
        }
    
    def get_name(self) -> str:
        return "templates"
    
    def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
        """Find relevant code templates."""
        topic_lower = topic.lower()
        recommendations = []
        
        for language, templates in self.templates.items():
            if language in topic_lower:
                for template_key, template in templates.items():
                    if any(keyword in topic_lower for keyword in template_key.split('_')):
                        recommendation = {
                            'title': template['name'],
                            'description': template['description'],
                            'code': template['code'],
                            'language': template['language'],
                            'type': 'code_template',
                            'relevance_score': self._calculate_template_score(topic_lower, template_key)
                        }
                        recommendations.append(recommendation)
        
        return sorted(recommendations, key=lambda x: x['relevance_score'], reverse=True)
    
    def _calculate_template_score(self, topic: str, template_key: str) -> float:
        """Calculate relevance score for templates."""
        template_words = template_key.replace('_', ' ').split()
        topic_words = topic.split()
        
        matches = sum(1 for word in template_words if word in topic_words)
        return matches * 2.5 + 5.0
    
    def get_formatted_recommendations(self, topic: str, context: Optional[Dict] = None) -> str:
        """Format templates for documentation."""
        templates = self.get_recommendations(topic, context)
        
        if not templates:
            return ""
        
        formatted = "\n## Code Templates\n\n"
        for template in templates:
            formatted += f"### {template['title']}\n\n"
            formatted += f"{template['description']}\n\n"
            formatted += f"```{template['language']}\n{template['code']}\n```\n\n"
        
        return formatted
```

### 3. Testing Your Plugins

```python
# tests/test_my_plugins.py
import pytest
from my_plugins.resource_recommender import ResourceRecommender
from my_plugins.template_recommender import TemplateRecommender

class TestResourceRecommender:
    def test_basic_functionality(self):
        plugin = ResourceRecommender()
        
        assert plugin.get_name() == "resources"
        assert "documentation" in plugin.get_supported_types()
    
    def test_python_recommendations(self):
        plugin = ResourceRecommender()
        results = plugin.get_recommendations("Python programming tutorial")
        
        assert len(results) > 0
        assert any("Python" in rec['title'] for rec in results)
        
        for rec in results:
            assert 'title' in rec
            assert 'url' in rec
            assert 'relevance_score' in rec
            assert isinstance(rec['relevance_score'], (int, float))
    
    def test_empty_topic(self):
        plugin = ResourceRecommender()
        results = plugin.get_recommendations("")
        
        assert isinstance(results, list)

class TestTemplateRecommender:
    def test_template_retrieval(self):
        plugin = TemplateRecommender()
        results = plugin.get_recommendations("Python data analysis")
        
        assert len(results) > 0
        template = results[0]
        
        assert 'code' in template
        assert 'language' in template
        assert template['language'] == 'python'
        assert 'import pandas' in template['code']
    
    def test_slurm_templates(self):
        plugin = TemplateRecommender()
        results = plugin.get_recommendations("SLURM job submission")
        
        assert len(results) > 0
        template = results[0]
        
        assert template['language'] == 'bash'
        assert '#SBATCH' in template['code']
    
    def test_formatted_output(self):
        plugin = TemplateRecommender()
        formatted = plugin.get_formatted_recommendations("Python machine learning")
        
        assert "## Code Templates" in formatted
        assert "```python" in formatted
```

## External API Plugins

### GitHub Repository Recommender

```python
# my_plugins/github_recommender.py
import requests
from typing import List, Dict, Optional
from doc_generator.plugins.base import RecommendationEngine

class GitHubRecommender(RecommendationEngine):
    """Recommends relevant GitHub repositories."""
    
    def __init__(self, terminology: Optional[Dict] = None, **kwargs):
        super().__init__(terminology, **kwargs)
        self.api_url = "https://api.github.com/search/repositories"
    
    def get_name(self) -> str:
        return "github"
    
    def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
        """Search GitHub for relevant repositories."""
        try:
            # Extract keywords for search
            keywords = self._extract_keywords(topic)
            query = " ".join(keywords[:3])  # Limit to top 3 keywords
            
            params = {
                'q': f"{query} language:python",
                'sort': 'stars',
                'order': 'desc',
                'per_page': 5
            }
            
            response = requests.get(self.api_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            repositories = []
            
            for repo in data.get('items', []):
                repository = {
                    'title': repo['full_name'],
                    'description': repo['description'] or 'No description available',
                    'url': repo['html_url'],
                    'stars': repo['stargazers_count'],
                    'language': repo['language'],
                    'relevance_score': self._calculate_github_score(repo, keywords),
                    'type': 'repository'
                }
                repositories.append(repository)
            
            return repositories
            
        except Exception as e:
            self.logger.warning(f"GitHub API error: {e}")
            return []
    
    def _extract_keywords(self, topic: str) -> List[str]:
        """Extract searchable keywords from topic."""
        import re
        words = re.findall(r'\w+', topic.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with'}
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def _calculate_github_score(self, repo: dict, keywords: List[str]) -> float:
        """Calculate relevance score for GitHub repository."""
        score = 0.0
        
        # Star-based score (0-5 points)
        stars = repo['stargazers_count']
        star_score = min(5.0, stars / 1000)
        score += star_score
        
        # Keyword matching in name and description
        text = f"{repo['name']} {repo.get('description', '')}".lower()
        keyword_score = sum(2.0 for keyword in keywords if keyword in text)
        score += keyword_score
        
        return min(10.0, score)
```

### Paper Recommender (arXiv)

```python
# my_plugins/arxiv_recommender.py
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from doc_generator.plugins.base import RecommendationEngine

class ArxivRecommender(RecommendationEngine):
    """Recommends academic papers from arXiv."""
    
    def __init__(self, terminology: Optional[Dict] = None, **kwargs):
        super().__init__(terminology, **kwargs)
        self.api_url = "http://export.arxiv.org/api/query"
    
    def get_name(self) -> str:
        return "papers"
    
    def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
        """Search arXiv for relevant papers."""
        try:
            # Build search query
            keywords = self._extract_keywords(topic)
            query = " AND ".join(keywords[:3])
            
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': 5,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = requests.get(self.api_url, params=params, timeout=15)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                paper = self._parse_arxiv_entry(entry, keywords)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            self.logger.warning(f"arXiv API error: {e}")
            return []
    
    def _parse_arxiv_entry(self, entry, keywords: List[str]) -> Dict:
        """Parse a single arXiv entry."""
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        title = entry.find('atom:title', ns)
        summary = entry.find('atom:summary', ns)
        link = entry.find('atom:id', ns)
        published = entry.find('atom:published', ns)
        
        # Get authors
        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns)
            if name is not None:
                authors.append(name.text)
        
        return {
            'title': title.text.strip() if title is not None else 'Unknown Title',
            'description': summary.text.strip()[:200] + '...' if summary is not None else '',
            'url': link.text if link is not None else '',
            'authors': ', '.join(authors[:3]),
            'published': published.text[:10] if published is not None else '',
            'relevance_score': self._calculate_paper_score(title.text if title else '', keywords),
            'type': 'academic_paper'
        }
    
    def _calculate_paper_score(self, title: str, keywords: List[str]) -> float:
        """Calculate relevance score for papers."""
        title_lower = title.lower()
        score = 5.0  # Base score
        
        for keyword in keywords:
            if keyword in title_lower:
                score += 1.5
        
        return min(10.0, score)
    
    def _extract_keywords(self, topic: str) -> List[str]:
        """Extract keywords suitable for academic search."""
        import re
        words = re.findall(r'\w+', topic.lower())
        
        # Filter for more academic/technical terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'using', 'how', 'what'}
        technical_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        return technical_words[:5]  # Limit to avoid overly complex queries
```

## Plugin Integration Examples

### Plugin Package Structure

```
my-doc-plugins/
├── pyproject.toml
├── README.md
├── src/
│   └── my_doc_plugins/
│       ├── __init__.py
│       ├── resource_recommender.py
│       ├── template_recommender.py
│       ├── github_recommender.py
│       └── arxiv_recommender.py
├── tests/
│   ├── test_resource_recommender.py
│   ├── test_template_recommender.py
│   └── test_integration.py
└── examples/
    └── usage_examples.py
```

### pyproject.toml for Plugin Package

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-doc-plugins"
version = "0.1.0"
description = "Additional plugins for doc-generator"
authors = [{name = "Your Name", email = "your.email@example.com"}]
readme = "README.md"
license = {text = "MIT"}
keywords = ["documentation", "plugins", "recommendations"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
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

# Register plugins with doc-generator
[project.entry-points."doc_generator.plugins"]
resources = "my_doc_plugins.resource_recommender:ResourceRecommender"
templates = "my_doc_plugins.template_recommender:TemplateRecommender"
github = "my_doc_plugins.github_recommender:GitHubRecommender"
papers = "my_doc_plugins.arxiv_recommender:ArxivRecommender"

[project.urls]
Homepage = "https://github.com/yourusername/my-doc-plugins"
Repository = "https://github.com/yourusername/my-doc-plugins"
```

### Installation and Usage

```bash
# Install your plugin package
pip install -e ./my-doc-plugins

# Verify plugins are discovered
doc-gen --list-plugins

# Use with documentation generation
doc-gen --topic "Python machine learning research" --output-dir ./research-docs

# Generated documentation will now include:
# - HPC module recommendations (built-in)
# - Online resource links (your resource plugin)
# - Code templates (your template plugin)
# - GitHub repository suggestions (your github plugin)
# - Academic paper references (your arxiv plugin)
```

### Advanced Plugin Usage

```bash
# Enable only specific plugins
doc-gen --topic "Academic Research" --enable-only modules,papers,resources

# Disable external API plugins for faster generation
doc-gen --topic "Quick Test" --disable-plugins github,papers

# Use plugins with custom configuration
export GITHUB_PLUGIN_LANGUAGE=python
export ARXIV_PLUGIN_CATEGORIES="cs.LG,cs.AI"
doc-gen --topic "Machine Learning" --verbose
```

## Plugin Testing Strategies

### Integration Testing

```python
# tests/test_plugin_integration.py
import pytest
from doc_generator import DocumentationGenerator
from my_doc_plugins.github_recommender import GitHubRecommender

def test_plugin_integration_with_generator():
    """Test that custom plugins work with DocumentationGenerator."""
    
    # Mock the plugin discovery to include our plugin
    from unittest.mock import patch, Mock
    
    mock_entry_point = Mock()
    mock_entry_point.name = "github"
    mock_entry_point.load.return_value = GitHubRecommender
    
    with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
        mock_eps.return_value = [mock_entry_point]
        
        # Create generator with our plugin
        generator = DocumentationGenerator()
        
        # Plugin should be loaded
        assert "github" in generator.plugin_manager.engines
        
        # Test that plugin contributes to context
        # (Note: This requires mocking the GitHub API)
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                'items': [{
                    'full_name': 'test/repo',
                    'description': 'Test repository',
                    'html_url': 'https://github.com/test/repo',
                    'stargazers_count': 100,
                    'language': 'Python'
                }]
            }
            mock_get.return_value = mock_response
            
            context = generator._build_terminology_context("Python testing")
            # Verify GitHub recommendations are included
            assert "test/repo" in str(context)
```

### Performance Testing

```python
# tests/test_plugin_performance.py
import time
import pytest
from my_doc_plugins.github_recommender import GitHubRecommender

def test_github_plugin_performance():
    """Test that GitHub plugin responds within reasonable time."""
    plugin = GitHubRecommender()
    
    start_time = time.time()
    results = plugin.get_recommendations("Python machine learning")
    end_time = time.time()
    
    # Should complete within 15 seconds (GitHub API timeout + processing)
    assert (end_time - start_time) < 15.0
    
    # Should return some results (assuming internet connection)
    assert isinstance(results, list)

def test_plugin_error_handling():
    """Test that plugins handle API errors gracefully."""
    plugin = GitHubRecommender()
    
    # Mock network error
    with patch('requests.get') as mock_get:
        mock_get.side_effect = requests.RequestException("Network error")
        
        # Should not crash
        results = plugin.get_recommendations("Any topic")
        
        # Should return empty list
        assert results == []
```

---

These examples show how to create powerful, real-world plugins that extend doc-generator's capabilities. Start with simple plugins and gradually add more sophisticated features as needed. For more details, see the comprehensive [Creating Plugins Guide](../guides/creating-plugins.md).