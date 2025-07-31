# Advanced Workflows

This page demonstrates sophisticated usage patterns, automation, and integration examples for doc-generator.

!!! warning "Example Plugins Notice"
    Some examples in this guide reference **DatasetRecommender** and **WorkflowRecommender** plugins. These are **conceptual examples (TBD)** for illustration purposes and are not currently implemented.

## 🔄 Batch Processing

### Processing Multiple Topics

```bash
# Create topics file
cat > topics.txt << 'EOF'
Python Data Analysis with Pandas
R Statistical Computing on HPC
MATLAB Numerical Methods
Julia High-Performance Computing
Fortran Parallel Programming
EOF

# Process all topics with consistent settings
while IFS= read -r topic; do
    echo "Processing: $topic"
    doc-gen --topic "$topic" \
      --output-dir batch-output \
      --runs 2 \
      --model gpt-4 \
      --analyze
    sleep 3  # Rate limiting
done < topics.txt
```

### Parallel Processing with GNU Parallel

```bash
# Install GNU parallel if not available
# brew install parallel  # macOS
# sudo apt install parallel  # Ubuntu

# Process topics in parallel (be careful with API rate limits)
parallel -j 2 --delay 5 \
  'doc-gen --topic "{}" --output-dir parallel-output --runs 1' \
  :::: topics.txt
```

### Automated Quality Assessment

```bash
#!/bin/bash
# batch-generate-with-quality.sh

TOPICS=(
    "Deep Learning with TensorFlow"
    "Containerized Applications with Singularity"
    "Distributed Computing with Spark"
    "Quantum Computing Simulations"
)

OUTPUT_BASE="quality-docs"
mkdir -p "$OUTPUT_BASE"

for topic in "${TOPICS[@]}"; do
    echo "=== Processing: $topic ==="
    
    # Generate with full quality pipeline
    doc-gen --topic "$topic" \
      --output-dir "$OUTPUT_BASE" \
      --runs 3 \
      --model gpt-4 \
      --temperature 0.3 \
      --analyze \
      --quality-eval \
      --verbose
    
    echo "✓ Completed: $topic"
    echo "---"
    sleep 5
done

echo "🎉 Batch processing complete!"
echo "Results in: $OUTPUT_BASE/"
```

## 🎛️ Advanced Configuration

### Environment-Based Configuration

```bash
# .env.production
OPENAI_API_KEY=prod-key-here
DOC_GEN_DEFAULT_MODEL=gpt-4
DOC_GEN_DEFAULT_TEMPERATURE=0.2
DOC_GEN_DEFAULT_RUNS=3
DOC_GEN_OUTPUT_DIR=production-docs
DOC_GEN_VERBOSE=true

# .env.development  
OPENAI_API_KEY=dev-key-here
DOC_GEN_DEFAULT_MODEL=gpt-3.5-turbo
DOC_GEN_DEFAULT_TEMPERATURE=0.5
DOC_GEN_DEFAULT_RUNS=1
DOC_GEN_OUTPUT_DIR=dev-docs
DOC_GEN_VERBOSE=false
```

```bash
# Use different environments
source .env.production && doc-gen --topic "Production Guide"
source .env.development && doc-gen --topic "Development Test"
```

### Multi-Organization Setup

```yaml
# config/organizations.yaml
organizations:
  fasrc:
    name: "FASRC Research Computing"
    prompt_template: "prompts/generator/fasrc.yaml"
    terminology: "terminology/fasrc.yaml"
    output_prefix: "fasrc"
    
  mit:
    name: "MIT SuperCloud"
    prompt_template: "prompts/generator/mit.yaml"  
    terminology: "terminology/mit.yaml"
    output_prefix: "mit"
```

```bash
# Organization-specific generation script
#!/bin/bash
generate_for_org() {
    local org=$1
    local topic=$2
    
    doc-gen --topic "$topic" \
      --prompt-yaml "prompts/generator/${org}.yaml" \
      --terminology-path "terminology/${org}.yaml" \
      --output-dir "docs/${org}" \
      --runs 2
}

# Generate for multiple organizations
generate_for_org "fasrc" "Python Machine Learning"
generate_for_org "mit" "Python Machine Learning"
```

## 🧪 Advanced Analysis and Quality Control

### Custom Quality Metrics

```python
# scripts/quality_analyzer.py
import yaml
import json
from pathlib import Path
from doc_generator import DocumentationGenerator, DocumentAnalyzer

class AdvancedQualityAnalyzer:
    def __init__(self):
        self.analyzer = DocumentAnalyzer()
        self.quality_thresholds = {
            'min_length': 2000,
            'min_code_examples': 2,
            'min_sections': 4,
            'min_links': 1
        }
    
    def batch_analyze(self, docs_dir: str) -> dict:
        results = {}
        docs_path = Path(docs_dir)
        
        for html_file in docs_path.glob("*.html"):
            with open(html_file, 'r') as f:
                content = f.read()
            
            analysis = self.analyzer.analyze_document(content)
            quality_score = self.calculate_quality_score(analysis)
            
            results[html_file.name] = {
                'analysis': analysis,
                'quality_score': quality_score,
                'passes_threshold': quality_score > 7.0
            }
        
        return results
    
    def calculate_quality_score(self, analysis: dict) -> float:
        score = 0.0
        
        # Length score (0-3 points)
        length_score = min(3.0, analysis['total_length'] / 1000)
        score += length_score
        
        # Code examples score (0-2 points)
        code_score = min(2.0, analysis['code_blocks'] * 0.5)
        score += code_score
        
        # Structure score (0-3 points)
        section_score = min(3.0, analysis['sections'] * 0.6)
        score += section_score
        
        # Links score (0-2 points)
        links_score = min(2.0, analysis['links'] * 0.4)
        score += links_score
        
        return round(score, 1)

# Usage
analyzer = AdvancedQualityAnalyzer()
results = analyzer.batch_analyze("./output")
print(json.dumps(results, indent=2))
```

### A/B Testing Framework

```python
# scripts/ab_testing.py
import random
from doc_generator import DocumentationGenerator

class ABTestingFramework:
    def __init__(self):
        self.generator = DocumentationGenerator()
        
    def run_temperature_test(self, topic: str, temperatures: list, runs_per_temp: int = 3):
        """Test different temperature settings."""
        results = {}
        
        for temp in temperatures:
            temp_results = []
            for run in range(runs_per_temp):
                result = self.generator.generate_documentation(
                    query=topic,
                    temperature=temp,
                    runs=1
                )[0]
                temp_results.append(result)
            
            results[f"temp_{temp}"] = temp_results
        
        return results
    
    def run_model_comparison(self, topic: str, models: list):
        """Compare different models."""
        results = {}
        
        for model in models:
            result = self.generator.generate_documentation(
                query=topic,
                model=model,
                runs=1
            )[0]
            results[model] = result
            
        return results

# Usage
tester = ABTestingFramework()

# Test temperature variations
temp_results = tester.run_temperature_test(
    "Python Machine Learning",
    temperatures=[0.1, 0.3, 0.5, 0.7],
    runs_per_temp=2
)

# Test model comparison
model_results = tester.run_model_comparison(
    "Database Design Patterns",
    models=["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"]
)
```

## 🔌 Advanced Plugin Development

### Multi-Source Plugin

```python
# plugins/advanced_dataset_recommender.py
import asyncio
import aiohttp
from typing import List, Dict, Optional
from doc_generator.plugins.base import RecommendationEngine

class AdvancedDatasetRecommender(RecommendationEngine):
    """Advanced dataset recommender with multiple APIs and caching. (TBD - Conceptual Example)"""
    
    def __init__(self, terminology: Optional[Dict] = None, **kwargs):
        super().__init__(terminology, **kwargs)
        self.sources = [
            {
                'name': 'Zenodo',
                'url': 'https://zenodo.org/api/records',
                'parser': self._parse_zenodo
            },
            {
                'name': 'DataHub',
                'url': 'https://datahub.io/api/search',
                'parser': self._parse_datahub
            },
            {
                'name': 'NASA',
                'url': 'https://data.nasa.gov/api/search',
                'parser': self._parse_nasa
            }
        ]
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def get_recommendations_async(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
        """Async version for better performance."""
        keywords = self._extract_keywords(topic)
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._search_source_async(session, source, keywords)
                for source in self.sources
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine and rank results
        all_datasets = []
        for result in results:
            if isinstance(result, list):
                all_datasets.extend(result)
        
        # Sort by relevance and return top 10
        all_datasets.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return all_datasets[:10]
    
    def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
        """Sync wrapper for async method."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.get_recommendations_async(topic, context)
        )
    
    async def _search_source_async(self, session: aiohttp.ClientSession, source: dict, keywords: List[str]) -> List[Dict]:
        """Search a single source asynchronously."""
        try:
            query = ' '.join(keywords)
            async with session.get(
                source['url'],
                params={'q': query, 'size': 5},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                data = await response.json()
                return source['parser'](data, keywords)
        except Exception as e:
            self.logger.warning(f"Error searching {source['name']}: {e}")
            return []
    
    def _parse_zenodo(self, data: dict, keywords: List[str]) -> List[Dict]:
        """Parse Zenodo API response."""
        datasets = []
        for hit in data.get('hits', {}).get('hits', []):
            metadata = hit.get('metadata', {})
            dataset = {
                'title': metadata.get('title', 'Unknown Dataset'),
                'description': metadata.get('description', '')[:200] + '...',
                'url': hit.get('links', {}).get('html', ''),
                'source': 'Zenodo',
                'relevance_score': self._calculate_relevance(metadata.get('title', ''), keywords)
            }
            datasets.append(dataset)
        return datasets
```

### Plugin with Machine Learning

```python
# plugins/ml_recommender.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from doc_generator.plugins.base import RecommendationEngine

class MLRecommender(RecommendationEngine):
    """ML-powered recommendation engine using TF-IDF similarity."""
    
    def __init__(self, terminology: Optional[Dict] = None, **kwargs):
        super().__init__(terminology, **kwargs)
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.knowledge_base = self._build_knowledge_base()
        self.vectors = None
        self._fit_vectorizer()
    
    def _build_knowledge_base(self) -> List[Dict]:
        """Build knowledge base from various sources."""
        knowledge = []
        
        # HPC modules
        if self.terminology and 'hpc_modules' in self.terminology:
            for module in self.terminology['hpc_modules']:
                knowledge.append({
                    'type': 'module',
                    'title': f"HPC Module: {module['name']}",
                    'content': f"{module['name']} {module.get('description', '')} {' '.join(module.get('keywords', []))}",
                    'recommendation': f"module load {module['name']}"
                })
        
        # Add more knowledge sources here
        # - Documentation links
        # - Code examples
        # - Best practices
        
        return knowledge
    
    def _fit_vectorizer(self):
        """Fit TF-IDF vectorizer on knowledge base."""
        if self.knowledge_base:
            texts = [item['content'] for item in self.knowledge_base]
            self.vectors = self.vectorizer.fit_transform(texts)
    
    def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
        """Get recommendations using ML similarity."""
        if not self.knowledge_base or self.vectors is None:
            return []
        
        # Vectorize the topic
        topic_vector = self.vectorizer.transform([topic])
        
        # Calculate similarities
        similarities = cosine_similarity(topic_vector, self.vectors)[0]
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[::-1][:5]
        
        recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                item = self.knowledge_base[idx]
                recommendations.append({
                    'title': item['title'],
                    'description': item.get('recommendation', ''),
                    'relevance_score': float(similarities[idx]) * 10,
                    'type': item['type']
                })
        
        return recommendations
```

## 🔗 Integration Examples

### CI/CD Pipeline Integration

```yaml
# .github/workflows/docs-generation.yml
name: Generate Documentation
on:
  push:
    paths: ['docs/topics/*.txt']
  workflow_dispatch:

jobs:
  generate-docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -e .
        
    - name: Generate documentation
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        python scripts/batch_generate.py
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: generated-docs
        path: output/
        
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./output
```

### WordPress Integration

```python
# scripts/wordpress_publisher.py
import requests
from pathlib import Path
import base64

class WordPressPublisher:
    def __init__(self, site_url: str, username: str, password: str):
        self.site_url = site_url.rstrip('/')
        self.api_url = f"{self.site_url}/wp-json/wp/v2"
        self.auth = (username, password)
    
    def publish_documentation(self, html_file: Path, title: str = None):
        """Publish generated HTML as WordPress post."""
        with open(html_file, 'r') as f:
            content = f.read()
        
        # Extract title from filename if not provided
        if not title:
            title = html_file.stem.replace('_', ' ').title()
        
        post_data = {
            'title': title,
            'content': content,
            'status': 'publish',
            'categories': [1],  # Adjust category ID
            'tags': ['documentation', 'auto-generated']
        }
        
        response = requests.post(
            f"{self.api_url}/posts",
            json=post_data,
            auth=self.auth
        )
        
        if response.status_code == 201:
            post_id = response.json()['id']
            post_url = response.json()['link']
            print(f"✓ Published: {title}")
            print(f"  URL: {post_url}")
            return post_id
        else:
            print(f"✗ Failed to publish: {title}")
            print(f"  Error: {response.text}")
            return None

# Usage
publisher = WordPressPublisher(
    site_url="https://your-site.com",
    username="your-username", 
    password="your-app-password"
)

# Publish all generated docs
output_dir = Path("./output")
for html_file in output_dir.glob("*.html"):
    publisher.publish_documentation(html_file)
```

### Slack Integration

```python
# scripts/slack_notifier.py
import requests
import json
from pathlib import Path

class SlackNotifier:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def notify_generation_complete(self, topic: str, output_files: list, quality_score: float = None):
        """Send notification when documentation generation completes."""
        
        message = {
            "text": f"📚 Documentation Generated: {topic}",
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"📚 Documentation Generated"}
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Topic:*\n{topic}"},
                        {"type": "mrkdwn", "text": f"*Files:*\n{len(output_files)} generated"}
                    ]
                }
            ]
        }
        
        if quality_score:
            message["blocks"].append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Quality Score:* {quality_score}/10"}
            })
        
        # Add file list
        file_list = "\n".join([f"• `{f}`" for f in output_files[:5]])
        if len(output_files) > 5:
            file_list += f"\n• ... and {len(output_files) - 5} more"
            
        message["blocks"].append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Generated Files:*\n{file_list}"}
        })
        
        response = requests.post(self.webhook_url, json=message)
        return response.status_code == 200

# Usage in batch script
notifier = SlackNotifier("https://hooks.slack.com/services/YOUR/WEBHOOK/URL")
notifier.notify_generation_complete(
    topic="Python Machine Learning",
    output_files=["ml_guide_v1.html", "ml_guide_v2.html"],
    quality_score=8.5
)
```

## 📊 Performance Optimization

### Concurrent Processing

```python
# scripts/concurrent_generator.py
import asyncio
import concurrent.futures
from doc_generator import DocumentationGenerator

class ConcurrentGenerator:
    def __init__(self, max_workers: int = 3):
        self.generator = DocumentationGenerator()
        self.max_workers = max_workers
    
    def batch_generate(self, topics: list, **kwargs) -> dict:
        """Generate documentation for multiple topics concurrently."""
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_topic = {
                executor.submit(self._generate_single, topic, **kwargs): topic
                for topic in topics
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_topic):
                topic = future_to_topic[future]
                try:
                    result = future.result()
                    results[topic] = result
                    print(f"✓ Completed: {topic}")
                except Exception as e:
                    print(f"✗ Failed: {topic} - {e}")
                    results[topic] = None
        
        return results
    
    def _generate_single(self, topic: str, **kwargs):
        """Generate documentation for a single topic."""
        return self.generator.generate_documentation(
            query=topic,
            **kwargs
        )

# Usage
generator = ConcurrentGenerator(max_workers=2)
topics = [
    "Python Data Science",
    "R Statistical Analysis", 
    "Julia Numerical Computing"
]

results = generator.batch_generate(
    topics,
    runs=2,
    model="gpt-4",
    temperature=0.3
)
```

### Caching Implementation

```python
# scripts/cached_generator.py
import hashlib
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from doc_generator import DocumentationGenerator

class CachedGenerator:
    def __init__(self, cache_dir: str = ".cache", cache_ttl: int = 3600):
        self.generator = DocumentationGenerator()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = cache_ttl
    
    def generate_with_cache(self, query: str, **kwargs) -> list:
        """Generate documentation with caching."""
        # Create cache key
        cache_key = self._create_cache_key(query, kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Check cache
        if self._is_cache_valid(cache_file):
            print(f"📦 Using cached result for: {query}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Generate fresh content
        print(f"🔄 Generating fresh content for: {query}")
        results = self.generator.generate_documentation(query=query, **kwargs)
        
        # Cache results
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)
        
        return results
    
    def _create_cache_key(self, query: str, kwargs: dict) -> str:
        """Create deterministic cache key."""
        cache_data = {'query': query, 'kwargs': kwargs}
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file exists and is still valid."""
        if not cache_file.exists():
            return False
        
        file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        expiry_time = datetime.now() - timedelta(seconds=self.cache_ttl)
        
        return file_time > expiry_time
    
    def clear_cache(self):
        """Clear all cached results."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print(f"🗑️ Cleared cache directory: {self.cache_dir}")

# Usage
cached_generator = CachedGenerator(cache_ttl=7200)  # 2 hours

# First call - generates fresh content
results1 = cached_generator.generate_with_cache(
    "Python Machine Learning",
    runs=2,
    model="gpt-4"
)

# Second call - uses cached content
results2 = cached_generator.generate_with_cache(
    "Python Machine Learning", 
    runs=2,
    model="gpt-4"
)
```

---

These advanced examples demonstrate the full power of doc-generator for sophisticated workflows, automation, and integration scenarios. For more specific use cases, check out the [Plugin Examples](plugins.md) or [Basic Usage](basic.md) guides.