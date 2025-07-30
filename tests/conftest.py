import pytest
import tempfile
import shutil
from pathlib import Path
import yaml
import os
from unittest.mock import Mock

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_yaml_config():
    """Sample YAML configuration for testing."""
    return {
        'terms': {
            'FASRC': 'Faculty Arts and Sciences Research Computing'
        },
        'documentation_structure': [
            'Description',
            'Installation',
            'Usage',
            'Examples',
            'References'
        ],
        'examples': 'Sample examples content'
    }

@pytest.fixture
def sample_terminology():
    """Sample terminology configuration."""
    return {
        'hpc_modules': [
            {
                'name': 'python/3.12.8-fasrc01',
                'description': 'Python 3.12 with Anaconda distribution',
                'category': 'programming'
            },
            {
                'name': 'python/3.10.13-fasrc01',
                'description': 'Python 3.10 with Anaconda distribution',
                'category': 'programming'
            },
            {
                'name': 'gcc/14.2.0-fasrc01',
                'description': 'GNU Compiler Collection (latest)',
                'category': 'compiler'
            },
            {
                'name': 'R/4.4.3-fasrc01',
                'description': 'R statistical computing environment',
                'category': 'programming'
            },
            {
                'name': 'cuda/12.9.1-fasrc01',
                'description': 'CUDA toolkit for GPU computing',
                'category': 'gpu'
            },
            {
                'name': 'matlab/R2024b-fasrc01',
                'description': 'MATLAB technical computing platform',
                'category': 'programming'
            }
        ],
        'code_examples': {
            'python': [
                {
                    'name': 'Test Script',
                    'file_path': 'src/test.py',
                    'language': 'python',
                    'description': 'A sample Python script for testing',
                    'relevance_score': 10.0
                }
            ]
        },
        'cluster_commands': [
            {
                'name': 'sbatch',
                'description': 'Submit a batch job',
                'usage': 'sbatch script.sh'
            }
        ]
    }

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    client = Mock()
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "Generated documentation content"
    client.chat.completions.create.return_value = response
    return client