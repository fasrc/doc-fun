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