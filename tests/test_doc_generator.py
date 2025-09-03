import pytest
import yaml
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import sys
import os

# Add the src directory to the path so we can import doc_generator
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

from doc_generator import DocumentationGenerator, DocumentAnalyzer, GPTQualityEvaluator, CodeExampleScanner
from doc_generator.plugins.modules import ModuleRecommender


class TestDocumentationGenerator:
    """Test cases for DocumentationGenerator class."""

    def test_init_with_defaults(self, temp_dir, sample_yaml_config, sample_terminology, mock_plugin_discovery, sample_plugins):
        """Test initialization with default parameters."""
        # Create temporary config files
        prompt_file = temp_dir / "prompt.yaml"
        terminology_file = temp_dir / "terminology.yaml"
        shots_dir = temp_dir / "examples"
        shots_dir.mkdir()
        
        with open(prompt_file, 'w') as f:
            yaml.dump(sample_yaml_config, f)
        with open(terminology_file, 'w') as f:
            yaml.dump(sample_terminology, f)

        with mock_plugin_discovery(sample_plugins):
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                with patch('doc_generator.generator.OpenAI') as mock_openai:
                    generator = DocumentationGenerator(
                        prompt_yaml_path=str(prompt_file),
                        shots_dir=str(shots_dir),
                        terminology_path=str(terminology_file)
                    )
                    
                    assert generator.prompt_config == sample_yaml_config
                    assert generator.terminology == sample_terminology
                    assert hasattr(generator, 'plugin_manager')
                    assert len(generator.plugin_manager.engines) > 0
                    mock_openai.assert_called_once_with(api_key='test-key')

    def test_load_examples_from_directory(self, temp_dir, sample_yaml_config, sample_terminology, mock_plugin_discovery, sample_plugins):
        """Test loading examples from directory."""
        prompt_file = temp_dir / "prompt.yaml"
        terminology_file = temp_dir / "terminology.yaml"
        shots_dir = temp_dir / "examples"
        shots_dir.mkdir()
        
        # Create sample YAML files with dict structure
        (shots_dir / "example1.yaml").write_text("example: content1")
        (shots_dir / "example2.yaml").write_text("example: content2")
        
        with open(prompt_file, 'w') as f:
            yaml.dump(sample_yaml_config, f)
        with open(terminology_file, 'w') as f:
            yaml.dump(sample_terminology, f)

        with mock_plugin_discovery(sample_plugins):
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                with patch('doc_generator.core.OpenAI'):
                    generator = DocumentationGenerator(
                        prompt_yaml_path=str(prompt_file),
                        shots_dir=str(shots_dir),
                        terminology_path=str(terminology_file)
                    )
                    
                    assert len(generator.examples) == 2

    def test_generate_documentation(self, temp_dir, sample_yaml_config, sample_terminology, mock_openai_client, mock_plugin_discovery, sample_plugins):
        """Test documentation generation."""
        prompt_file = temp_dir / "prompt.yaml"
        terminology_file = temp_dir / "terminology.yaml"
        shots_dir = temp_dir / "examples"
        shots_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        with open(prompt_file, 'w') as f:
            yaml.dump(sample_yaml_config, f)
        with open(terminology_file, 'w') as f:
            yaml.dump(sample_terminology, f)

        with mock_plugin_discovery(sample_plugins):
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                with patch('doc_generator.core.Path.cwd', return_value=temp_dir):
                    # Mock the OpenAI provider's generate_completion method
                    with patch('doc_generator.providers.openai_provider.OpenAI') as mock_openai_class:
                        mock_openai_class.return_value = mock_openai_client
                        
                        generator = DocumentationGenerator(
                            prompt_yaml_path=str(prompt_file),
                            shots_dir=str(shots_dir),
                            terminology_path=str(terminology_file)
                        )
                        
                        result = generator.generate_documentation("Python programming", runs=1)
                        
                        assert isinstance(result, list)
                        assert len(result) == 1
                        mock_openai_client.chat.completions.create.assert_called()

    def test_prompt_config_loading(self, temp_dir, sample_yaml_config, sample_terminology, mock_plugin_discovery, sample_plugins):
        """Test that prompt configuration is loaded correctly."""
        prompt_file = temp_dir / "prompt.yaml"
        terminology_file = temp_dir / "terminology.yaml"
        shots_dir = temp_dir / "examples"
        shots_dir.mkdir()
        
        with open(prompt_file, 'w') as f:
            yaml.dump(sample_yaml_config, f)
        with open(terminology_file, 'w') as f:
            yaml.dump(sample_terminology, f)

        with mock_plugin_discovery(sample_plugins):
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                with patch('doc_generator.core.OpenAI'):
                    generator = DocumentationGenerator(
                        prompt_yaml_path=str(prompt_file),
                        shots_dir=str(shots_dir),
                        terminology_path=str(terminology_file)
                    )
                    
                    assert 'terms' in generator.prompt_config
                    assert generator.prompt_config['terms']['FASRC'] == 'Faculty Arts and Sciences Research Computing'


class TestDocumentAnalyzer:
    """Test cases for DocumentAnalyzer class."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = DocumentAnalyzer()
        expected_sections = ['Description', 'Installation', 'Usage', 'Examples', 'References']
        assert analyzer.section_headers == expected_sections

    def test_extract_sections(self):
        """Test HTML section extraction."""
        html_content = '''
        <html>
        <body>
            <h1>Description</h1>
            <p>This is the description section.</p>
            <h1>Installation</h1>
            <p>Installation instructions here.</p>
            <h1>Usage</h1>
            <p>Usage information.</p>
        </body>
        </html>
        '''
        
        analyzer = DocumentAnalyzer()
        sections = analyzer.extract_sections(html_content)
        
        assert 'Description' in sections
        assert 'Installation' in sections
        assert 'Usage' in sections
        assert 'This is the description section.' in sections['Description']

    def test_calculate_section_score(self):
        """Test section scoring algorithm."""
        analyzer = DocumentAnalyzer()
        
        # Test with good content (realistic expectations)
        good_content = "This is a detailed section with multiple sentences. " * 10 + \
                      "<pre><code>sample code</code></pre>" + \
                      '<a href="http://example.com">link</a>'
        
        score = analyzer.calculate_section_score(good_content, 'Description')
        assert score > 10  # Should get some score for good content
        
        # Test with minimal content
        minimal_content = "Short."
        score = analyzer.calculate_section_score(minimal_content, 'Description')
        assert score >= 0  # Should get a low but non-negative score

    def test_analyze_document(self):
        """Test full document analysis."""
        html_content = '''
        <html>
        <body>
            <h1>Description</h1>
            <p>This is a comprehensive description with multiple sentences. It provides detailed information about the topic and covers various aspects thoroughly.</p>
            
            <h1>Installation</h1>
            <p>Installation instructions with code examples.</p>
            <pre><code>pip install example</code></pre>
            
            <h1>Usage</h1>
            <p>Usage information with helpful <a href="http://example.com">links</a>.</p>
        </body>
        </html>
        '''
        
        analyzer = DocumentAnalyzer()
        sections = analyzer.extract_sections(html_content)
        
        assert 'Description' in sections
        assert 'Installation' in sections
        assert 'Usage' in sections
        assert len(sections) > 0


class TestGPTQualityEvaluator:
    """Test cases for GPTQualityEvaluator class."""

    def test_init_with_config(self, temp_dir):
        """Test evaluator initialization with config file."""
        config = {
            'analysis_prompts': {
                'technical_accuracy': 'Test prompt for {section_name} about {topic}: {content}'
            }
        }
        config_file = temp_dir / "analysis_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        mock_client = Mock()
        evaluator = GPTQualityEvaluator(mock_client, analysis_prompt_path=str(config_file))
        
        assert evaluator.analysis_config == config

    def test_create_evaluation_prompt(self, temp_dir):
        """Test prompt creation."""
        config = {
            'analysis_prompts': {
                'technical_accuracy': 'Evaluate {section_name} about {topic}: {content}'
            }
        }
        config_file = temp_dir / "analysis_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        mock_client = Mock()
        evaluator = GPTQualityEvaluator(mock_client, analysis_prompt_path=str(config_file))
        
        prompt = evaluator.create_evaluation_prompt("test content", "Description", "Python", "technical_accuracy")
        
        assert "Evaluate Description about Python: test content" == prompt

    def test_parse_gpt_response(self, temp_dir):
        """Test GPT response parsing."""
        config = {'analysis_prompts': {'test': 'test'}}
        config_file = temp_dir / "analysis_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        mock_client = Mock()
        evaluator = GPTQualityEvaluator(mock_client, analysis_prompt_path=str(config_file))
        
        # Test valid JSON response
        response = '{"score": 85, "explanation": "Good quality"}'
        score, explanation = evaluator.parse_gpt_response(response)
        assert score == 85.0
        assert explanation == "Good quality"
        
        # Test invalid response (fallback behavior returns 50.0)
        response = "Invalid response"
        score, explanation = evaluator.parse_gpt_response(response)
        assert score == 50.0
        assert explanation == "Invalid response"


class TestCodeExampleScanner:
    """Test cases for CodeExampleScanner class."""

    def test_init(self):
        """Test scanner initialization."""
        scanner = CodeExampleScanner()
        assert hasattr(scanner, 'has_pygments')

    def test_detect_language_by_extension(self, temp_dir):
        """Test language detection by file extension."""
        scanner = CodeExampleScanner()
        
        python_file = temp_dir / "test.py"
        python_file.write_text("print('hello')")
        
        language = scanner._detect_language(python_file)
        assert language == 'python'
        
        cpp_file = temp_dir / "test.cpp"
        cpp_file.write_text("#include <iostream>")
        
        language = scanner._detect_language(cpp_file)
        assert language == 'cpp'

    def test_detect_comment_style(self):
        """Test comment style detection."""
        scanner = CodeExampleScanner()
        
        # Python-style comments
        python_content = "# This is a Python comment\nprint('hello')"
        assert scanner._detect_comment_style(python_content) == '#'
        
        # C-style comments
        c_content = "// This is a C comment\nint main() {}"
        assert scanner._detect_comment_style(c_content) == '//'

    def test_extract_description(self):
        """Test description extraction from comments."""
        scanner = CodeExampleScanner()
        
        content = '''# This is a comprehensive description of the code
# It explains what the program does in detail
import sys
'''
        description = scanner._extract_description(content)
        assert "comprehensive description" in description
        assert len(description) > 10

    def test_extract_file_info(self, temp_dir):
        """Test file information extraction."""
        scanner = CodeExampleScanner()
        
        # Create a test Python file
        test_file = temp_dir / "example.py"
        test_file.write_text('''#!/usr/bin/env python3
# This is a sample Python script for demonstration
# It shows how to calculate mathematical constants

import math

def calculate_pi():
    """Calculate pi approximation."""
    return 3.14159

if __name__ == "__main__":
    print(f"Pi is approximately {calculate_pi()}")
''')
        
        file_info = scanner._extract_file_info(test_file)
        
        assert file_info is not None
        assert file_info['language'] == 'python'
        assert file_info['name'] == 'Example'
        assert 'sample Python script' in file_info['description']
        assert file_info['file_size'] > 0

    def test_scan_directory(self, temp_dir):
        """Test directory scanning."""
        scanner = CodeExampleScanner()
        
        # Create test files
        (temp_dir / "script.py").write_text("# Python script\nprint('hello')")
        (temp_dir / "program.cpp").write_text("// C++ program\n#include <iostream>")
        (temp_dir / "readme.txt").write_text("This is not code")
        
        results = scanner.scan_directory(str(temp_dir), max_files=10)
        
        # Should find at least 2 code files (may include txt file depending on pygments)
        assert len(results) >= 2
        languages = [r['language'] for r in results]
        assert 'python' in languages
        assert 'cpp' in languages

    def test_update_terminology_file(self, temp_dir):
        """Test terminology file updates."""
        scanner = CodeExampleScanner()
        terminology_file = temp_dir / "terminology.yaml"
        
        # Create initial terminology
        initial_terminology = {'existing_key': 'existing_value'}
        with open(terminology_file, 'w') as f:
            yaml.dump(initial_terminology, f)
        
        # Sample code examples
        code_examples = [
            {
                'name': 'Test Script',
                'language': 'python',
                'description': 'A test Python script',
                'file_path': 'test.py'
            }
        ]
        
        scanner.update_terminology_file(str(terminology_file), code_examples)
        
        # Load updated terminology
        with open(terminology_file, 'r') as f:
            updated_terminology = yaml.safe_load(f)
        
        assert 'existing_key' in updated_terminology
        assert 'code_examples' in updated_terminology
        assert 'python' in updated_terminology['code_examples']
        assert len(updated_terminology['code_examples']['python']) == 1


class TestModuleRecommender:
    """Test cases for ModuleRecommender class."""
    
    def test_init_with_modules(self, sample_terminology):
        """Test ModuleRecommender initialization."""
        recommender = ModuleRecommender(terminology=sample_terminology)
        
        assert recommender.hpc_modules == sample_terminology['hpc_modules']
        assert 'python' in recommender.keyword_mappings
        assert 'gcc' in recommender.keyword_mappings
        assert 'cuda' in recommender.keyword_mappings
    
    def test_extract_keywords_from_topic(self, sample_terminology):
        """Test keyword extraction from topics."""
        recommender = ModuleRecommender(terminology=sample_terminology)
        
        # Test basic topic
        keywords = recommender._extract_keywords_from_topic("Parallel Python")
        assert 'parallel' in keywords
        assert 'python' in keywords
        assert 'the' not in keywords  # Stop word should be filtered
        
        # Test with stop words
        keywords = recommender._extract_keywords_from_topic("How to use C Programming")
        assert 'programming' in keywords
        assert 'how' not in keywords
        assert 'to' not in keywords
        # 'use' is only 3 chars so it passes the length filter, but it should be in stop words
    
    def test_calculate_module_relevance(self, sample_terminology):
        """Test module relevance scoring."""
        recommender = ModuleRecommender(terminology=sample_terminology)
        
        # Test Python module with Python keywords
        python_module = sample_terminology['hpc_modules'][0]  # python/3.12.8-fasrc01
        keywords = ['python', 'parallel']
        score = recommender._calculate_module_relevance(python_module, keywords)
        assert score > 0
        
        # Test irrelevant module - use keywords that shouldn't match CUDA
        cuda_module = sample_terminology['hpc_modules'][4]  # cuda/12.9.1-fasrc01  
        keywords = ['biology', 'chemistry']  # These shouldn't match CUDA at all
        score = recommender._calculate_module_relevance(cuda_module, keywords)
        assert score == 0
        
        # Test R module with statistics keywords
        r_module = sample_terminology['hpc_modules'][3]  # R/4.4.3-fasrc01
        keywords = ['statistics', 'data']
        score = recommender._calculate_module_relevance(r_module, keywords)
        assert score > 0
    
    def test_get_priority_score(self, sample_terminology):
        """Test priority scoring for modules."""
        recommender = ModuleRecommender(terminology=sample_terminology)
        
        # Test FASRC01 module (should get priority)
        python_module = sample_terminology['hpc_modules'][0]  # python/3.12.8-fasrc01
        score = recommender._get_priority_score(python_module)
        assert score >= 1.0  # Should get points for fasrc01 and programming category
        
        # Test module with programming category
        gcc_module = sample_terminology['hpc_modules'][2]  # gcc/14.2.0-fasrc01
        score = recommender._get_priority_score(gcc_module)
        assert score >= 1.0  # Should get points for compiler category
    
    def test_get_modules_for_topic(self, sample_terminology):
        """Test getting recommended modules for a topic."""
        recommender = ModuleRecommender(terminology=sample_terminology)
        
        # Test Python topic
        modules = recommender.get_recommendations("Parallel Python")
        assert len(modules) > 0
        assert len(modules) <= 3  # max_modules default
        
        # Should return Python modules first
        top_module = modules[0]
        assert 'python' in top_module['name'].lower()
        assert 'load_command' in top_module
        assert top_module['load_command'].startswith('module load')
        assert 'relevance_score' in top_module
        
        # Test CUDA topic
        modules = recommender.get_recommendations("GPU Computing with CUDA")
        assert len(modules) > 0
        top_module = modules[0]
        assert 'cuda' in top_module['name'].lower()
        
        # Test topic with no matches
        modules = recommender.get_recommendations("Nonexistent Technology")
        assert len(modules) == 0
    
    def test_get_formatted_recommendations(self, sample_terminology):
        """Test formatted output for documentation context."""
        recommender = ModuleRecommender(terminology=sample_terminology)
        
        # Test Python recommendations
        formatted = recommender.get_formatted_recommendations("Parallel Python")
        assert "Recommended Modules:" in formatted
        assert "module load python/" in formatted
        assert "Description:" in formatted
        
        # Test topic with no matches
        formatted = recommender.get_formatted_recommendations("Nonexistent Technology")
        assert formatted == ""
    
    def test_special_r_case(self, sample_terminology):
        """Test special handling for R modules."""
        recommender = ModuleRecommender(terminology=sample_terminology)
        
        # Test that statistics keywords trigger R module detection
        r_module = sample_terminology['hpc_modules'][3]  # R/4.4.3-fasrc01
        keywords = ['statistics', 'analysis']
        score = recommender._calculate_module_relevance(r_module, keywords)
        assert score >= 15.0  # Should get the special R bonus


class TestDocumentationGeneratorEnhancements:
    """Test cases for enhanced DocumentationGenerator functionality."""
    
    def test_build_system_prompt_with_parameterization(self, temp_dir, sample_terminology, mock_plugin_discovery, sample_plugins):
        """Test enhanced system prompt building with parameterization."""
        prompt_file = temp_dir / "prompt.yaml"
        terminology_file = temp_dir / "terminology.yaml"
        shots_dir = temp_dir / "examples"
        shots_dir.mkdir()
        
        # Create prompt config with parameterized template
        prompt_config = {
            'system_prompt': 'You are creating {format} docs for {topic} at {organization}.',
            'placeholders': {
                'format': 'HTML',
                'organization': 'FASRC'
            }
        }
        
        with open(prompt_file, 'w') as f:
            yaml.dump(prompt_config, f)
        with open(terminology_file, 'w') as f:
            yaml.dump(sample_terminology, f)
        
        with mock_plugin_discovery(sample_plugins):
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                with patch('doc_generator.core.OpenAI'):
                    generator = DocumentationGenerator(
                        prompt_yaml_path=str(prompt_file),
                        shots_dir=str(shots_dir),
                        terminology_path=str(terminology_file)
                    )
                    
                    system_prompt = generator._build_system_prompt("Python Programming")
                    
                    # Check parameterization worked
                    assert "HTML docs" in system_prompt
                    assert "Python Programming" in system_prompt
                    assert "FASRC" in system_prompt
                    
                    # Check terminology context is included (plugins should contribute)
                    assert len(system_prompt) > len("You are creating HTML docs for Python Programming at FASRC.")
    
    def test_build_terminology_context_with_module_recommender(self, temp_dir, sample_terminology, mock_plugin_discovery, sample_plugins):
        """Test terminology context building with ModuleRecommender integration."""
        prompt_file = temp_dir / "prompt.yaml" 
        terminology_file = temp_dir / "terminology.yaml"
        shots_dir = temp_dir / "examples"
        shots_dir.mkdir()
        
        with open(prompt_file, 'w') as f:
            yaml.dump({'system_prompt': 'test'}, f)
        with open(terminology_file, 'w') as f:
            yaml.dump(sample_terminology, f)
        
        with mock_plugin_discovery(sample_plugins):
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                with patch('doc_generator.core.OpenAI'):
                    generator = DocumentationGenerator(
                        prompt_yaml_path=str(prompt_file),
                        shots_dir=str(shots_dir), 
                        terminology_path=str(terminology_file)
                    )
                    
                    context = generator._build_terminology_context("Parallel Python")
                    
                    # Check plugin integration (may vary based on mock plugins)
                    assert isinstance(context, str)
                    
                    # Check other terminology sections still present
                    if 'cluster_commands' in sample_terminology:
                        assert "sbatch" in context  # cluster_commands should be included
    
    def test_find_relevant_code_examples_integration(self, temp_dir, sample_terminology, mock_plugin_discovery, sample_plugins):
        """Test code examples integration in terminology context."""
        prompt_file = temp_dir / "prompt.yaml"
        terminology_file = temp_dir / "terminology.yaml"
        shots_dir = temp_dir / "examples"
        shots_dir.mkdir()
        
        with open(prompt_file, 'w') as f:
            yaml.dump({'system_prompt': 'test'}, f)
        with open(terminology_file, 'w') as f:
            yaml.dump(sample_terminology, f)
        
        with mock_plugin_discovery(sample_plugins):
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                with patch('doc_generator.core.OpenAI'):
                    generator = DocumentationGenerator(
                        prompt_yaml_path=str(prompt_file),
                        shots_dir=str(shots_dir),
                        terminology_path=str(terminology_file)
                    )
                    
                    context = generator._build_terminology_context("Python Programming")
                    
                    # Plugin integration should work
                    assert isinstance(context, str)
                    # Basic terminology should be present
                    assert len(context) > 0


class TestDocumentationGeneratorExtended:
    """Extended tests for DocumentationGenerator to improve coverage."""
    
    def test_setup_provider_auto_success(self, temp_dir, sample_yaml_config, sample_terminology, mock_plugin_discovery, sample_plugins):
        """Test provider setup with auto selection - success case."""
        prompt_file = temp_dir / "prompt.yaml"
        terminology_file = temp_dir / "terminology.yaml"
        shots_dir = temp_dir / "examples"
        shots_dir.mkdir()
        
        with open(prompt_file, 'w') as f:
            yaml.dump(sample_yaml_config, f)
        with open(terminology_file, 'w') as f:
            yaml.dump(sample_terminology, f)
        
        with mock_plugin_discovery(sample_plugins):
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                with patch('doc_generator.core.OpenAI'):
                    generator = DocumentationGenerator(
                        prompt_yaml_path=str(prompt_file),
                        shots_dir=str(shots_dir),
                        terminology_path=str(terminology_file),
                        provider='auto'
                    )
                    # Should have set default provider through auto selection
                    assert generator.default_provider is not None
    
    def test_setup_provider_specific_success(self, temp_dir, sample_yaml_config, sample_terminology, mock_plugin_discovery, sample_plugins):
        """Test provider setup with specific provider - success case."""
        prompt_file = temp_dir / "prompt.yaml"
        terminology_file = temp_dir / "terminology.yaml"
        shots_dir = temp_dir / "examples"
        shots_dir.mkdir()
        
        with open(prompt_file, 'w') as f:
            yaml.dump(sample_yaml_config, f)
        with open(terminology_file, 'w') as f:
            yaml.dump(sample_terminology, f)
        
        with mock_plugin_discovery(sample_plugins):
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                with patch('doc_generator.core.OpenAI'):
                    # Mock provider manager to return openai as available
                    with patch('doc_generator.providers.manager.ProviderManager') as mock_pm:
                        mock_pm.return_value.get_available_providers.return_value = ['openai']
                        
                        generator = DocumentationGenerator(
                            prompt_yaml_path=str(prompt_file),
                            shots_dir=str(shots_dir),
                            terminology_path=str(terminology_file),
                            provider='openai'
                        )
                        assert generator.default_provider == 'openai'
    
    def test_setup_provider_unavailable_error(self, temp_dir, sample_yaml_config, sample_terminology, mock_plugin_discovery, sample_plugins):
        """Test provider setup with unavailable provider - error case."""
        prompt_file = temp_dir / "prompt.yaml"
        terminology_file = temp_dir / "terminology.yaml"
        shots_dir = temp_dir / "examples"
        shots_dir.mkdir()
        
        with open(prompt_file, 'w') as f:
            yaml.dump(sample_yaml_config, f)
        with open(terminology_file, 'w') as f:
            yaml.dump(sample_terminology, f)
        
        with mock_plugin_discovery(sample_plugins):
            with patch('doc_generator.providers.manager.ProviderManager') as mock_pm:
                mock_pm.return_value.get_available_providers.return_value = []
                
                from doc_generator.exceptions import ProviderError
                with pytest.raises(ProviderError):
                    DocumentationGenerator(
                        prompt_yaml_path=str(prompt_file),
                        shots_dir=str(shots_dir),
                        terminology_path=str(terminology_file),
                        provider='nonexistent'
                    )
    
    def test_extract_topic_from_query_patterns(self):
        """Test topic extraction from various query patterns."""
        generator = Mock()
        generator._extract_topic_from_query = DocumentationGenerator._extract_topic_from_query
        
        # Test various patterns
        test_cases = [
            ("documentation for Python Programming", "python_programming"),
            ("using Machine Learning", "machine_learning"),
            ("about Parallel Computing", "parallel_computing"),
            ("for Deep Learning documentation", "deep_learning"),
            ("create docs for Data Science", "data_science"),
            ("generate documentation for Neural Networks on GPUs", "neural_networks"),
        ]
        
        for query, expected in test_cases:
            result = generator._extract_topic_from_query(generator, query)
            assert result == expected
    
    def test_extract_topic_from_query_fallback(self):
        """Test topic extraction fallback mechanisms."""
        generator = Mock()
        generator._extract_topic_from_query = DocumentationGenerator._extract_topic_from_query
        
        # Test fallback with meaningful words
        result = generator._extract_topic_from_query(generator, "quantum computing algorithms")
        assert result == "quantum_computing_algorithms"
        
        # Test final fallback
        result = generator._extract_topic_from_query(generator, "create make generate")
        assert result == "documentation"
    
    def test_extract_topic_keywords(self):
        """Test keyword extraction from topics."""
        generator = Mock()
        generator._extract_topic_keywords = DocumentationGenerator._extract_topic_keywords
        
        # Test basic keyword extraction
        result = generator._extract_topic_keywords(generator, "Python Machine Learning")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(keyword, str) for keyword in result)
        
        # Test with underscores - should still be treated as one keyword
        result = generator._extract_topic_keywords(generator, "deep_learning_neural_networks")
        assert len(result) >= 1  # May be treated as single keyword
    
    def test_extract_target_language(self):
        """Test programming language extraction."""
        generator = Mock()
        generator._extract_target_language = DocumentationGenerator._extract_target_language
        
        test_cases = [
            (["python", "programming", "tutorial"], "python"),
            (["c", "algorithms", "guide"], "c"),
            (["fortran", "numerical", "computing"], "fortran"),
            (["mpi", "parallel"], "c"),  # MPI maps to C
            (["machine", "learning", "concepts"], None),  # No specific language
        ]
        
        for keywords, expected in test_cases:
            result = generator._extract_target_language(generator, keywords)
            assert result == expected
    
    def test_calculate_example_relevance(self):
        """Test example relevance calculation."""
        generator = Mock()
        generator._calculate_example_relevance = DocumentationGenerator._calculate_example_relevance
        
        example = {
            'file_path': 'machine_learning/python_examples.py',
            'description': 'python scikit-learn tutorial',
            'name': 'ml_example.py'
        }
        keywords = ['machine', 'learning', 'python']
        
        relevance = generator._calculate_example_relevance(generator, example, keywords)
        assert isinstance(relevance, (int, float))
        assert relevance >= 0
    
    def test_find_relevant_code_examples_with_examples(self):
        """Test finding relevant code examples when examples exist."""
        generator = Mock()
        generator.terminology = {
            'code_examples': {
                'python': [
                    {'file_path': 'ml/example.py', 'description': 'machine learning', 'name': 'ml_example'},
                    {'file_path': 'web/scraper.py', 'description': 'web scraping', 'name': 'scraper'}
                ],
                'java': [
                    {'file_path': 'spring/app.java', 'description': 'spring boot', 'name': 'app'}
                ]
            }
        }
        generator._calculate_example_relevance = Mock(side_effect=[0.8, 0.6, 0.2])
        generator._extract_target_language = Mock(return_value='python')
        generator._find_relevant_code_examples = DocumentationGenerator._find_relevant_code_examples
        
        keywords = ['python', 'programming']
        result = generator._find_relevant_code_examples(generator, keywords)
        assert isinstance(result, list)
        assert len(result) <= 8  # Should be limited to top 8
    
    def test_find_relevant_code_examples_empty(self):
        """Test finding relevant code examples when no examples exist."""
        generator = Mock()
        generator.terminology = {}  # No code_examples key
        generator._find_relevant_code_examples = DocumentationGenerator._find_relevant_code_examples
        
        keywords = ['python', 'programming']
        result = generator._find_relevant_code_examples(generator, keywords)
        assert result == []
    
    def test_find_relevant_modules_with_plugin(self):
        """Test finding relevant modules with plugin manager."""
        generator = Mock()
        generator.terminology = {
            'hpc_modules': [
                {'name': 'scipy', 'description': 'scientific computing library', 'category': 'scientific'},
                {'name': 'numpy', 'description': 'numerical computing library', 'category': 'math'}
            ]
        }
        generator.plugin_manager = Mock()
        mock_engine = Mock()
        mock_engine.get_modules_for_topic.return_value = [
            {'name': 'numpy', 'description': 'numerical computing'}
        ]
        generator.plugin_manager.get_recommendation_engines.return_value = [mock_engine]
        generator._find_relevant_modules = DocumentationGenerator._find_relevant_modules
        
        keywords = ['python', 'programming']
        result = generator._find_relevant_modules(generator, keywords)
        assert isinstance(result, list)
    
    def test_find_relevant_modules_without_plugin(self):
        """Test finding relevant modules without plugin manager."""
        generator = Mock()
        generator.terminology = {}  # No hpc_modules key
        generator.plugin_manager = None
        generator._find_relevant_modules = DocumentationGenerator._find_relevant_modules
        
        keywords = ['python', 'programming']
        result = generator._find_relevant_modules(generator, keywords)
        assert result == []
    
    def test_build_terminology_context_comprehensive(self):
        """Test comprehensive terminology context building."""
        generator = Mock()
        generator.terminology = {
            'terms': {
                'python': 'A programming language',
                'machine_learning': 'AI subset'
            },
            'cluster_commands': [
                {'name': 'srun', 'description': 'Run job', 'usage': 'srun -n 1 job'}
            ]
        }
        generator._extract_topic_keywords = Mock(return_value=['python', 'machine', 'learning'])
        generator.plugin_manager = Mock()
        generator.plugin_manager.get_formatted_recommendations = Mock(return_value="Recommended modules: scikit-learn")
        generator._find_relevant_code_examples = Mock(return_value=[])
        generator._find_relevant_modules = Mock(return_value=[])
        generator._build_terminology_context = DocumentationGenerator._build_terminology_context
        
        result = generator._build_terminology_context(generator, "Python Machine Learning")
        assert isinstance(result, str)
        assert 'scikit-learn' in result
    
    def test_build_system_prompt_with_context(self):
        """Test system prompt building with context."""
        generator = Mock()
        generator.prompt_config = {
            'system_prompt': 'You are a documentation expert for {topic}.'
        }
        generator._build_terminology_context = Mock(return_value="HPC environment info")
        generator._build_system_prompt = DocumentationGenerator._build_system_prompt
        
        result = generator._build_system_prompt(generator, "Test topic")
        assert isinstance(result, str)
        assert 'Test topic' in result
        assert 'documentation expert' in result
    
    def test_load_prompt_config_error_handling(self, temp_dir):
        """Test prompt config loading with error cases."""
        # Test with invalid YAML
        prompt_file = temp_dir / "invalid.yaml"
        prompt_file.write_text("invalid: yaml: content: [")
        
        generator = Mock()
        generator.prompt_yaml_path = str(prompt_file)
        generator.logger = Mock()
        generator._load_prompt_config = DocumentationGenerator._load_prompt_config
        
        with pytest.raises(Exception):  # Should raise YAML parsing error
            generator._load_prompt_config(generator)
    
    def test_load_prompt_config_with_cache_key(self, temp_dir):
        """Test prompt config loading with cache key validation."""
        config = {
            'system_prompt': 'test prompt {terminology_context}',
            'cache_key_template': '{topic}_{provider}_{model}'
        }
        prompt_file = temp_dir / "config.yaml"
        with open(prompt_file, 'w') as f:
            yaml.dump(config, f)
        
        generator = Mock()
        generator.prompt_yaml_path = prompt_file
        generator.logger = Mock()
        generator._load_prompt_config = DocumentationGenerator._load_prompt_config
        
        result = generator._load_prompt_config(generator)
        assert result['system_prompt'] == config['system_prompt']
        assert result['cache_key_template'] == config['cache_key_template']
    
    def test_load_terminology_error_handling(self, temp_dir):
        """Test terminology loading with error cases."""
        # Test with missing file
        terminology_file = temp_dir / "nonexistent.yaml"
        
        generator = Mock()
        generator.terminology_path = terminology_file
        generator.logger = Mock()
        generator._load_terminology = DocumentationGenerator._load_terminology
        
        result = generator._load_terminology(generator)
        assert result == {}
    
    def test_load_examples_error_handling(self, temp_dir):
        """Test examples loading with error cases."""
        # Test with missing directory
        shots_dir = temp_dir / "nonexistent"
        
        generator = Mock()
        generator.shots_dir = shots_dir
        generator.logger = Mock()
        generator._load_examples = DocumentationGenerator._load_examples
        
        result = generator._load_examples(generator)
        assert result == []
    
    def test_load_examples_with_mixed_files(self, temp_dir):
        """Test examples loading with mixed file types."""
        shots_dir = temp_dir / "examples"
        shots_dir.mkdir()
        
        # Create valid YAML
        (shots_dir / "valid.yaml").write_text("example: valid content")
        # Create invalid YAML
        (shots_dir / "invalid.yaml").write_text("invalid: yaml: [")
        # Create non-YAML file
        (shots_dir / "readme.txt").write_text("This is not YAML")
        
        generator = Mock()
        generator.shots_dir = shots_dir
        generator.logger = Mock()
        generator._load_examples = DocumentationGenerator._load_examples
        
        result = generator._load_examples(generator)
        # Should only load valid YAML files
        assert len(result) == 1
        assert result[0]['example'] == 'valid content'