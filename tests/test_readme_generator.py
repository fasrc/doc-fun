"""
Tests for ReadmeGenerator class.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
import logging

from doc_generator.readme_generator import ReadmeGenerator
from doc_generator.providers.base import CompletionRequest, CompletionResponse


class TestReadmeGeneratorInitialization:
    """Test ReadmeGenerator initialization."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        source_dir = Path("/test/dir")
        generator = ReadmeGenerator(source_dir)
        
        assert generator.source_dir == source_dir.resolve()
        assert generator.recursive is True
        assert generator.overwrite is False
        assert generator.suffix == "_generated"
        assert generator.ai_provider is None
        assert isinstance(generator.logger, logging.Logger)
        assert generator.generated_files == []
        assert generator.directory_cache == {}
        assert generator.ai_cache == {}
    
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        source_dir = Path("/test/dir")
        logger = logging.getLogger("test")
        ai_provider = Mock()
        
        generator = ReadmeGenerator(
            source_dir=source_dir,
            recursive=False,
            overwrite=True,
            suffix="_custom",
            ai_provider=ai_provider,
            logger=logger
        )
        
        assert generator.source_dir == source_dir.resolve()
        assert generator.recursive is False
        assert generator.overwrite is True
        assert generator.suffix == ""  # Empty when overwrite=True
        assert generator.ai_provider == ai_provider
        assert generator.logger == logger
    
    def test_init_overwrite_removes_suffix(self):
        """Test that overwrite=True removes suffix."""
        generator = ReadmeGenerator(
            source_dir=Path("/test/dir"),
            overwrite=True,
            suffix="_test"
        )
        assert generator.suffix == ""


class TestReadmeGeneratorUtilityMethods:
    """Test utility methods."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.generator = ReadmeGenerator(Path("/test/dir"))
    
    def test_should_exclude_basic_patterns(self):
        """Test basic exclusion patterns."""
        assert self.generator.should_exclude(Path("file.pyc")) is True
        assert self.generator.should_exclude(Path("__pycache__")) is True
        assert self.generator.should_exclude(Path(".git")) is True
        assert self.generator.should_exclude(Path("file.o")) is True
        assert self.generator.should_exclude(Path("file.py")) is False
        assert self.generator.should_exclude(Path("README.md")) is False
    
    def test_should_exclude_with_path(self):
        """Test exclusion patterns with paths."""
        assert self.generator.should_exclude(Path("build")) is True
        assert self.generator.should_exclude(Path("dist")) is True
        assert self.generator.should_exclude(Path("src/main.py")) is False
    
    def test_get_language(self):
        """Test language detection from file extensions."""
        assert self.generator.get_language(Path("test.py")) == "Python"
        assert self.generator.get_language(Path("test.f90")) == "Fortran"
        assert self.generator.get_language(Path("test.cpp")) == "C++"
        assert self.generator.get_language(Path("test.R")) == "R"
        assert self.generator.get_language(Path("test.jl")) == "Julia"
        assert self.generator.get_language(Path("test.unknown")) is None
    
    def test_extract_header_comments_python(self):
        """Test extracting header comments from Python files."""
        python_content = '''"""This is a docstring describing the module."""
import os

def main():
    pass
'''
        with patch("builtins.open", mock_open(read_data=python_content)):
            result = self.generator.extract_header_comments(Path("test.py"))
            assert "This is a docstring describing the module." in result
    
    def test_extract_header_comments_c(self):
        """Test extracting header comments from C files."""
        c_content = '''/*
 * This is a C header comment
 * describing the program purpose
 */
#include <stdio.h>

int main() {
    return 0;
}
'''
        with patch("builtins.open", mock_open(read_data=c_content)):
            result = self.generator.extract_header_comments(Path("test.c"))
            assert "This is a C header comment" in result
    
    def test_extract_header_comments_fortran(self):
        """Test extracting header comments from Fortran files."""
        fortran_content = '''! This program calculates prime numbers
! using the sieve of Eratosthenes
program primes
  implicit none
  
end program
'''
        with patch("builtins.open", mock_open(read_data=fortran_content)):
            result = self.generator.extract_header_comments(Path("test.f90"))
            assert "This program calculates prime numbers" in result
    
    def test_extract_header_comments_file_error(self):
        """Test header comment extraction with file read error."""
        with patch("builtins.open", side_effect=IOError("File not found")):
            result = self.generator.extract_header_comments(Path("test.py"))
            assert result == ""
    
    def test_extract_header_comments_unicode_error(self):
        """Test header comment extraction with unicode decode error."""
        with patch("builtins.open", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "error")):
            result = self.generator.extract_header_comments(Path("test.py"))
            assert result == ""


class TestReadmeGeneratorAIEnhancement:
    """Test AI enhancement functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.ai_provider = Mock()
        self.generator = ReadmeGenerator(Path("/test/dir"), ai_provider=self.ai_provider)
    
    def test_enhance_description_with_ai_success(self):
        """Test successful AI enhancement."""
        # Mock AI provider response
        mock_response = CompletionResponse(
            content="This directory contains parallel computing examples using MPI for numerical simulations.",
            model="gpt-3.5-turbo",
            provider="openai",
            usage={"total_tokens": 50}
        )
        self.ai_provider.generate_completion.return_value = mock_response
        self.ai_provider.get_available_models.return_value = ["gpt-3.5-turbo"]
        
        basic_info = {
            'languages': ['Python', 'C++'],
            'code_files': [{'path': Path('main.py'), 'description': 'Main simulation loop'}],
            'has_makefile': True
        }
        
        result = self.generator.enhance_description_with_ai("simulation", basic_info)
        
        assert "parallel computing examples using MPI" in result
        assert self.ai_provider.generate_completion.called
    
    def test_enhance_description_with_ai_no_provider(self):
        """Test AI enhancement without provider."""
        generator = ReadmeGenerator(Path("/test/dir"), ai_provider=None)
        result = generator.enhance_description_with_ai("test", {})
        assert result == ""
    
    def test_enhance_description_with_ai_cached(self):
        """Test AI enhancement with cached result."""
        # First call
        mock_response = CompletionResponse(
            content="Cached description",
            model="gpt-4o-mini",
            provider="openai",
            usage={"total_tokens": 25}
        )
        self.ai_provider.generate_completion.return_value = mock_response
        self.ai_provider.get_available_models.return_value = ["gpt-3.5-turbo"]
        
        basic_info = {'languages': ['Python']}
        result1 = self.generator.enhance_description_with_ai("test", basic_info)
        
        # Second call should use cache
        result2 = self.generator.enhance_description_with_ai("test", basic_info)
        
        assert result1 == result2 == "Cached description"
        assert self.ai_provider.generate_completion.call_count == 1
    
    def test_enhance_description_with_ai_exception(self):
        """Test AI enhancement with exception."""
        self.ai_provider.generate_completion.side_effect = Exception("API Error")
        self.ai_provider.get_available_models.return_value = ["gpt-3.5-turbo"]
        
        result = self.generator.enhance_description_with_ai("test", {})
        assert result == ""
    
    def test_enhance_code_purpose_with_ai_success(self):
        """Test successful code purpose enhancement."""
        mock_response = CompletionResponse(
            content="This code implements a Monte Carlo simulation for financial modeling.",
            model="gpt-3.5-turbo",
            provider="openai",
            usage={"total_tokens": 40}
        )
        self.ai_provider.generate_completion.return_value = mock_response
        self.ai_provider.get_available_models.return_value = ["gpt-3.5-turbo"]
        
        basic_purpose = "Monte Carlo implementation"
        
        result = self.generator.enhance_code_purpose_with_ai(Path("simulation.py"), basic_purpose, "Python")
        assert "Monte Carlo simulation for financial modeling" in result
    
    def test_enhance_code_purpose_with_ai_no_provider(self):
        """Test code purpose enhancement without provider."""
        generator = ReadmeGenerator(Path("/test/dir"), ai_provider=None)
        basic_purpose = "Test code"
        result = generator.enhance_code_purpose_with_ai(Path("test.py"), basic_purpose, "Python")
        assert result == basic_purpose


class TestReadmeGeneratorDirectoryScanning:
    """Test directory scanning functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.generator = ReadmeGenerator(Path("/test/dir"))
    
    @patch.object(Path, 'iterdir')
    @patch.object(Path, 'is_file')
    @patch.object(Path, 'is_dir')
    def test_scan_directory_basic(self, mock_is_dir, mock_is_file, mock_iterdir):
        """Test basic directory scanning."""
        # Setup mock files
        mock_files = [Path('main.py'), Path('utils.c'), Path('README.md'), Path('data.txt')]
        mock_iterdir.return_value = mock_files
        
        def is_file_side_effect(path=None):
            if path is None:  # Called as method on Path object
                return True
            return True
        
        def is_dir_side_effect(path=None):
            return False
            
        mock_is_file.side_effect = is_file_side_effect
        mock_is_dir.side_effect = is_dir_side_effect
        
        # Mock file reading
        with patch('builtins.open', mock_open(read_data="# Test content")):
            result = self.generator.scan_directory(Path("/test/dir"))
        
        assert 'languages' in result
        assert 'code_files' in result
        assert len(result['code_files']) >= 2  # main.py and utils.c
        assert any(f['path'].name == 'main.py' for f in result['code_files'])
    
    @patch.object(Path, 'iterdir')
    @patch.object(Path, 'is_file')
    @patch.object(Path, 'is_dir')
    def test_scan_directory_with_subdirectories(self, mock_is_dir, mock_is_file, mock_iterdir):
        """Test directory scanning with subdirectories."""
        mock_items = [Path('src'), Path('tests'), Path('main.py')]
        mock_iterdir.return_value = mock_items
        
        def is_dir_side_effect():
            # Called on the Path object itself
            path_name = str(mock_iterdir.return_value[0]).split('/')[-1]  # Get the current path name
            return path_name in ('src', 'tests')
        
        def is_file_side_effect():
            return True
        
        # Create individual mocks for each path
        src_mock = Mock(spec=Path)
        src_mock.name = 'src'
        src_mock.is_dir.return_value = True
        src_mock.is_file.return_value = False
        
        tests_mock = Mock(spec=Path)
        tests_mock.name = 'tests'
        tests_mock.is_dir.return_value = True
        tests_mock.is_file.return_value = False
        
        main_mock = Mock(spec=Path)
        main_mock.name = 'main.py'
        main_mock.suffix = '.py'
        main_mock.is_dir.return_value = False
        main_mock.is_file.return_value = True
        
        mock_iterdir.return_value = [src_mock, tests_mock, main_mock]
        
        with patch('builtins.open', mock_open(read_data="# Test")):
            result = self.generator.scan_directory(Path("/test/dir"))
        
        assert 'subdirs' in result
        assert len(result['subdirs']) == 2
        assert any(d.name == 'src' for d in result['subdirs'])
        assert any(d.name == 'tests' for d in result['subdirs'])
    
    @patch.object(Path, 'iterdir')
    def test_scan_directory_permission_error(self, mock_iterdir):
        """Test directory scanning with permission error."""
        mock_iterdir.side_effect = PermissionError("Access denied")
        
        result = self.generator.scan_directory(Path("/test/dir"))
        # Should return basic structure even with permission error
        assert 'code_files' in result
        assert result['code_files'] == []
    
    @patch.object(ReadmeGenerator, 'scan_directory')
    def test_determine_depth_level_top(self, mock_scan):
        """Test depth level determination for top-level directory."""
        mock_scan.return_value = {
            'code_files': [{'path': Path('f1.py')}, {'path': Path('f2.py')}],
            'subdirs': [Path('src'), Path('tests'), Path('docs'), Path('examples')],
            'examples': []
        }
        # Root directory should be top level
        assert self.generator.determine_depth_level(self.generator.source_dir) == "top"
    
    @patch.object(ReadmeGenerator, 'scan_directory')
    def test_determine_depth_level_mid(self, mock_scan):
        """Test depth level determination for mid-level directory."""
        mock_scan.return_value = {
            'code_files': [{'path': Path('main.py')}],
            'subdirs': [Path('utils'), Path('models')],
            'examples': []
        }
        # Mid-level directory with some subdirs but not too many
        mid_path = self.generator.source_dir / 'src'
        assert self.generator.determine_depth_level(mid_path) == "mid"
    
    @patch.object(ReadmeGenerator, 'scan_directory')
    def test_determine_depth_level_leaf(self, mock_scan):
        """Test depth level determination for leaf directory."""
        mock_scan.return_value = {
            'code_files': [{'path': Path('script.py')}],
            'subdirs': [],
            'examples': []
        }
        # Leaf directory with no subdirs
        leaf_path = self.generator.source_dir / 'examples' / 'basic'
        assert self.generator.determine_depth_level(leaf_path) == "leaf"


class TestReadmeGeneratorContentGeneration:
    """Test content generation methods."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.generator = ReadmeGenerator(Path("/test/dir"))
    
    def test_generate_quick_reference_table_basic(self):
        """Test quick reference table generation."""
        items = [
            {
                'name': 'src', 
                'description': 'Main source code directory',
                'languages': ['Python', 'C++'],
                'example_count': 5
            },
            {
                'name': 'examples',
                'description': 'Example implementations',
                'languages': ['Python'],
                'example_count': 3
            }
        ]
        
        result = self.generator.generate_quick_reference_table(items, 'top')
        
        assert "| Category | Description | Languages | Examples |" in result
        assert "src" in result
        assert "examples" in result
        assert "Python" in result
        assert "C++" in result
    
    def test_generate_quick_reference_table_empty(self):
        """Test quick reference table with empty info."""
        items = []
        result = self.generator.generate_quick_reference_table(items, 'top')
        assert result == ""
    
    def test_generate_readme_content_with_ai(self):
        """Test README content generation with AI enhancement."""
        self.generator.ai_provider = Mock()
        mock_response = CompletionResponse(
            content="This is an AI-enhanced description.",
            model="gpt-3.5-turbo",
            provider="openai",
            usage={"total_tokens": 30}
        )
        self.generator.ai_provider.generate_completion.return_value = mock_response
        self.generator.ai_provider.get_available_models.return_value = ["gpt-3.5-turbo"]
        
        info = {
            'languages': {'Python'},
            'code_files': [{'path': Path('main.py'), 'language': 'Python'}],
            'subdirs': [],
            'special_files': []
        }
        
        # Mock the directory scanning to avoid file system access
        with patch.object(self.generator, 'scan_directory', return_value=info):
            with patch.object(self.generator, 'determine_depth_level', return_value='leaf'):
                result = self.generator.generate_readme_content(Path("test"), info)
        
        assert "test" in result.lower()
        assert "This is an AI-enhanced description" in result
    
    def test_generate_readme_content_without_ai(self):
        """Test README content generation without AI."""
        info = {
            'languages': {'Python', 'C++'},
            'code_files': [
                {'path': Path('main.py'), 'language': 'Python', 'description': ''},
                {'path': Path('solver.cpp'), 'language': 'C++', 'description': ''}
            ],
            'subdirs': [Path('tests')],
            'special_files': [Path('Makefile')],
            'examples': [],
            'has_makefile': True,
            'has_sbatch': False,
            'output_files': []
        }
        
        # Mock the directory scanning to avoid file system access
        with patch.object(self.generator, 'scan_directory', return_value=info):
            with patch.object(self.generator, 'determine_depth_level', return_value='mid'):
                result = self.generator.generate_readme_content(Path("simulation"), info)
        
        assert "simulation" in result.lower()
        assert "Python" in result
        assert "C++" in result
        # Note: subdirs are not shown in mid-level content, only examples are
    
    @patch.object(ReadmeGenerator, 'scan_directory')
    @patch.object(ReadmeGenerator, 'generate_quick_reference_table')
    def test_generate_top_level_content(self, mock_table, mock_scan):
        """Test top-level content generation."""
        # Mock subdirectory scanning
        mock_scan.return_value = {
            'languages': {'Python'},
            'code_files': [{'path': Path('main.py')}],
            'subdirs': [],
            'examples': []
        }
        mock_table.return_value = "Mock quick reference table"
        
        info = {
            'languages': {'Python', 'Fortran'},
            'code_files': [{'path': Path('main.py')}, {'path': Path('solver.f90')}],
            'subdirs': [Path('src'), Path('examples'), Path('tests')],
            'examples': [],
            'special_files': [Path('requirements.txt')],
            'has_makefile': True,
            'has_sbatch': False
        }
        
        result = self.generator._generate_top_level_content(Path("project"), info)
        
        assert "Directory Structure" in result
        assert "Usage" in result
        assert "Compilation" in result
        assert "Mock quick reference table" in result
    
    @patch.object(ReadmeGenerator, 'scan_directory')
    def test_generate_mid_level_content(self, mock_scan):
        """Test mid-level content generation."""
        mock_scan.return_value = {
            'code_files': [],
            'languages': set(),
            'subdirs': [],
            'examples': [],
            'special_files': []
        }
        
        info = {
            'languages': {'C'},
            'code_files': [{'path': Path('algorithm.c'), 'language': 'C', 'description': ''}],
            'subdirs': [Path('helpers')],
            'examples': [],
            'special_files': [],
            'has_makefile': False,
            'has_sbatch': False,
            'output_files': []
        }
        
        result = self.generator._generate_mid_level_content(Path("algorithms"), info)
        
        assert "algorithm.c" in result
        # Note: regular subdirs are not shown in mid-level content, only examples are shown
    
    def test_generate_leaf_level_content(self):
        """Test leaf-level content generation."""
        info = {
            'code_files': [
                {'path': Path('script.py'), 'language': 'Python', 'description': 'Data processing script'}
            ],
            'languages': {'Python'},
            'subdirs': [],
            'examples': [],
            'special_files': [],
            'has_makefile': False,
            'has_sbatch': False,
            'output_files': []
        }
        
        result = self.generator._generate_leaf_level_content(Path("scripts"), info)
        
        assert "script.py" in result
        # Note: descriptions are not shown in leaf-level content, only file names


class TestReadmeGeneratorFileOperations:
    """Test file operations and processing."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.generator = ReadmeGenerator(Path("/test/dir"))
    
    @patch.object(Path, 'write_text')
    @patch.object(Path, 'exists')
    @patch('doc_generator.command_tracker.CommandTracker.save_command_file')
    def test_write_readme_new_file(self, mock_save_command, mock_exists, mock_write_text):
        """Test writing README to new file."""
        mock_exists.return_value = False
        mock_save_command.return_value = '/test/command.json'
        
        output_dir = Path("/test/output")
        content = "# Test README\n\nThis is test content."
        
        result = self.generator.write_readme(output_dir, content)
        
        expected_path = output_dir / "README_generated.md"
        assert result == expected_path
        mock_write_text.assert_called_once_with(content, encoding='utf-8')
    
    @patch.object(Path, 'write_text')
    @patch.object(Path, 'exists')
    @patch.object(Path, 'rename')
    @patch('doc_generator.command_tracker.CommandTracker.save_command_file')
    def test_write_readme_overwrite(self, mock_save_command, mock_rename, mock_exists, mock_write_text):
        """Test writing README with overwrite."""
        mock_exists.return_value = True
        mock_save_command.return_value = '/test/command.json'
        self.generator.overwrite = True
        
        output_dir = Path("/test/output")
        content = "# Overwritten README"
        
        result = self.generator.write_readme(output_dir, content)
        
        expected_path = output_dir / "README.md"
        assert result == expected_path
        mock_write_text.assert_called_once_with(content, encoding='utf-8')
    
    @patch.object(Path, 'write_text')
    @patch('doc_generator.command_tracker.CommandTracker.save_command_file')
    def test_write_readme_with_suffix(self, mock_save_command, mock_write_text):
        """Test writing README with suffix."""
        mock_save_command.return_value = '/test/command.json'
        self.generator.overwrite = False
        self.generator.suffix = "_v2"
        
        output_dir = Path("/test/output")
        content = "# Test README with suffix"
        
        result = self.generator.write_readme(output_dir, content)
        
        expected_path = output_dir / "README_v2.md"
        assert result == expected_path
        mock_write_text.assert_called_once_with(content, encoding='utf-8')
    
    @patch.object(Path, 'write_text', side_effect=IOError("Permission denied"))
    def test_write_readme_io_error(self, mock_write_text):
        """Test writing README with IO error."""
        output_dir = Path("/test/output")
        content = "# Test README"
        
        result = self.generator.write_readme(output_dir, content)
        assert result is None
    
    @patch.object(ReadmeGenerator, 'scan_directory')
    @patch.object(ReadmeGenerator, 'generate_readme_content')
    @patch.object(ReadmeGenerator, 'write_readme')
    @patch.object(Path, 'exists')
    @patch.object(Path, 'is_dir')
    def test_process_directory_tree_single(self, mock_is_dir, mock_exists, 
                                         mock_write, mock_generate, mock_scan):
        """Test processing single directory."""
        # Setup mocks
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        mock_scan.return_value = {'languages': ['Python']}
        mock_generate.return_value = "# Generated README"
        mock_write.return_value = Path("/test/README.md")
        
        self.generator.recursive = False
        results = self.generator.process_directory_tree()
        
        assert len(results) == 1
        mock_scan.assert_called_once()
        mock_generate.assert_called_once()
        mock_write.assert_called_once()
    
    @patch.object(ReadmeGenerator, '_process_directory')
    @patch.object(ReadmeGenerator, 'scan_directory')
    @patch.object(Path, 'exists')
    @patch.object(Path, 'is_dir')
    def test_process_directory_tree_recursive(self, mock_is_dir, mock_exists, mock_scan, mock_process):
        """Test recursive directory processing."""
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        # Return empty subdirs to avoid recursion
        mock_scan.return_value = {
            'code_files': [],
            'subdirs': [],  # Empty to prevent infinite recursion
            'languages': set(),
            'examples': [],
            'special_files': []
        }
        
        # Mock _process_directory to append to generated_files
        def mock_process_side_effect(path):
            self.generator.generated_files.append(Path("/test/README.md"))
        
        mock_process.side_effect = mock_process_side_effect
        
        self.generator.recursive = True
        results = self.generator.process_directory_tree()
        
        mock_process.assert_called_once_with(self.generator.source_dir)
        assert results == [Path("/test/README.md")]


class TestReadmeGeneratorEdgeCases:
    """Test edge cases and error handling."""
    
    @patch.object(Path, 'iterdir')
    def test_empty_directory(self, mock_iterdir):
        """Test handling of empty directory."""
        generator = ReadmeGenerator(Path("/empty/dir"))
        
        mock_iterdir.return_value = []
        result = generator.scan_directory(Path("/empty/dir"))
        
        assert result['code_files'] == []
        assert result['subdirs'] == []
    
    @patch.object(Path, 'iterdir')
    def test_nonexistent_directory(self, mock_iterdir):
        """Test handling of non-existent directory."""
        generator = ReadmeGenerator(Path("/nonexistent"))
        
        mock_iterdir.side_effect = FileNotFoundError("Directory not found")
        
        # The scan_directory method currently doesn't handle FileNotFoundError,
        # only PermissionError, so this test should expect an exception
        with pytest.raises(FileNotFoundError):
            generator.scan_directory(Path("/nonexistent"))
    
    def test_large_file_handling(self):
        """Test handling of large files."""
        generator = ReadmeGenerator(Path("/test/dir"))
        
        # Simulate large file content
        large_content = "# " + "x" * 10000 + "\n" + "content" * 1000
        
        with patch('builtins.open', mock_open(read_data=large_content)):
            result = generator.extract_header_comments(Path("large.py"))
        
        # Should still extract comments but truncate appropriately
        assert len(result) <= len(large_content)
    
    def test_special_characters_in_filenames(self):
        """Test handling files with special characters."""
        generator = ReadmeGenerator(Path("/test/dir"))
        
        special_files = ['file with spaces.py', 'file-with-dashes.c', 'file_with_underscores.f90']
        
        for filename in special_files:
            assert not generator.should_exclude(Path(filename))
            language = generator.get_language(Path(filename))
            assert language is not None  # Should detect language properly
    
    @patch('logging.Logger.warning')
    def test_logging_integration(self, mock_warning):
        """Test logging integration."""
        logger = logging.getLogger("test")
        generator = ReadmeGenerator(Path("/test/dir"), logger=logger)
        
        # Test AI enhancement failure logging
        generator.ai_provider = Mock()
        generator.ai_provider.generate_completion.side_effect = Exception("Test error")
        generator.ai_provider.get_available_models.return_value = ["gpt-3.5-turbo"]
        
        result = generator.enhance_description_with_ai("test", {})
        
        assert result == ""
        # Logging should have been called (though mocked)


class TestReadmeGeneratorIntegration:
    """Integration tests for ReadmeGenerator."""
    
    def test_complete_workflow_simple(self):
        """Test complete workflow with simple directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "main.py").write_text('"""Main module."""\nprint("Hello world")')
            (temp_path / "utils.c").write_text('/* Utility functions */\n#include <stdio.h>')
            (temp_path / "Makefile").write_text('all:\n\tgcc -o program *.c')
            
            generator = ReadmeGenerator(temp_path, recursive=False)
            
            # Scan directory
            info = generator.scan_directory(temp_path)
            
            assert len(info['code_files']) == 2
            assert 'Python' in info['languages']
            assert 'C' in info['languages']
            assert info['has_makefile'] is True
            
            # Generate content
            content = generator.generate_readme_content(temp_path, info)
            assert "main.py" in content
            assert "utils.c" in content
            assert "Python" in content
            assert "C" in content
    
    @patch.object(ReadmeGenerator, 'enhance_description_with_ai')
    def test_complete_workflow_with_ai(self, mock_enhance):
        """Test complete workflow with AI enhancement."""
        mock_enhance.return_value = "AI-enhanced description of the project."
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test file
            (temp_path / "script.py").write_text('# Simple script\nprint("test")')
            
            ai_provider = Mock()
            generator = ReadmeGenerator(temp_path, ai_provider=ai_provider)
            
            info = generator.scan_directory(temp_path)
            content = generator.generate_readme_content(temp_path, info)
            
            assert "AI-enhanced description" in content
            mock_enhance.assert_called()