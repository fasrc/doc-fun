"""
Code Example Scanner Module

Provides code discovery and terminology management functionality.
Extracted from core.py following single responsibility principle.
"""

import os
import re
import yaml
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass

from .config import get_settings
from .cache import cached
from .exceptions import DocGeneratorError, FileOperationError
from .utils import safe_file_read


@dataclass
class CodeFileInfo:
    """Information about a discovered code file."""
    path: str
    language: str
    description: str
    file_hash: str
    size: int
    modified_time: float


@dataclass
class ScanResults:
    """Results of code scanning operation."""
    files_found: List[CodeFileInfo]
    directories_scanned: int
    total_size: int
    scan_duration: float
    languages_detected: Set[str]


class CodeExampleScanner:
    """
    Scans directories for code examples and manages terminology.
    
    Discovers code files, analyzes their content, and maintains
    terminology databases for documentation generation.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize code scanner.
        
        Args:
            logger: Optional logger instance
        """
        self.settings = get_settings()
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize pygments support if available
        self._setup_language_detection()
        
        # Language extension mappings
        self.language_extensions = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.sh': 'bash',
            '.r': 'r',
            '.m': 'matlab',
            '.sql': 'sql',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.md': 'markdown',
            '.tex': 'latex'
        }
        
        # Comment patterns for description extraction
        self.comment_patterns = {
            'python': [r'#\s*(.+)', r'"""(.+?)"""', r"'''(.+?)'''"],
            'javascript': [r'//\s*(.+)', r'/\*(.+?)\*/'],
            'java': [r'//\s*(.+)', r'/\*(.+?)\*/'],
            'cpp': [r'//\s*(.+)', r'/\*(.+?)\*/'],
            'c': [r'//\s*(.+)', r'/\*(.+?)\*/'],
            'bash': [r'#\s*(.+)'],
            'r': [r'#\s*(.+)'],
            'sql': [r'--\s*(.+)', r'/\*(.+?)\*/'],
            'html': [r'<!--(.+?)-->'],
            'css': [r'/\*(.+?)\*/']
        }
        
        self.logger.info("CodeExampleScanner initialized")
    
    def _setup_language_detection(self) -> None:
        """Setup pygments language detection if available."""
        try:
            from pygments.lexers import get_lexer_for_filename
            from pygments.util import ClassNotFound
            
            self.get_lexer_for_filename = get_lexer_for_filename
            self.ClassNotFound = ClassNotFound
            self.has_pygments = True
            
            self.logger.debug("Pygments available for language detection")
            
        except ImportError:
            self.get_lexer_for_filename = None
            self.ClassNotFound = None
            self.has_pygments = False
            
            self.logger.info("Pygments not available, using basic language detection")
    
    def _detect_language(self, file_path: Path) -> str:
        """
        Detect programming language from file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected language name
        """
        # Try pygments first if available
        if self.has_pygments and self.get_lexer_for_filename:
            try:
                lexer = self.get_lexer_for_filename(str(file_path))
                return lexer.name.lower()
            except self.ClassNotFound:
                pass
            except Exception as e:
                self.logger.warning(f"Pygments detection failed for {file_path}: {e}")
        
        # Fall back to extension-based detection
        extension = file_path.suffix.lower()
        return self.language_extensions.get(extension, 'text')
    
    def _detect_comment_style(self, language: str) -> List[str]:
        """
        Get comment patterns for a programming language.
        
        Args:
            language: Programming language name
            
        Returns:
            List of regex patterns for comments
        """
        return self.comment_patterns.get(language.lower(), [r'#\s*(.+)'])
    
    def _extract_description(self, content: str, language: str) -> str:
        """
        Extract description from file content using comment patterns.
        
        Args:
            content: File content
            language: Programming language
            
        Returns:
            Extracted description or empty string
        """
        patterns = self._detect_comment_style(language)
        descriptions = []
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            descriptions.extend(match.strip() for match in matches if match.strip())
        
        # Return first substantial description (>= 10 chars)
        for desc in descriptions:
            if len(desc) >= 10:
                return desc[:200]  # Limit length
        
        # Fall back to first line if no good description
        first_line = content.split('\n')[0].strip()
        if first_line and not first_line.startswith(('#!', '<?')):
            return first_line[:100]
        
        return ""
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate MD5 hash of file content.
        
        Args:
            file_path: Path to file
            
        Returns:
            MD5 hash string
        """
        content = safe_file_read(file_path)
        if content is None:
            return ""
        
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _find_slurm_files(self, directory: Path) -> List[Path]:
        """
        Find SLURM job script files in directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of SLURM script paths
        """
        slurm_files = []
        
        # Common SLURM file patterns
        slurm_patterns = ['*.sbatch', '*.slurm', '*.sub', '*job*', '*submit*']
        
        for pattern in slurm_patterns:
            slurm_files.extend(directory.glob(pattern))
        
        # Also check files with SLURM directives
        for file_path in directory.glob('*'):
            if file_path.is_file():
                try:
                    content = safe_file_read(file_path)
                    if content and re.search(r'#SBATCH|#!/bin/bash.*SLURM', content, re.MULTILINE):
                        slurm_files.append(file_path)
                except Exception:
                    continue
        
        return list(set(slurm_files))  # Remove duplicates
    
    def _extract_file_info(self, file_path: Path) -> Optional[CodeFileInfo]:
        """
        Extract information from a code file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            CodeFileInfo object or None if extraction fails
        """
        try:
            stat = file_path.stat()
            content = safe_file_read(file_path)
            
            if content is None:
                return None
            
            language = self._detect_language(file_path)
            description = self._extract_description(content, language)
            file_hash = self._calculate_file_hash(file_path)
            
            return CodeFileInfo(
                path=str(file_path),
                language=language,
                description=description,
                file_hash=file_hash,
                size=stat.st_size,
                modified_time=stat.st_mtime
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to extract info from {file_path}: {e}")
            return None
    
    @cached(ttl=3600)  # Cache for 1 hour
    def scan_directory(
        self,
        directory: str,
        recursive: bool = True,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> ScanResults:
        """
        Scan directory for code files.
        
        Args:
            directory: Directory path to scan
            recursive: Whether to scan recursively
            include_patterns: File patterns to include
            exclude_patterns: File patterns to exclude
            
        Returns:
            ScanResults with discovered files
            
        Raises:
            DocGeneratorError: If scanning fails
        """
        import time
        start_time = time.time()
        
        directory_path = Path(directory)
        if not directory_path.exists():
            raise DocGeneratorError(
                f"Directory not found: {directory}",
                error_code="DIRECTORY_NOT_FOUND",
                context={'directory': str(directory)}
            )
        
        if not directory_path.is_dir():
            raise DocGeneratorError(
                f"Path is not a directory: {directory}",
                error_code="NOT_A_DIRECTORY",
                context={'directory': str(directory)}
            )
        
        # Set default patterns
        include_patterns = include_patterns or ['*']
        exclude_patterns = exclude_patterns or [
            '.*', '__pycache__', '*.pyc', '*.pyo', '*.class',
            'node_modules', '.git', '.svn', '.hg', 'dist', 'build'
        ]
        
        files_found = []
        directories_scanned = 0
        total_size = 0
        languages_detected = set()
        
        try:
            # Determine scanning method
            if recursive:
                file_iterator = directory_path.rglob('*')
            else:
                file_iterator = directory_path.iterdir()
            
            for path in file_iterator:
                if path.is_dir():
                    directories_scanned += 1
                    continue
                
                # Apply exclude patterns
                if any(path.match(pattern) for pattern in exclude_patterns):
                    continue
                
                # Apply include patterns
                if not any(path.match(pattern) for pattern in include_patterns):
                    continue
                
                # Extract file information
                file_info = self._extract_file_info(path)
                if file_info:
                    files_found.append(file_info)
                    total_size += file_info.size
                    languages_detected.add(file_info.language)
            
            scan_duration = time.time() - start_time
            
            self.logger.info(
                f"Scanned {directories_scanned} directories, "
                f"found {len(files_found)} files in {scan_duration:.2f}s"
            )
            
            return ScanResults(
                files_found=files_found,
                directories_scanned=directories_scanned,
                total_size=total_size,
                scan_duration=scan_duration,
                languages_detected=languages_detected
            )
            
        except Exception as e:
            raise DocGeneratorError(
                f"Directory scan failed: {e}",
                error_code="SCAN_ERROR",
                context={'directory': str(directory)}
            )
    
    def update_terminology_file(
        self,
        scan_results: ScanResults,
        terminology_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update terminology file based on scan results.
        
        Args:
            scan_results: Results from directory scan
            terminology_path: Path to terminology file
            
        Returns:
            Updated terminology dictionary
            
        Raises:
            DocGeneratorError: If update fails
        """
        if not terminology_path:
            terminology_path = self.settings.paths.terminology_path
        else:
            terminology_path = Path(terminology_path)
        
        try:
            # Load existing terminology
            if terminology_path.exists():
                with open(terminology_path, 'r', encoding='utf-8') as f:
                    terminology = yaml.safe_load(f) or {}
            else:
                terminology = {}
            
            # Initialize sections if not present
            if 'code_examples' not in terminology:
                terminology['code_examples'] = {}
            
            if 'languages' not in terminology:
                terminology['languages'] = {}
            
            # Update with scan results
            for file_info in scan_results.files_found:
                # Add file to code examples
                terminology['code_examples'][file_info.path] = {
                    'language': file_info.language,
                    'description': file_info.description,
                    'size': file_info.size,
                    'hash': file_info.file_hash,
                    'last_modified': file_info.modified_time
                }
                
                # Update language statistics
                if file_info.language not in terminology['languages']:
                    terminology['languages'][file_info.language] = {
                        'count': 0,
                        'total_size': 0,
                        'examples': []
                    }
                
                lang_info = terminology['languages'][file_info.language]
                lang_info['count'] += 1
                lang_info['total_size'] += file_info.size
                
                # Add example if description is good
                if (file_info.description and 
                    len(file_info.description) > 20 and
                    file_info.path not in lang_info['examples']):
                    lang_info['examples'].append(file_info.path)
                    # Keep only top 10 examples per language
                    lang_info['examples'] = lang_info['examples'][:10]
            
            # Add scan metadata
            terminology['scan_metadata'] = {
                'last_scan': time.time(),
                'directories_scanned': scan_results.directories_scanned,
                'files_found': len(scan_results.files_found),
                'total_size': scan_results.total_size,
                'languages_detected': list(scan_results.languages_detected),
                'scan_duration': scan_results.scan_duration
            }
            
            # Save updated terminology
            terminology_path.parent.mkdir(parents=True, exist_ok=True)
            with open(terminology_path, 'w', encoding='utf-8') as f:
                yaml.dump(terminology, f, default_flow_style=False, sort_keys=True)
            
            self.logger.info(
                f"Updated terminology with {len(scan_results.files_found)} files, "
                f"saved to {terminology_path}"
            )
            
            return terminology
            
        except Exception as e:
            raise DocGeneratorError(
                f"Failed to update terminology: {e}",
                error_code="TERMINOLOGY_UPDATE_ERROR",
                context={'terminology_path': str(terminology_path)}
            )
    
    def check_for_updates(
        self,
        directory: str,
        terminology_path: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if directory has changes since last scan.
        
        Args:
            directory: Directory to check
            terminology_path: Path to terminology file
            
        Returns:
            Tuple of (has_changes, list_of_changes)
        """
        if not terminology_path:
            terminology_path = self.settings.paths.terminology_path
        else:
            terminology_path = Path(terminology_path)
        
        if not terminology_path.exists():
            return True, ["No previous scan found"]
        
        try:
            # Load existing terminology
            with open(terminology_path, 'r', encoding='utf-8') as f:
                terminology = yaml.safe_load(f) or {}
            
            if 'code_examples' not in terminology:
                return True, ["No code examples in terminology"]
            
            changes = []
            directory_path = Path(directory)
            
            # Check for new/modified files
            for file_path in directory_path.rglob('*'):
                if not file_path.is_file():
                    continue
                
                str_path = str(file_path)
                
                if str_path not in terminology['code_examples']:
                    changes.append(f"New file: {str_path}")
                    continue
                
                # Check if file was modified
                file_info = terminology['code_examples'][str_path]
                try:
                    current_hash = self._calculate_file_hash(file_path)
                    if current_hash != file_info.get('hash', ''):
                        changes.append(f"Modified: {str_path}")
                except Exception:
                    changes.append(f"Cannot read: {str_path}")
            
            # Check for deleted files
            for example_path in terminology['code_examples']:
                if not Path(example_path).exists():
                    changes.append(f"Deleted: {example_path}")
            
            return len(changes) > 0, changes
            
        except Exception as e:
            self.logger.warning(f"Failed to check for updates: {e}")
            return True, [f"Error checking updates: {e}"]