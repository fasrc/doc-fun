"""
Module for generating README documentation from code directory structures.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict


class ReadmeGenerator:
    """
    Generates README.md files from code examples and directory structures.
    
    This class scans directory structures, analyzes code files, and generates
    comprehensive README documentation with depth-adaptive templates and
    Quick Reference tables.
    """
    
    # File extensions for different languages
    CODE_EXTENSIONS = {
        '.py': 'Python',
        '.f90': 'Fortran',
        '.f': 'Fortran',
        '.c': 'C',
        '.cpp': 'C++',
        '.h': 'C',
        '.hpp': 'C++',
        '.R': 'R',
        '.r': 'R',
        '.m': 'MATLAB',
        '.jl': 'Julia',
        '.sh': 'Bash',
        '.do': 'Stata'
    }
    
    # Files to exclude from scanning
    EXCLUDE_PATTERNS = {
        '*.o', '*.x', '*.pyc', '__pycache__', '.git', '.gitignore',
        '*.so', '*.a', '*.mod', '*.exe', 'build', 'dist',
        '.DS_Store', 'Thumbs.db', '*.swp', '*.bak', '~*'
    }
    
    # Special files to look for
    SPECIAL_FILES = {
        'Makefile', 'makefile', 'run.sbatch', '*.sbatch', '*.slurm',
        'requirements.txt', 'environment.yml', 'setup.py', 'pyproject.toml'
    }
    
    def __init__(self, 
                 source_dir: Path,
                 recursive: bool = True,
                 overwrite: bool = False,
                 suffix: str = "_generated",
                 ai_provider = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the README generator.
        
        Args:
            source_dir: Root directory to scan
            recursive: Whether to recursively generate READMEs for subdirectories
            overwrite: Whether to overwrite existing README.md files
            suffix: Suffix for generated files (ignored if overwrite=True)
            ai_provider: AI provider for enhanced descriptions (optional)
            logger: Logger instance for output
        """
        self.source_dir = Path(source_dir).resolve()
        self.recursive = recursive
        self.overwrite = overwrite
        self.suffix = suffix if not overwrite else ""
        self.ai_provider = ai_provider
        self.logger = logger or logging.getLogger(__name__)
        
        # Track generated files
        self.generated_files: List[Path] = []
        
        # Cache for directory analysis
        self.directory_cache: Dict[Path, Dict] = {}
        
        # AI enhancement cache to avoid duplicate API calls
        self.ai_cache: Dict[str, str] = {}
    
    def should_exclude(self, path: Path) -> bool:
        """
        Check if a file or directory should be excluded from scanning.
        
        Args:
            path: Path to check
            
        Returns:
            True if should be excluded, False otherwise
        """
        name = path.name
        
        # Check exact matches
        if name in {'.git', '__pycache__', 'build', 'dist', '.DS_Store', 'Thumbs.db'}:
            return True
        
        # Check patterns
        if name.endswith(('.o', '.x', '.pyc', '.so', '.a', '.mod', '.exe', '.swp', '.bak')):
            return True
        
        if name.startswith(('~', '.')):
            return True
            
        return False
    
    def get_language(self, file_path: Path) -> Optional[str]:
        """
        Determine the programming language of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Language name or None if not recognized
        """
        suffix = file_path.suffix.lower()
        return self.CODE_EXTENSIONS.get(suffix)
    
    def extract_header_comments(self, file_path: Path, lines: int = 10) -> str:
        """
        Extract header comments from a code file.
        
        Args:
            file_path: Path to the code file
            lines: Number of lines to read
            
        Returns:
            Extracted comments or description
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.readlines()[:lines]
            
            comments = []
            for line in content:
                line = line.strip()
                # Look for common comment patterns
                if line.startswith(('#', '//', '!', '--', '%', '*')):
                    # Remove comment markers
                    cleaned = re.sub(r'^[#!/*%-]+\s*', '', line)
                    if cleaned and not cleaned.startswith('='):
                        comments.append(cleaned)
                elif line.startswith('"""') or line.startswith("'''"):
                    # Python docstring
                    cleaned = line.strip('"\' ')
                    if cleaned:
                        comments.append(cleaned)
            
            # Return first meaningful comment
            for comment in comments:
                if len(comment) > 10 and not comment.lower().startswith(('author', 'date', 'version')):
                    return comment[:200]  # Limit length
                    
        except Exception as e:
            self.logger.debug(f"Could not read {file_path}: {e}")
        
        return ""
    
    def enhance_description_with_ai(self, directory_name: str, basic_info: Dict) -> str:
        """
        Use AI to enhance directory descriptions.
        
        Args:
            directory_name: Name of the directory
            basic_info: Basic information about the directory (languages, files, etc.)
            
        Returns:
            Enhanced description string
        """
        if not self.ai_provider:
            return ""
        
        # Create cache key
        cache_key = f"{directory_name}:{str(sorted(basic_info.items()))}"
        if cache_key in self.ai_cache:
            return self.ai_cache[cache_key]
        
        # Build context for AI
        context = f"Directory: {directory_name}\n"
        if basic_info.get('languages'):
            context += f"Languages: {', '.join(basic_info['languages'])}\n"
        if basic_info.get('code_files'):
            context += f"Code files: {len(basic_info['code_files'])}\n"
        if basic_info.get('examples'):
            context += f"Example subdirectories: {len(basic_info['examples'])}\n"
        if basic_info.get('has_makefile'):
            context += "Has compilation (Makefile)\n"
        if basic_info.get('has_sbatch'):
            context += "Has job scripts (SLURM)\n"
        
        # Add sample code comments if available
        comments = []
        for file_info in basic_info.get('code_files', [])[:3]:
            if file_info.get('description'):
                comments.append(f"- {file_info['path'].name}: {file_info['description']}")
        if comments:
            context += f"Code descriptions:\n" + "\n".join(comments) + "\n"
        
        prompt = f"""Analyze this code directory structure and provide a brief, informative description (2-3 sentences max).

{context}

Focus on:
- What type of computational problem this directory addresses
- The parallel computing paradigm or approach used
- Key technical features or capabilities

Provide only the description, no preamble or explanation."""
        
        try:
            from .providers.base import CompletionRequest
            
            # Get a suitable model for this provider
            available_models = self.ai_provider.get_available_models()
            model = available_models[0] if available_models else "gpt-3.5-turbo"
            
            request = CompletionRequest(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.3,
                max_tokens=200
            )
            
            response = self.ai_provider.generate_completion(request)
            description = response.content.strip()
            
            # Cache the result
            self.ai_cache[cache_key] = description
            self.logger.debug(f"AI enhanced description for {directory_name}: {description[:100]}...")
            return description
            
        except Exception as e:
            self.logger.warning(f"AI enhancement failed for {directory_name}: {e}")
            return ""
    
    def enhance_code_purpose_with_ai(self, file_path: Path, basic_purpose: str, language: str) -> str:
        """
        Use AI to enhance code file purpose descriptions.
        
        Args:
            file_path: Path to the code file
            basic_purpose: Basic purpose extracted from comments
            language: Programming language
            
        Returns:
            Enhanced purpose description
        """
        if not self.ai_provider or not basic_purpose:
            return basic_purpose
        
        cache_key = f"code:{file_path.name}:{basic_purpose[:50]}"
        if cache_key in self.ai_cache:
            return self.ai_cache[cache_key]
        
        prompt = f"""Analyze this {language} code file and provide a clear, concise purpose description (1 sentence).

File: {file_path.name}
Language: {language}
Code comment: {basic_purpose}

Make the description more informative and technically precise while keeping it brief."""
        
        try:
            from .providers.base import CompletionRequest
            
            # Get a suitable model for this provider
            available_models = self.ai_provider.get_available_models()
            model = available_models[0] if available_models else "gpt-3.5-turbo"
            
            request = CompletionRequest(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.2,
                max_tokens=100
            )
            
            response = self.ai_provider.generate_completion(request)
            enhanced = response.content.strip()
            self.ai_cache[cache_key] = enhanced
            return enhanced
            
        except Exception as e:
            self.logger.debug(f"AI enhancement failed for {file_path.name}: {e}")
            return basic_purpose
    
    def scan_directory(self, path: Path) -> Dict:
        """
        Scan a directory and analyze its contents.
        
        Args:
            path: Directory path to scan
            
        Returns:
            Dictionary with directory analysis
        """
        if path in self.directory_cache:
            return self.directory_cache[path]
        
        analysis = {
            'path': path,
            'subdirs': [],
            'code_files': [],
            'special_files': [],
            'output_files': [],
            'languages': set(),
            'has_makefile': False,
            'has_sbatch': False,
            'examples': []
        }
        
        try:
            for item in path.iterdir():
                if self.should_exclude(item):
                    continue
                
                if item.is_dir():
                    # Check if it's an example directory
                    if item.name.lower().startswith('example'):
                        analysis['examples'].append(item)
                    else:
                        analysis['subdirs'].append(item)
                        
                elif item.is_file():
                    # Categorize files
                    if item.name in ('Makefile', 'makefile'):
                        analysis['has_makefile'] = True
                        analysis['special_files'].append(item)
                    elif item.suffix in ('.sbatch', '.slurm'):
                        analysis['has_sbatch'] = True
                        analysis['special_files'].append(item)
                    elif item.suffix in ('.out', '.err', '.log'):
                        analysis['output_files'].append(item)
                    else:
                        lang = self.get_language(item)
                        if lang:
                            analysis['code_files'].append({
                                'path': item,
                                'language': lang,
                                'description': self.extract_header_comments(item)
                            })
                            analysis['languages'].add(lang)
                            
        except PermissionError:
            self.logger.warning(f"Permission denied accessing {path}")
        
        self.directory_cache[path] = analysis
        return analysis
    
    def determine_depth_level(self, path: Path) -> str:
        """
        Determine the depth level of a directory.
        
        Args:
            path: Directory path
            
        Returns:
            'top', 'mid', or 'leaf'
        """
        # Calculate relative depth from source_dir
        try:
            relative = path.relative_to(self.source_dir)
            depth = len(relative.parts)
        except ValueError:
            depth = 0
        
        analysis = self.scan_directory(path)
        
        # Leaf level: directories with only code files, no subdirs
        if not analysis['subdirs'] and not analysis['examples']:
            return 'leaf'
        
        # Top level: root directory or major category directories
        if depth == 0 or (depth == 1 and len(analysis['subdirs']) > 3):
            return 'top'
        
        # Mid level: everything else
        return 'mid'
    
    def generate_quick_reference_table(self, items: List[Dict], level: str) -> str:
        """
        Generate a Quick Reference table appropriate for the depth level.
        
        Args:
            items: List of items to include in table
            level: Depth level ('top', 'mid', 'leaf')
            
        Returns:
            Markdown table string
        """
        if not items:
            return ""
        
        table = "\n## Quick Reference\n\n"
        
        if level == 'top':
            # Top-level overview table
            table += "| Category | Description | Languages | Examples |\n"
            table += "|----------|-------------|-----------|----------|\n"
            
            for item in items:
                category = item.get('name', 'Unknown')
                description = item.get('description', '')[:50]
                languages = ', '.join(item.get('languages', []))[:30]
                examples = item.get('example_count', 0)
                table += f"| **{category}** | {description} | {languages} | {examples} |\n"
                
        elif level == 'mid':
            # Mid-level details table
            table += "| Example | Purpose | Language | Key Files |\n"
            table += "|---------|---------|----------|----------|\n"
            
            for item in items:
                name = item.get('name', 'Unknown')
                purpose = item.get('purpose', '')[:40]
                language = item.get('language', 'Mixed')
                files = item.get('key_files', [])
                files_str = ', '.join(files[:3])[:30]
                table += f"| **{name}** | {purpose} | {language} | {files_str} |\n"
        
        return table + "\n"
    
    def generate_readme_content(self, path: Path, analysis: Dict) -> str:
        """
        Generate README content using depth-adaptive templates.
        
        Args:
            path: Directory path
            analysis: Directory analysis from scan_directory
            
        Returns:
            Generated README content as markdown
        """
        level = self.determine_depth_level(path)
        
        # Skip leaf directories if they're just example directories
        if level == 'leaf' and path.name.lower().startswith('example'):
            self.logger.debug(f"Skipping leaf example directory: {path}")
            return ""
        
        # Start with title
        title = path.name.replace('_', ' ').title()
        content = f"# {title}\n\n"
        
        # Add AI-enhanced description or basic description
        ai_description = self.enhance_description_with_ai(path.name, analysis)
        if ai_description:
            content += f"{ai_description}\n\n"
        elif analysis['code_files']:
            langs = list(analysis['languages'])
            if len(langs) == 1:
                content += f"This directory contains {langs[0]} code examples.\n\n"
            else:
                content += f"This directory contains code examples in {', '.join(langs)}.\n\n"
        
        if level == 'top':
            # Rich content for top-level
            content += self._generate_top_level_content(path, analysis)
        elif level == 'mid':
            # Moderate content for mid-level
            content += self._generate_mid_level_content(path, analysis)
        else:
            # Minimal content for leaf level (if we generate at all)
            content += self._generate_leaf_level_content(path, analysis)
        
        return content
    
    def _generate_top_level_content(self, path: Path, analysis: Dict) -> str:
        """Generate content for top-level directories."""
        content = ""
        
        # Add subdirectory overview
        if analysis['subdirs'] or analysis['examples']:
            content += "## Directory Structure\n\n"
            
            # Group subdirectories
            for subdir in sorted(analysis['subdirs']):
                sub_analysis = self.scan_directory(subdir)
                subdir_name = subdir.name
                
                # Create description from contents
                desc = ""
                if sub_analysis['languages']:
                    desc = f"{', '.join(sub_analysis['languages'])} examples"
                elif sub_analysis['subdirs']:
                    desc = f"{len(sub_analysis['subdirs'])} subdirectories"
                
                content += f"- **[{subdir_name}/]({subdir_name}/)**: {desc}\n"
            
            # Add examples if present
            if analysis['examples']:
                content += "\n### Examples\n\n"
                for example_dir in sorted(analysis['examples']):
                    example_analysis = self.scan_directory(example_dir)
                    example_name = example_dir.name
                    
                    # Try to get purpose from code comments or AI enhancement
                    purpose = "Code example"
                    if example_analysis['code_files']:
                        first_file = example_analysis['code_files'][0]
                        if first_file['description']:
                            purpose = self.enhance_code_purpose_with_ai(
                                first_file['path'], 
                                first_file['description'],
                                first_file['language']
                            ) or first_file['description']
                    
                    content += f"- **[{example_name}/]({example_name}/)**: {purpose}\n"
            
            content += "\n"
        
        # Add compilation/execution info if present
        if analysis['has_makefile'] or analysis['has_sbatch']:
            content += "## Usage\n\n"
            
            if analysis['has_makefile']:
                content += "### Compilation\n"
                content += "```bash\nmake\n```\n\n"
            
            if analysis['has_sbatch']:
                content += "### Execution\n"
                content += "```bash\nsbatch run.sbatch\n```\n\n"
        
        # Generate Quick Reference table
        table_items = []
        for subdir in analysis['subdirs'][:10]:  # Limit to 10 entries
            sub_analysis = self.scan_directory(subdir)
            table_items.append({
                'name': subdir.name,
                'description': f"{len(sub_analysis['code_files'])} files",
                'languages': list(sub_analysis['languages']),
                'example_count': len(sub_analysis['examples'])
            })
        
        if table_items:
            content += self.generate_quick_reference_table(table_items, 'top')
        
        return content
    
    def _generate_mid_level_content(self, path: Path, analysis: Dict) -> str:
        """Generate content for mid-level directories."""
        content = ""
        
        # List examples or subdirectories
        if analysis['examples']:
            content += "## Examples\n\n"
            for example_dir in sorted(analysis['examples']):
                example_analysis = self.scan_directory(example_dir)
                example_name = example_dir.name
                
                # Build description
                desc_parts = []
                if example_analysis['languages']:
                    desc_parts.append(f"{', '.join(example_analysis['languages'])}")
                if example_analysis['has_makefile']:
                    desc_parts.append("Makefile")
                if example_analysis['has_sbatch']:
                    desc_parts.append("SLURM job")
                
                desc = " - ".join(desc_parts) if desc_parts else "Example code"
                content += f"- **{example_name}/**: {desc}\n"
            
            content += "\n"
        
        # Add code files listing
        if analysis['code_files']:
            content += "## Code Files\n\n"
            for file_info in analysis['code_files'][:10]:  # Limit listing
                file_name = file_info['path'].name
                lang = file_info['language']
                desc = file_info['description'] or "Source code"
                content += f"- **{file_name}** ({lang}): {desc}\n"
            content += "\n"
        
        # Generate Quick Reference table for examples
        if analysis['examples']:
            table_items = []
            for example_dir in analysis['examples'][:10]:
                example_analysis = self.scan_directory(example_dir)
                
                # Get main language
                main_lang = "Mixed"
                if len(example_analysis['languages']) == 1:
                    main_lang = list(example_analysis['languages'])[0]
                
                # Get key files
                key_files = []
                for f in example_analysis['code_files'][:3]:
                    key_files.append(f['path'].name)
                
                # Get purpose from first code file with AI enhancement
                purpose = "Example implementation"
                if example_analysis['code_files'] and example_analysis['code_files'][0]['description']:
                    first_file = example_analysis['code_files'][0]
                    purpose = self.enhance_code_purpose_with_ai(
                        first_file['path'], 
                        first_file['description'],
                        first_file['language']
                    ) or first_file['description']
                
                table_items.append({
                    'name': example_dir.name,
                    'purpose': purpose,
                    'language': main_lang,
                    'key_files': key_files
                })
            
            if table_items:
                content += self.generate_quick_reference_table(table_items, 'mid')
        
        return content
    
    def _generate_leaf_level_content(self, path: Path, analysis: Dict) -> str:
        """Generate minimal content for leaf-level directories."""
        content = ""
        
        # Just list the files
        if analysis['code_files']:
            content += "## Files\n\n"
            for file_info in analysis['code_files']:
                file_name = file_info['path'].name
                lang = file_info['language']
                content += f"- **{file_name}** ({lang})\n"
            content += "\n"
        
        if analysis['special_files']:
            content += "## Build/Run Files\n\n"
            for file_path in analysis['special_files']:
                content += f"- {file_path.name}\n"
            content += "\n"
        
        return content
    
    def write_readme(self, path: Path, content: str) -> Optional[Path]:
        """
        Write README content to file with suffix/overwrite handling.
        
        Args:
            path: Directory path where README should be written
            content: README content to write
            
        Returns:
            Path to written file or None if skipped
        """
        if not content:
            return None
        
        # Determine output filename
        if self.overwrite:
            readme_path = path / "README.md"
            # Backup existing if present
            if readme_path.exists():
                backup_path = path / "README.md.backup"
                if not backup_path.exists():  # Don't overwrite existing backups
                    readme_path.rename(backup_path)
                    self.logger.info(f"Backed up existing README to {backup_path}")
        else:
            readme_path = path / f"README{self.suffix}.md"
        
        try:
            readme_path.write_text(content, encoding='utf-8')
            self.logger.info(f"Generated {readme_path}")
            return readme_path
        except Exception as e:
            self.logger.error(f"Failed to write {readme_path}: {e}")
            return None
    
    def process_directory_tree(self) -> List[Path]:
        """
        Process the entire directory tree and generate README files.
        
        Returns:
            List of paths to generated README files
        """
        self.generated_files = []
        
        # Process source directory
        self._process_directory(self.source_dir)
        
        # Process subdirectories if recursive
        if self.recursive:
            self._process_subdirectories(self.source_dir)
        
        return self.generated_files
    
    def _process_directory(self, path: Path) -> None:
        """Process a single directory."""
        self.logger.debug(f"Processing {path}")
        
        # Scan and analyze
        analysis = self.scan_directory(path)
        
        # Generate content
        content = self.generate_readme_content(path, analysis)
        
        # Write file
        if content:
            written_path = self.write_readme(path, content)
            if written_path:
                self.generated_files.append(written_path)
    
    def _process_subdirectories(self, path: Path) -> None:
        """Recursively process subdirectories."""
        analysis = self.scan_directory(path)
        
        # Process regular subdirectories
        for subdir in analysis['subdirs']:
            if not self.should_exclude(subdir):
                self._process_directory(subdir)
                self._process_subdirectories(subdir)
        
        # Process example directories
        for example_dir in analysis['examples']:
            if not self.should_exclude(example_dir):
                level = self.determine_depth_level(example_dir)
                # Only process non-leaf example directories
                if level != 'leaf':
                    self._process_directory(example_dir)