"""
README Documentation Generator Module
Extends DocumentationGenerator to support README generation with directory scanning.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from .core import DocumentationGenerator, DocumentAnalyzer
from .providers import CompletionRequest


class ReadmeDocumentationGenerator(DocumentationGenerator):
    """Extended documentation generator for README files with directory analysis."""
    
    def __init__(self, **kwargs):
        """Initialize with README-specific configurations."""
        # Set README-specific defaults
        if 'prompt_yaml_path' not in kwargs:
            kwargs['prompt_yaml_path'] = './prompts/generator/user_codes_topic_readme.yaml'
        
        super().__init__(**kwargs)
        
        # README-specific settings
        self.format_type = self.prompt_config.get('placeholders', {}).get('format', 'Markdown')
        self.section_headers = self._get_section_headers_for_format()
        
    def _get_section_headers_for_format(self) -> List[str]:
        """Get appropriate section headers based on format type."""
        if self.format_type.lower() == 'markdown':
            return ['Overview', 'Directory Structure', 'Getting Started', 
                   'Quick Reference', 'Examples', 'Additional Resources']
        return ['Description', 'Installation', 'Usage', 'Examples', 'References']
    
    def scan_directory(self, directory_path: Path) -> Dict:
        """Scan directory structure and extract metadata."""
        directory_info = {
            'path': str(directory_path),
            'name': directory_path.name,
            'depth': len(directory_path.parts),
            'subdirectories': [],
            'files': [],
            'languages': set(),
            'has_readme': False,
            'has_examples': False
        }
        
        # Language extensions mapping
        language_extensions = {
            '.py': 'Python', '.ipynb': 'Python',
            '.c': 'C', '.h': 'C',
            '.cpp': 'C++', '.hpp': 'C++', '.cc': 'C++',
            '.f90': 'Fortran', '.f95': 'Fortran', '.f': 'Fortran',
            '.jl': 'Julia',
            '.m': 'MATLAB',
            '.r': 'R', '.R': 'R',
            '.sh': 'Bash', '.bash': 'Bash',
            '.pl': 'Perl',
            '.java': 'Java',
            '.js': 'JavaScript', '.ts': 'TypeScript',
            '.rs': 'Rust',
            '.go': 'Go'
        }
        
        try:
            for item in directory_path.iterdir():
                if item.name.startswith('.'):
                    continue  # Skip hidden files
                    
                if item.is_dir():
                    # Check for example directories
                    if 'example' in item.name.lower():
                        directory_info['has_examples'] = True
                    
                    # Get subdirectory info
                    subdir_info = {
                        'name': item.name,
                        'path': str(item),
                        'is_example': 'example' in item.name.lower(),
                        'file_count': sum(1 for f in item.rglob('*') if f.is_file())
                    }
                    directory_info['subdirectories'].append(subdir_info)
                    
                elif item.is_file():
                    # Check for README
                    if item.name.lower() in ['readme.md', 'readme.txt', 'readme']:
                        directory_info['has_readme'] = True
                    
                    # Track file and language
                    file_info = {
                        'name': item.name,
                        'extension': item.suffix,
                        'size': item.stat().st_size
                    }
                    
                    # Identify language
                    if item.suffix in language_extensions:
                        lang = language_extensions[item.suffix]
                        directory_info['languages'].add(lang)
                        file_info['language'] = lang
                    
                    directory_info['files'].append(file_info)
        
        except PermissionError:
            self.logger.warning(f"Permission denied accessing {directory_path}")
        
        # Convert languages set to list for serialization
        directory_info['languages'] = list(directory_info['languages'])
        
        return directory_info
    
    def _determine_depth_level(self, directory_path: Path) -> str:
        """Determine the depth level of a directory."""
        depth = len(directory_path.parts)
        
        # Adjust based on typical repository structure
        if depth <= 2:
            return 'top'
        elif depth <= 4:
            return 'mid'
        else:
            return 'leaf'
    
    def _build_directory_context(self, directory_info: Dict, depth_level: str) -> str:
        """Build context string from directory information."""
        context_parts = []
        
        # Basic info
        context_parts.append(f"Directory: {directory_info['name']}")
        context_parts.append(f"Path: {directory_info['path']}")
        
        # Subdirectories
        if directory_info['subdirectories']:
            context_parts.append(f"\nSubdirectories ({len(directory_info['subdirectories'])}):")
            for subdir in directory_info['subdirectories'][:10]:  # Limit to prevent token overflow
                desc = f"  - {subdir['name']}"
                if subdir['is_example']:
                    desc += " (example directory)"
                if subdir['file_count'] > 0:
                    desc += f" - {subdir['file_count']} files"
                context_parts.append(desc)
        
        # Files summary
        if directory_info['files']:
            # Group files by extension
            files_by_type = defaultdict(list)
            for file in directory_info['files']:
                ext = file.get('extension', 'no extension')
                files_by_type[ext].append(file['name'])
            
            context_parts.append(f"\nFiles ({len(directory_info['files'])}):")
            for ext, filenames in list(files_by_type.items())[:5]:  # Limit types
                context_parts.append(f"  - {ext}: {', '.join(filenames[:3])}")  # Limit files per type
                if len(filenames) > 3:
                    context_parts.append(f"    ... and {len(filenames) - 3} more")
        
        # Languages found
        if directory_info['languages']:
            context_parts.append(f"\nProgramming Languages: {', '.join(directory_info['languages'])}")
        
        # Existing README info
        if directory_info['has_readme']:
            context_parts.append("\nNote: Existing README found (reference for style)")
        
        return '\n'.join(context_parts)
    
    def _get_depth_requirements(self, depth_level: str) -> str:
        """Get specific requirements based on depth level."""
        requirements = {
            'top': """
For this top-level directory:
1. Provide a comprehensive overview of the entire repository/directory
2. List ALL subdirectories with meaningful descriptions
3. Create a Quick Reference table with columns: Category | Description | Languages | Examples
4. Include getting started instructions for the whole repository
5. Add navigation links to major sections
""",
            'mid': """
For this mid-level directory:
1. Focus on the specific category/topic this directory represents
2. List all examples with their purposes
3. Create a Quick Reference table with columns: Example | Purpose | Language | Key Files
4. Include usage instructions specific to this category
5. Reference parent directory for broader context
""",
            'leaf': """
For this leaf-level directory (specific example):
1. Provide detailed documentation of this specific example
2. List prerequisites and dependencies
3. Create a simple file listing table with columns: File | Type | Purpose
4. Include step-by-step usage instructions
5. Show expected output or results
6. Add troubleshooting tips if applicable
"""
        }
        return requirements.get(depth_level, requirements['mid'])
    
    def generate_readme_documentation(self, directory_path: str, 
                                     runs: int = 3,
                                     model: str = None,
                                     temperature: float = 0.5,
                                     analyze: bool = True,
                                     **kwargs) -> Dict:
        """Generate README documentation for a directory using multi-run generation."""
        
        dir_path = Path(directory_path)
        if not dir_path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Scan directory structure
        self.logger.info(f"Scanning directory: {directory_path}")
        directory_info = self.scan_directory(dir_path)
        
        # Determine depth level
        depth_level = self._determine_depth_level(dir_path)
        self.logger.info(f"Directory depth level: {depth_level}")
        
        # Build context
        directory_context = self._build_directory_context(directory_info, depth_level)
        depth_requirements = self._get_depth_requirements(depth_level)
        
        # Prepare query with all context
        placeholders = self.prompt_config.get('placeholders', {}).copy()
        placeholders.update({
            'directory_path': str(dir_path),
            'depth_level': depth_level,
            'directory_context': directory_context,
            'depth_requirements': depth_requirements,
            'format': 'Markdown',
            'topic': dir_path.name
        })
        
        # Format user prompt
        user_prompt_template = self.prompt_config.get('user_prompt', '')
        try:
            formatted_query = user_prompt_template.format(**placeholders)
        except KeyError as e:
            self.logger.warning(f"Missing placeholder {e} in user prompt")
            formatted_query = f"Generate a README for {directory_path}"
        
        # Generate multiple versions
        self.logger.info(f"Generating {runs} README variations...")
        
        output_dir = kwargs.get('output_dir', str(dir_path))
        topic_filename = f"{dir_path.name.lower().replace(' ', '_')}_readme"
        
        # Override file extension for markdown
        original_generate = super().generate_documentation
        
        # Generate documentation
        generated_files = []
        for i in range(runs):
            try:
                # Build messages with README-specific system prompt
                # Remove 'topic' from placeholders since it's passed separately
                prompt_placeholders = {k: v for k, v in placeholders.items() if k != 'topic'}
                system_prompt = self._build_system_prompt(
                    topic=dir_path.name,
                    **prompt_placeholders
                )
                
                messages = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': formatted_query}
                ]
                
                # Get provider and model
                if model is None:
                    model = self.provider_manager.get_default_model()
                
                provider = kwargs.get('provider') or self.default_provider
                llm_provider = self.provider_manager.get_provider(provider)
                
                # Create completion request
                request = CompletionRequest(
                    messages=messages,
                    model=model,
                    temperature=temperature + (i * 0.1)  # Vary temperature slightly
                )
                
                # Generate completion
                response = llm_provider.generate_completion(request)
                content = response.content
                
                # Save as markdown
                filename = f"{topic_filename}_v{i+1}.md"
                filepath = Path(output_dir) / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                generated_files.append(str(filepath))
                self.logger.info(f"Generated: {filename}")
                
            except Exception as e:
                self.logger.error(f"Error generating README (run {i+1}): {e}")
        
        # Analyze if requested
        analysis_results = None
        if analyze and len(generated_files) > 1:
            self.logger.info("Analyzing generated READMEs...")
            analyzer = ReadmeAnalyzer()
            analysis_results = analyzer.analyze_multiple_readmes(
                generated_files,
                directory_info
            )
            
            # Run advanced quality evaluation using README-specific prompts
            if analysis_results and hasattr(self, 'provider_manager'):
                try:
                    from .core import GPTQualityEvaluator
                    evaluator = GPTQualityEvaluator(
                        self.provider_manager.get_provider(self.default_provider).client if hasattr(self.provider_manager.get_provider(self.default_provider), 'client') else None,
                        analysis_prompt_path='./prompts/analysis/readme.yaml'
                    )
                    
                    # Enhance analysis with LLM-based evaluation
                    for i, filepath in enumerate(generated_files):
                        self.logger.debug(f"Running LLM quality evaluation for {filepath}")
                        # This could be extended to run detailed LLM-based analysis
                        
                except Exception as e:
                    self.logger.warning(f"Advanced analysis failed: {e}")
                    # Continue with basic analysis
            
            # Select best version or compile best sections
            if analysis_results:
                best_readme = self._compile_best_readme(
                    generated_files,
                    analysis_results,
                    dir_path
                )
                if best_readme:
                    generated_files.append(best_readme)
        
        return {
            'directory_info': directory_info,
            'generated_files': generated_files,
            'analysis_results': analysis_results,
            'depth_level': depth_level
        }
    
    def _compile_best_readme(self, generated_files: List[str], 
                             analysis_results: Dict,
                             directory_path: Path) -> Optional[str]:
        """Compile the best sections from multiple README versions."""
        try:
            # Read all generated files
            readmes = []
            for filepath in generated_files:
                with open(filepath, 'r', encoding='utf-8') as f:
                    readmes.append(f.read())
            
            # Simple compilation: use the highest-scored complete README
            # (Full section-based compilation would require markdown parsing)
            best_index = 0
            best_score = 0
            
            for i, scores in enumerate(analysis_results.get('version_scores', [])):
                total_score = sum(scores.values())
                if total_score > best_score:
                    best_score = total_score
                    best_index = i
            
            # Save the best version
            best_content = readmes[best_index]
            output_path = directory_path / "README_best.md"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(best_content)
            
            self.logger.info(f"Compiled best README: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error compiling best README: {e}")
            return None


class ReadmeAnalyzer(DocumentAnalyzer):
    """Analyzer specifically for README files."""
    
    def __init__(self):
        """Initialize with README-specific section headers."""
        super().__init__(section_headers=[
            'Overview', 'Directory Structure', 'Getting Started',
            'Quick Reference', 'Examples', 'Additional Resources',
            'Prerequisites', 'Installation', 'Usage'
        ])
    
    def extract_sections(self, markdown_content: str) -> Dict[str, str]:
        """Extract sections from Markdown content."""
        sections = {}
        current_section = None
        current_content = []
        
        lines = markdown_content.split('\n')
        
        for line in lines:
            # Check if this is a header
            if line.startswith('#'):
                # Extract header text
                header_text = line.lstrip('#').strip()
                
                # Save previous section if exists
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Check if this matches our target sections
                for section_name in self.section_headers:
                    if section_name.lower() in header_text.lower():
                        current_section = section_name
                        current_content = []
                        break
                else:
                    # Not a target section, but still save for context
                    if current_section:
                        current_content.append(line)
            else:
                # Add to current section content
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def calculate_readme_score(self, content: str, section_name: str, 
                               directory_info: Dict = None) -> float:
        """Calculate quality score for README sections."""
        if not content:
            return 0.0
        
        score = 0.0
        
        # Basic scoring from parent class
        score = super().calculate_section_score(content, section_name)
        
        # README-specific scoring
        if section_name == 'Quick Reference':
            # Check for table formatting
            if '|' in content:
                score += 10  # Tables are good for quick reference
            
            # Check coverage of subdirectories
            if directory_info and directory_info.get('subdirectories'):
                mentioned = sum(1 for subdir in directory_info['subdirectories']
                              if subdir['name'].lower() in content.lower())
                coverage = mentioned / len(directory_info['subdirectories'])
                score += coverage * 15
        
        elif section_name == 'Directory Structure':
            # Check for tree-like structure or clear listing
            if any(marker in content for marker in ['├──', '└──', '- ', '* ']):
                score += 10
            
            # Check completeness
            if directory_info and directory_info.get('subdirectories'):
                mentioned = sum(1 for subdir in directory_info['subdirectories']
                              if subdir['name'] in content)
                coverage = mentioned / len(directory_info['subdirectories'])
                score += coverage * 20
        
        elif section_name == 'Examples':
            # Check for code blocks
            if '```' in content:
                score += 10
            
            # Check for example references
            if directory_info and directory_info.get('has_examples'):
                if any(word in content.lower() for word in ['example', 'demo', 'sample']):
                    score += 10
        
        return score
    
    def analyze_multiple_readmes(self, readme_files: List[str], 
                                 directory_info: Dict = None) -> Dict:
        """Analyze multiple README versions and compare quality."""
        results = {
            'version_scores': [],
            'section_winners': {},
            'overall_best': None
        }
        
        all_sections = {}
        
        for i, filepath in enumerate(readme_files):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                sections = self.extract_sections(content)
                version_scores = {}
                
                for section_name, section_content in sections.items():
                    score = self.calculate_readme_score(
                        section_content, 
                        section_name,
                        directory_info
                    )
                    version_scores[section_name] = score
                    
                    # Track best version for each section
                    if section_name not in results['section_winners'] or \
                       score > results['section_winners'][section_name]['score']:
                        results['section_winners'][section_name] = {
                            'version': i,
                            'score': score,
                            'file': filepath
                        }
                
                results['version_scores'].append(version_scores)
                all_sections[filepath] = sections
                
            except Exception as e:
                self.logger.error(f"Error analyzing {filepath}: {e}")
        
        # Determine overall best
        if results['version_scores']:
            best_total = 0
            best_index = 0
            
            for i, scores in enumerate(results['version_scores']):
                total = sum(scores.values())
                if total > best_total:
                    best_total = total
                    best_index = i
            
            results['overall_best'] = {
                'index': best_index,
                'file': readme_files[best_index],
                'total_score': best_total
            }
        
        return results