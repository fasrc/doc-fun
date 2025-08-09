"""
Document compiler plugin for generating best compilation from multiple runs.
"""

from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import logging
from bs4 import BeautifulSoup

from ..plugins.analysis_base import AnalysisPlugin
from ..core import DocumentAnalyzer, GPTQualityEvaluator


class DocumentCompiler(AnalysisPlugin):
    """
    Compiles the best sections from multiple document runs into a single optimal document.
    
    This plugin analyzes multiple generated documents, scores each section using both
    algorithmic metrics and optional GPT evaluation, then creates a compilation
    using the highest-scoring sections.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, config: Optional[Dict] = None, **kwargs):
        """
        Initialize the DocumentCompiler.
        
        Config options:
            weights: Dict with 'algorithmic' and 'gpt_quality' weights (default: {algorithmic: 0.7, gpt_quality: 0.3})
            use_gpt: Boolean to enable GPT quality evaluation (default: False)
            section_headers: List of section headers to analyze (default: standard sections)
            min_runs: Minimum number of runs required for compilation (default: 2)
        """
        super().__init__(logger, config, **kwargs)
        
        # Configuration
        self.weights = self.config.get('weights', {'algorithmic': 0.7, 'gpt_quality': 0.3})
        self.use_gpt = self.config.get('use_gpt', False)
        self.section_headers = self.config.get('section_headers', [
            'Description', 'Installation', 'Usage', 'Examples', 'References'
        ])
        self.min_runs = self.config.get('min_runs', 2)
        
        # Initialize analyzer
        self.analyzer = DocumentAnalyzer(section_headers=self.section_headers)
        self.gpt_evaluator = None  # Will be initialized if needed
    
    def get_name(self) -> str:
        """Return the plugin name."""
        return 'compiler'
    
    def analyze(self, documents: List[Dict[str, Any]], topic: str) -> Dict[str, Any]:
        """
        Analyze documents and select best sections for compilation.
        
        Args:
            documents: List of document dictionaries with 'path' and 'content'
            topic: The topic used for generation
            
        Returns:
            Dictionary containing:
                - best_sections: Dict mapping section names to best content and metadata
                - section_scores: Dict with all section scores across documents
                - compilation_html: The compiled HTML document
        """
        if len(documents) < self.min_runs:
            self.logger.warning(f"Only {len(documents)} documents provided, minimum {self.min_runs} required for compilation")
            return {
                'best_sections': {},
                'section_scores': {},
                'compilation_html': None,
                'message': f'Insufficient documents for compilation (need at least {self.min_runs})'
            }
        
        # Extract and score all sections from all documents
        all_sections = {}  # {section_name: [(content, score, doc_index), ...]}
        
        for doc_idx, doc in enumerate(documents):
            content = doc.get('content', '')
            if not content:
                self.logger.warning(f"Document {doc_idx} has no content")
                continue
            
            # Extract sections from this document
            sections = self.analyzer.extract_sections(content)
            
            # Score each section
            for section_name, section_content in sections.items():
                if section_name not in all_sections:
                    all_sections[section_name] = []
                
                # Calculate algorithmic score
                algo_score = self.analyzer.calculate_section_score(section_content, section_name)
                
                # Optionally add GPT score (simplified for now)
                gpt_score = 0
                if self.use_gpt and self.gpt_evaluator:
                    # This would require the GPT evaluator to be initialized with a client
                    # For now, we'll skip GPT evaluation in the initial implementation
                    pass
                
                # Combined score
                total_score = (self.weights['algorithmic'] * algo_score + 
                             self.weights['gpt_quality'] * gpt_score)
                
                all_sections[section_name].append({
                    'content': section_content,
                    'score': total_score,
                    'algo_score': algo_score,
                    'gpt_score': gpt_score,
                    'doc_index': doc_idx,
                    'doc_path': doc.get('path', f'document_{doc_idx}')
                })
        
        # Select best section for each header
        best_sections = {}
        section_scores = {}
        
        for section_name, section_variants in all_sections.items():
            if not section_variants:
                continue
            
            # Sort by score and select the best
            section_variants.sort(key=lambda x: x['score'], reverse=True)
            best_sections[section_name] = section_variants[0]
            
            # Store all scores for reporting
            section_scores[section_name] = [
                {
                    'doc_index': v['doc_index'],
                    'doc_path': v['doc_path'],
                    'score': v['score'],
                    'algo_score': v['algo_score'],
                    'gpt_score': v['gpt_score']
                }
                for v in section_variants
            ]
        
        # Generate compilation HTML
        compilation_html = self._create_compilation_html(best_sections, topic)
        
        return {
            'best_sections': best_sections,
            'section_scores': section_scores,
            'compilation_html': compilation_html
        }
    
    def _create_compilation_html(self, best_sections: Dict[str, Dict], topic: str) -> str:
        """
        Create a compiled HTML document from the best sections.
        
        Args:
            best_sections: Dictionary mapping section names to their best content
            topic: The documentation topic
            
        Returns:
            Complete HTML document string
        """
        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '<meta charset="UTF-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f'<title>{topic} - Best Compilation</title>',
            '<style>',
            'body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; ',
            '       line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; }',
            'h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }',
            'h2 { color: #34495e; margin-top: 30px; }',
            'pre { background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }',
            'code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }',
            '.compilation-note { background: #e8f4fd; border-left: 4px solid #3498db; ',
            '                    padding: 10px; margin: 20px 0; font-style: italic; }',
            '</style>',
            '</head>',
            '<body>',
            f'<h1>{topic}</h1>',
            '<div class="compilation-note">',
            'This is a compilation of the best sections from multiple generation runs.',
            '</div>'
        ]
        
        # Add each section in the standard order
        for section_name in self.section_headers:
            if section_name in best_sections:
                section_data = best_sections[section_name]
                html_parts.append(f'<h2>{section_name}</h2>')
                html_parts.append(section_data['content'])
                html_parts.append(f'<!-- Source: {section_data["doc_path"]} (score: {section_data["score"]:.2f}) -->')
        
        html_parts.extend([
            '</body>',
            '</html>'
        ])
        
        return '\n'.join(html_parts)
    
    def generate_report(self, analysis_results: Dict[str, Any], topic: str) -> str:
        """
        Generate a markdown report of the compilation analysis.
        
        Args:
            analysis_results: Results from analyze() method
            topic: The topic used for generation
            
        Returns:
            Markdown-formatted report
        """
        report_lines = [
            f'# Document Compilation Report: {topic}',
            '',
            '## Summary',
            ''
        ]
        
        if 'message' in analysis_results:
            report_lines.append(f"**Note:** {analysis_results['message']}")
            report_lines.append('')
            return '\n'.join(report_lines)
        
        best_sections = analysis_results.get('best_sections', {})
        section_scores = analysis_results.get('section_scores', {})
        
        report_lines.extend([
            f'- **Sections Compiled:** {len(best_sections)}',
            f'- **Total Documents Analyzed:** {self._count_unique_docs(section_scores)}',
            '',
            '## Best Section Selection',
            ''
        ])
        
        # Table of best sections
        report_lines.extend([
            '| Section | Best Source | Score | Algorithm Score | GPT Score |',
            '|---------|-------------|--------|-----------------|-----------|'
        ])
        
        for section_name in self.section_headers:
            if section_name in best_sections:
                section = best_sections[section_name]
                doc_name = Path(section['doc_path']).name if isinstance(section['doc_path'], str) else f"Doc {section['doc_index']}"
                report_lines.append(
                    f"| {section_name} | {doc_name} | "
                    f"{section['score']:.2f} | {section['algo_score']:.2f} | "
                    f"{section['gpt_score']:.2f} |"
                )
        
        report_lines.append('')
        
        # Detailed section scores
        report_lines.extend([
            '## Detailed Section Scores',
            ''
        ])
        
        for section_name in self.section_headers:
            if section_name in section_scores:
                report_lines.extend([
                    f'### {section_name}',
                    ''
                ])
                
                scores = section_scores[section_name]
                scores.sort(key=lambda x: x['score'], reverse=True)
                
                for idx, score_data in enumerate(scores, 1):
                    doc_name = Path(score_data['doc_path']).name if isinstance(score_data['doc_path'], str) else f"Doc {score_data['doc_index']}"
                    winner = '**[SELECTED]**' if idx == 1 else ''
                    report_lines.append(
                        f"{idx}. {doc_name}: {score_data['score']:.2f} "
                        f"(algo: {score_data['algo_score']:.2f}, gpt: {score_data['gpt_score']:.2f}) {winner}"
                    )
                
                report_lines.append('')
        
        return '\n'.join(report_lines)
    
    def _count_unique_docs(self, section_scores: Dict) -> int:
        """Count unique documents across all sections."""
        unique_docs = set()
        for section_variants in section_scores.values():
            for variant in section_variants:
                unique_docs.add(variant['doc_index'])
        return len(unique_docs)
    
    def save_artifacts(self, results: Dict[str, Any], output_dir: Path, topic: str) -> List[Path]:
        """
        Save the compilation HTML and report to files.
        
        Args:
            results: Analysis results including compilation HTML and report
            output_dir: Directory to save artifacts
            topic: The topic used for generation
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        safe_topic = self.sanitize_filename(topic)
        
        # Save compilation HTML if available
        if results.get('compilation_html'):
            compilation_path = output_dir / f'{safe_topic}_best_compilation.html'
            with open(compilation_path, 'w', encoding='utf-8') as f:
                f.write(results['compilation_html'])
            saved_files.append(compilation_path)
            self.logger.info(f"Saved compilation to {compilation_path}")
        
        # Save compilation report
        if results:
            report = self.generate_report(results, topic)
            report_path = output_dir / f'{safe_topic}_compilation_report.md'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            saved_files.append(report_path)
            self.logger.info(f"Saved compilation report to {report_path}")
        
        return saved_files
    
    def get_priority(self) -> int:
        """Compiler should run first to create the compilation."""
        return 100
    
    def get_supported_formats(self) -> List[str]:
        """Return supported output formats."""
        return ['html', 'markdown']