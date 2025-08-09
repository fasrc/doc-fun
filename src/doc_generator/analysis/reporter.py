"""
Analysis reporter plugin for generating detailed analysis reports.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
import logging
import json
from datetime import datetime
from bs4 import BeautifulSoup

from ..plugins.analysis_base import AnalysisPlugin
from ..core import DocumentAnalyzer


class AnalysisReporter(AnalysisPlugin):
    """
    Generates comprehensive analysis reports from multiple document runs.
    
    This plugin analyzes generated documents to provide detailed metrics,
    comparisons, and insights about the documentation quality and structure.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, config: Optional[Dict] = None, **kwargs):
        """
        Initialize the AnalysisReporter.
        
        Config options:
            formats: List of output formats (default: ['markdown'])
            include_stats: Include detailed statistics (default: True)
            include_comparisons: Include cross-document comparisons (default: True)
            section_headers: List of section headers to analyze
        """
        super().__init__(logger, config, **kwargs)
        
        # Configuration
        self.formats = self.config.get('formats', ['markdown'])
        self.include_stats = self.config.get('include_stats', True)
        self.include_comparisons = self.config.get('include_comparisons', True)
        self.section_headers = self.config.get('section_headers', [
            'Description', 'Installation', 'Usage', 'Examples', 'References'
        ])
        
        # Initialize analyzer
        self.analyzer = DocumentAnalyzer(section_headers=self.section_headers)
    
    def get_name(self) -> str:
        """Return the plugin name."""
        return 'reporter'
    
    def analyze(self, documents: List[Dict[str, Any]], topic: str) -> Dict[str, Any]:
        """
        Analyze documents and gather comprehensive metrics.
        
        Args:
            documents: List of document dictionaries with 'path' and 'content'
            topic: The topic used for generation
            
        Returns:
            Dictionary containing:
                - document_metrics: List of metrics for each document
                - section_analysis: Section-level analysis across documents
                - overall_stats: Aggregate statistics
                - comparisons: Cross-document comparisons
        """
        document_metrics = []
        all_sections = {}  # {section_name: [metrics_per_doc]}
        
        for doc_idx, doc in enumerate(documents):
            content = doc.get('content', '')
            path = doc.get('path', f'document_{doc_idx}')
            
            if not content:
                self.logger.warning(f"Document {doc_idx} has no content")
                continue
            
            # Analyze this document
            doc_metrics = self._analyze_document(content, path, doc_idx)
            document_metrics.append(doc_metrics)
            
            # Collect section-level data
            for section_name, section_data in doc_metrics['sections'].items():
                if section_name not in all_sections:
                    all_sections[section_name] = []
                all_sections[section_name].append({
                    'doc_index': doc_idx,
                    'doc_path': path,
                    **section_data
                })
        
        # Calculate overall statistics
        overall_stats = self._calculate_overall_stats(document_metrics)
        
        # Generate comparisons if enabled
        comparisons = {}
        if self.include_comparisons and len(document_metrics) > 1:
            comparisons = self._generate_comparisons(document_metrics, all_sections)
        
        return {
            'document_metrics': document_metrics,
            'section_analysis': all_sections,
            'overall_stats': overall_stats,
            'comparisons': comparisons,
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_document(self, content: str, path: str, index: int) -> Dict[str, Any]:
        """
        Analyze a single document and extract metrics.
        
        Args:
            content: HTML content of the document
            path: Path to the document
            index: Document index
            
        Returns:
            Dictionary of document metrics
        """
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract sections
        sections = self.analyzer.extract_sections(content)
        
        # Document-level metrics
        doc_metrics = {
            'path': path,
            'index': index,
            'total_length': len(content),
            'total_words': len(soup.get_text().split()),
            'sections': {},
            'code_blocks': 0,
            'links': 0,
            'images': 0,
            'tables': 0,
            'lists': 0
        }
        
        # Count elements
        doc_metrics['code_blocks'] = len(soup.find_all('pre')) + len(soup.find_all('code'))
        doc_metrics['links'] = len(soup.find_all('a'))
        doc_metrics['images'] = len(soup.find_all('img'))
        doc_metrics['tables'] = len(soup.find_all('table'))
        doc_metrics['lists'] = len(soup.find_all('ul')) + len(soup.find_all('ol'))
        
        # Analyze each section
        for section_name, section_content in sections.items():
            section_soup = BeautifulSoup(section_content, 'html.parser')
            section_text = section_soup.get_text()
            
            section_metrics = {
                'exists': True,
                'length': len(section_content),
                'words': len(section_text.split()),
                'score': self.analyzer.calculate_section_score(section_content, section_name),
                'code_blocks': len(section_soup.find_all('pre')) + len(section_soup.find_all('code')),
                'links': len(section_soup.find_all('a')),
                'paragraphs': len(section_soup.find_all('p')),
                'lists': len(section_soup.find_all('ul')) + len(section_soup.find_all('ol'))
            }
            
            doc_metrics['sections'][section_name] = section_metrics
        
        # Add missing sections
        for header in self.section_headers:
            if header not in doc_metrics['sections']:
                doc_metrics['sections'][header] = {
                    'exists': False,
                    'length': 0,
                    'words': 0,
                    'score': 0,
                    'code_blocks': 0,
                    'links': 0,
                    'paragraphs': 0,
                    'lists': 0
                }
        
        return doc_metrics
    
    def _calculate_overall_stats(self, document_metrics: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate statistics across all documents."""
        if not document_metrics:
            return {}
        
        stats = {
            'total_documents': len(document_metrics),
            'average_length': sum(d['total_length'] for d in document_metrics) / len(document_metrics),
            'average_words': sum(d['total_words'] for d in document_metrics) / len(document_metrics),
            'average_code_blocks': sum(d['code_blocks'] for d in document_metrics) / len(document_metrics),
            'average_links': sum(d['links'] for d in document_metrics) / len(document_metrics),
            'section_coverage': {}
        }
        
        # Calculate section coverage
        for header in self.section_headers:
            coverage = sum(1 for d in document_metrics if d['sections'].get(header, {}).get('exists', False))
            stats['section_coverage'][header] = {
                'count': coverage,
                'percentage': (coverage / len(document_metrics)) * 100
            }
        
        return stats
    
    def _generate_comparisons(self, document_metrics: List[Dict], all_sections: Dict) -> Dict[str, Any]:
        """Generate cross-document comparisons."""
        comparisons = {
            'best_overall': None,
            'worst_overall': None,
            'most_comprehensive': None,
            'best_sections': {},
            'consistency_analysis': {}
        }
        
        if not document_metrics:
            return comparisons
        
        # Find best/worst overall by total score
        for doc in document_metrics:
            total_score = sum(s['score'] for s in doc['sections'].values())
            doc['total_score'] = total_score
        
        sorted_docs = sorted(document_metrics, key=lambda x: x.get('total_score', 0), reverse=True)
        comparisons['best_overall'] = {
            'path': sorted_docs[0]['path'],
            'index': sorted_docs[0]['index'],
            'total_score': sorted_docs[0]['total_score']
        }
        comparisons['worst_overall'] = {
            'path': sorted_docs[-1]['path'],
            'index': sorted_docs[-1]['index'],
            'total_score': sorted_docs[-1]['total_score']
        }
        
        # Find most comprehensive (most content)
        most_comp = max(document_metrics, key=lambda x: x['total_words'])
        comparisons['most_comprehensive'] = {
            'path': most_comp['path'],
            'index': most_comp['index'],
            'total_words': most_comp['total_words']
        }
        
        # Find best section for each header
        for section_name, section_variants in all_sections.items():
            if section_variants:
                best = max(section_variants, key=lambda x: x.get('score', 0))
                comparisons['best_sections'][section_name] = {
                    'doc_path': best['doc_path'],
                    'doc_index': best['doc_index'],
                    'score': best.get('score', 0)
                }
        
        # Consistency analysis
        for section_name in self.section_headers:
            if section_name in all_sections:
                word_counts = [s['words'] for s in all_sections[section_name] if s.get('exists', False)]
                if word_counts:
                    avg_words = sum(word_counts) / len(word_counts)
                    variance = sum((w - avg_words) ** 2 for w in word_counts) / len(word_counts)
                    comparisons['consistency_analysis'][section_name] = {
                        'average_words': avg_words,
                        'variance': variance,
                        'std_dev': variance ** 0.5
                    }
        
        return comparisons
    
    def generate_report(self, analysis_results: Dict[str, Any], topic: str) -> str:
        """
        Generate a comprehensive markdown report.
        
        Args:
            analysis_results: Results from analyze() method
            topic: The topic used for generation
            
        Returns:
            Markdown-formatted report
        """
        report_lines = [
            f'# Documentation Analysis Report: {topic}',
            '',
            f"**Generated:** {analysis_results.get('timestamp', 'N/A')}",
            ''
        ]
        
        # Overall Statistics
        overall_stats = analysis_results.get('overall_stats', {})
        if overall_stats:
            report_lines.extend([
                '## Overall Statistics',
                '',
                f"- **Total Documents:** {overall_stats.get('total_documents', 0)}",
                f"- **Average Length:** {overall_stats.get('average_length', 0):.0f} characters",
                f"- **Average Word Count:** {overall_stats.get('average_words', 0):.0f} words",
                f"- **Average Code Blocks:** {overall_stats.get('average_code_blocks', 0):.1f}",
                f"- **Average Links:** {overall_stats.get('average_links', 0):.1f}",
                '',
                '### Section Coverage',
                ''
            ])
            
            for section, coverage in overall_stats.get('section_coverage', {}).items():
                report_lines.append(f"- **{section}:** {coverage['count']}/{overall_stats['total_documents']} ({coverage['percentage']:.0f}%)")
            
            report_lines.append('')
        
        # Document-by-Document Analysis
        document_metrics = analysis_results.get('document_metrics', [])
        if document_metrics:
            report_lines.extend([
                '## Document Analysis',
                ''
            ])
            
            for doc in document_metrics:
                doc_name = Path(doc['path']).name if isinstance(doc['path'], str) else f"Document {doc['index']}"
                report_lines.extend([
                    f"### {doc_name}",
                    '',
                    f"- **Total Words:** {doc['total_words']}",
                    f"- **Code Blocks:** {doc['code_blocks']}",
                    f"- **Links:** {doc['links']}",
                    f"- **Tables:** {doc['tables']}",
                    f"- **Lists:** {doc['lists']}",
                    '',
                    '**Section Scores:**',
                    ''
                ])
                
                for section_name in self.section_headers:
                    section = doc['sections'].get(section_name, {})
                    if section.get('exists', False):
                        report_lines.append(
                            f"- {section_name}: {section['score']:.2f} "
                            f"({section['words']} words, {section['code_blocks']} code blocks)"
                        )
                    else:
                        report_lines.append(f"- {section_name}: *Missing*")
                
                report_lines.append('')
        
        # Comparisons
        comparisons = analysis_results.get('comparisons', {})
        if comparisons and self.include_comparisons:
            report_lines.extend([
                '## Cross-Document Comparisons',
                ''
            ])
            
            if comparisons.get('best_overall'):
                best = comparisons['best_overall']
                report_lines.append(f"**Best Overall:** Document {best['index']} (Score: {best['total_score']:.2f})")
            
            if comparisons.get('worst_overall'):
                worst = comparisons['worst_overall']
                report_lines.append(f"**Worst Overall:** Document {worst['index']} (Score: {worst['total_score']:.2f})")
            
            if comparisons.get('most_comprehensive'):
                most = comparisons['most_comprehensive']
                report_lines.append(f"**Most Comprehensive:** Document {most['index']} ({most['total_words']} words)")
            
            report_lines.append('')
            
            # Best sections
            if comparisons.get('best_sections'):
                report_lines.extend([
                    '### Best Sections by Score',
                    ''
                ])
                
                for section_name, best in comparisons['best_sections'].items():
                    report_lines.append(f"- **{section_name}:** Document {best['doc_index']} (Score: {best['score']:.2f})")
                
                report_lines.append('')
            
            # Consistency analysis
            if comparisons.get('consistency_analysis'):
                report_lines.extend([
                    '### Section Consistency Analysis',
                    '',
                    '| Section | Avg Words | Std Dev | Variability |',
                    '|---------|-----------|---------|-------------|'
                ])
                
                for section_name, stats in comparisons['consistency_analysis'].items():
                    variability = 'Low' if stats['std_dev'] < 50 else 'Medium' if stats['std_dev'] < 150 else 'High'
                    report_lines.append(
                        f"| {section_name} | {stats['average_words']:.0f} | "
                        f"{stats['std_dev']:.0f} | {variability} |"
                    )
                
                report_lines.append('')
        
        return '\n'.join(report_lines)
    
    def save_artifacts(self, results: Dict[str, Any], output_dir: Path, topic: str) -> List[Path]:
        """
        Save analysis reports in configured formats.
        
        Args:
            results: Analysis results
            output_dir: Directory to save artifacts
            topic: The topic used for generation
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        safe_topic = self.sanitize_filename(topic)
        
        # Save markdown report
        if 'markdown' in self.formats:
            report = self.generate_report(results, topic)
            report_path = output_dir / f'{safe_topic}_analysis_report.md'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            saved_files.append(report_path)
            self.logger.info(f"Saved analysis report to {report_path}")
        
        # Save JSON data if requested
        if 'json' in self.formats:
            json_path = output_dir / f'{safe_topic}_analysis_data.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                # Convert Path objects to strings for JSON serialization
                json_safe_results = self._make_json_serializable(results)
                json.dump(json_safe_results, f, indent=2)
            saved_files.append(json_path)
            self.logger.info(f"Saved analysis data to {json_path}")
        
        return saved_files
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert Path objects and other non-serializable types for JSON."""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def get_priority(self) -> int:
        """Reporter should run after compiler."""
        return 80
    
    def get_supported_formats(self) -> List[str]:
        """Return supported output formats."""
        return ['markdown', 'json', 'html']