"""Compare and evaluate generated documentation against existing documentation."""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
from bs4 import BeautifulSoup

from .downloader import DocumentationDownloader
from .metrics import SimilarityMetrics
from ..core import DocumentationGenerator, DocumentAnalyzer


class DocumentationComparator:
    """Compare generated documentation with existing documentation."""
    
    def __init__(self, generator: Optional[DocumentationGenerator] = None):
        """Initialize the comparator.
        
        Args:
            generator: Optional DocumentationGenerator instance
        """
        self.generator = generator or DocumentationGenerator()
        self.downloader = DocumentationDownloader()
        self.metrics = SimilarityMetrics()
        self.analyzer = DocumentAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def compare_with_url(
        self, 
        topic: str, 
        reference_url: str, 
        generation_params: Optional[Dict] = None
    ) -> Dict:
        """Compare generated documentation with an existing documentation page.
        
        Args:
            topic: Topic to generate documentation for
            reference_url: URL of existing documentation to compare against
            generation_params: Optional parameters for documentation generation
            
        Returns:
            Comparison results dictionary
        """
        # Download and extract reference documentation
        self.logger.info(f"Downloading reference documentation from {reference_url}")
        reference_doc = self.downloader.download_and_extract(reference_url)
        
        # Generate documentation
        self.logger.info(f"Generating documentation for topic: {topic}")
        gen_params = generation_params or {
            'runs': 1,
            'model': 'gpt-4o-mini',
            'temperature': 0.3
        }
        
        generated_files = self.generator.generate_documentation(
            query=topic,
            **gen_params
        )
        
        if not generated_files:
            raise ValueError("Failed to generate documentation")
        
        # Load and extract generated documentation
        generated_path = Path(generated_files[0])
        with open(generated_path, 'r', encoding='utf-8') as f:
            generated_html = f.read()
        
        generated_doc = self._extract_generated_content(generated_html)
        
        # Perform comparison
        comparison_results = self._compare_documents(reference_doc, generated_doc)
        
        # Add metadata
        comparison_results['metadata'] = {
            'topic': topic,
            'reference_url': reference_url,
            'generated_file': str(generated_path),
            'generation_params': gen_params,
            'comparison_date': datetime.now().isoformat()
        }
        
        return comparison_results
    
    def compare_existing_files(
        self, 
        generated_file: str, 
        reference_file: str
    ) -> Dict:
        """Compare existing generated file with a reference file.
        
        Args:
            generated_file: Path to generated documentation
            reference_file: Path to reference documentation
            
        Returns:
            Comparison results dictionary
        """
        # Load generated file
        with open(generated_file, 'r', encoding='utf-8') as f:
            generated_html = f.read()
        generated_doc = self._extract_generated_content(generated_html)
        
        # Load reference file
        with open(reference_file, 'r', encoding='utf-8') as f:
            reference_html = f.read()
        
        # Extract content from reference
        if reference_file.endswith('.html'):
            reference_doc = self.downloader.extract_content(reference_html)
        else:
            # Assume it's plain text/markdown
            reference_doc = {
                'raw_text': reference_html,
                'sections': [],
                'code_examples': [],
                'metadata': {}
            }
        
        # Perform comparison
        return self._compare_documents(reference_doc, generated_doc)
    
    def _extract_generated_content(self, html: str) -> Dict:
        """Extract content from generated HTML documentation."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract sections using DocumentAnalyzer
        sections_dict = self.analyzer.extract_sections(html)
        
        # Convert to list format
        sections = []
        for title, content in sections_dict.items():
            sections.append({
                'title': title,
                'content': BeautifulSoup(content, 'html.parser').get_text(strip=True),
                'level': 2  # Assume h2 for standard sections
            })
        
        # Extract code examples
        code_examples = []
        for code_block in soup.find_all(['pre', 'code']):
            code_examples.append({
                'code': code_block.get_text(strip=True),
                'language': self._detect_language(code_block)
            })
        
        # Extract metadata
        metadata = {
            'title': soup.title.get_text() if soup.title else '',
            'description': ''
        }
        
        # Get raw text
        raw_text = soup.get_text(separator='\n', strip=True)
        
        return {
            'raw_text': raw_text,
            'sections': sections,
            'code_examples': code_examples,
            'metadata': metadata,
            'content': soup.body if soup.body else soup
        }
    
    def _detect_language(self, code_element) -> str:
        """Detect programming language from code element."""
        classes = code_element.get('class', [])
        for cls in classes:
            if 'language-' in cls:
                return cls.replace('language-', '')
        return ''
    
    def _compare_documents(self, reference: Dict, generated: Dict) -> Dict:
        """Perform detailed comparison between two documents."""
        results = {
            'scores': {},
            'details': {},
            'recommendations': []
        }
        
        # Content similarity
        content_sim = self.metrics.sequence_similarity(
            reference.get('raw_text', ''),
            generated.get('raw_text', '')
        )
        results['scores']['content_similarity'] = content_sim
        
        # Structural similarity
        structural_sim = self.metrics.structural_similarity(
            reference.get('sections', []),
            generated.get('sections', [])
        )
        results['scores']['structural_similarity'] = structural_sim
        
        # Code similarity
        code_sim = self.metrics.code_similarity(
            reference.get('code_examples', []),
            generated.get('code_examples', [])
        )
        results['scores']['code_similarity'] = code_sim
        
        # Semantic similarity
        semantic_sim = self.metrics.semantic_similarity(
            reference.get('raw_text', ''),
            generated.get('raw_text', '')
        )
        results['scores']['semantic_similarity'] = semantic_sim
        
        # Jaccard similarity
        jaccard_sim = self.metrics.jaccard_similarity(
            reference.get('raw_text', ''),
            generated.get('raw_text', '')
        )
        results['scores']['jaccard_similarity'] = jaccard_sim
        
        # Cosine similarity
        cosine_sim = self.metrics.cosine_similarity(
            reference.get('raw_text', ''),
            generated.get('raw_text', '')
        )
        results['scores']['cosine_similarity'] = cosine_sim
        
        # Composite score
        composite_score = self.metrics.calculate_composite_score(
            content_sim, structural_sim, code_sim, semantic_sim
        )
        results['scores']['composite_score'] = composite_score
        
        # Detailed analysis
        results['details'] = self._analyze_differences(reference, generated)
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(
            results['scores'], 
            results['details']
        )
        
        return results
    
    def _analyze_differences(self, reference: Dict, generated: Dict) -> Dict:
        """Analyze specific differences between documents."""
        details = {}
        
        # Section comparison
        ref_sections = {s['title'].lower() for s in reference.get('sections', [])}
        gen_sections = {s['title'].lower() for s in generated.get('sections', [])}
        
        details['missing_sections'] = list(ref_sections - gen_sections)
        details['extra_sections'] = list(gen_sections - ref_sections)
        details['common_sections'] = list(ref_sections & gen_sections)
        
        # Code example comparison
        details['reference_code_count'] = len(reference.get('code_examples', []))
        details['generated_code_count'] = len(generated.get('code_examples', []))
        
        # Length comparison
        ref_length = len(reference.get('raw_text', ''))
        gen_length = len(generated.get('raw_text', ''))
        details['reference_length'] = ref_length
        details['generated_length'] = gen_length
        details['length_ratio'] = gen_length / ref_length if ref_length > 0 else 0
        
        # Language analysis for code
        ref_languages = {ex.get('language', 'unknown') for ex in reference.get('code_examples', [])}
        gen_languages = {ex.get('language', 'unknown') for ex in generated.get('code_examples', [])}
        details['reference_languages'] = list(ref_languages)
        details['generated_languages'] = list(gen_languages)
        
        return details
    
    def _generate_recommendations(self, scores: Dict, details: Dict) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []
        
        # Content similarity recommendations
        if scores['content_similarity'] < 0.3:
            recommendations.append(
                "Low content similarity - consider adjusting prompts to better match reference style"
            )
        elif scores['content_similarity'] > 0.9:
            recommendations.append(
                "Very high content similarity - generated content may be too similar to reference"
            )
        
        # Structural recommendations
        if scores['structural_similarity'] < 0.5:
            recommendations.append(
                "Low structural similarity - review section organization and hierarchy"
            )
        
        if details['missing_sections']:
            recommendations.append(
                f"Missing sections: {', '.join(details['missing_sections'][:3])} - consider adding these"
            )
        
        # Code recommendations
        if details['generated_code_count'] == 0 and details['reference_code_count'] > 0:
            recommendations.append(
                "No code examples generated - consider adding code examples to match reference"
            )
        elif details['generated_code_count'] < details['reference_code_count'] * 0.5:
            recommendations.append(
                "Fewer code examples than reference - consider adding more examples"
            )
        
        # Length recommendations
        if details['length_ratio'] < 0.5:
            recommendations.append(
                "Generated content is much shorter than reference - consider expanding content"
            )
        elif details['length_ratio'] > 2.0:
            recommendations.append(
                "Generated content is much longer than reference - consider being more concise"
            )
        
        # Semantic recommendations
        if scores['semantic_similarity'] < 0.4:
            recommendations.append(
                "Low semantic similarity - keywords and concepts differ significantly from reference"
            )
        
        # Overall recommendation
        if scores['composite_score'] < 0.4:
            recommendations.append(
                "Overall low similarity - consider significant adjustments to generation parameters"
            )
        elif scores['composite_score'] > 0.8:
            recommendations.append(
                "Excellent similarity - generated documentation closely matches reference quality"
            )
        else:
            recommendations.append(
                "Good similarity - minor adjustments could improve alignment with reference"
            )
        
        return recommendations
    
    def generate_report(self, comparison_results: Dict, output_file: Optional[str] = None) -> str:
        """Generate a detailed comparison report.
        
        Args:
            comparison_results: Results from compare_with_url or compare_existing_files
            output_file: Optional file path to save the report
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("# Documentation Comparison Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Metadata
        if 'metadata' in comparison_results:
            report.append("\n## Metadata")
            for key, value in comparison_results['metadata'].items():
                report.append(f"- **{key}**: {value}")
        
        # Scores
        report.append("\n## Similarity Scores")
        scores = comparison_results.get('scores', {})
        report.append(f"- **Composite Score**: {scores.get('composite_score', 0):.2%}")
        report.append(f"- **Content Similarity**: {scores.get('content_similarity', 0):.2%}")
        report.append(f"- **Structural Similarity**: {scores.get('structural_similarity', 0):.2%}")
        report.append(f"- **Code Similarity**: {scores.get('code_similarity', 0):.2%}")
        report.append(f"- **Semantic Similarity**: {scores.get('semantic_similarity', 0):.2%}")
        report.append(f"- **Jaccard Similarity**: {scores.get('jaccard_similarity', 0):.2%}")
        report.append(f"- **Cosine Similarity**: {scores.get('cosine_similarity', 0):.2%}")
        
        # Details
        report.append("\n## Detailed Analysis")
        details = comparison_results.get('details', {})
        
        report.append("\n### Section Analysis")
        if details.get('missing_sections'):
            report.append(f"**Missing Sections**: {', '.join(details['missing_sections'])}")
        if details.get('extra_sections'):
            report.append(f"**Extra Sections**: {', '.join(details['extra_sections'])}")
        if details.get('common_sections'):
            report.append(f"**Common Sections**: {', '.join(details['common_sections'])}")
        
        report.append("\n### Content Metrics")
        report.append(f"- **Reference Length**: {details.get('reference_length', 0):,} characters")
        report.append(f"- **Generated Length**: {details.get('generated_length', 0):,} characters")
        report.append(f"- **Length Ratio**: {details.get('length_ratio', 0):.2f}")
        
        report.append("\n### Code Examples")
        report.append(f"- **Reference Code Examples**: {details.get('reference_code_count', 0)}")
        report.append(f"- **Generated Code Examples**: {details.get('generated_code_count', 0)}")
        if details.get('reference_languages'):
            report.append(f"- **Reference Languages**: {', '.join(details['reference_languages'])}")
        if details.get('generated_languages'):
            report.append(f"- **Generated Languages**: {', '.join(details['generated_languages'])}")
        
        # Recommendations
        report.append("\n## Recommendations")
        for i, rec in enumerate(comparison_results.get('recommendations', []), 1):
            report.append(f"{i}. {rec}")
        
        # Quality Assessment
        report.append("\n## Quality Assessment")
        composite = scores.get('composite_score', 0)
        if composite >= 0.8:
            assessment = "⭐⭐⭐⭐⭐ Excellent"
        elif composite >= 0.6:
            assessment = "⭐⭐⭐⭐ Good"
        elif composite >= 0.4:
            assessment = "⭐⭐⭐ Acceptable"
        elif composite >= 0.2:
            assessment = "⭐⭐ Needs Improvement"
        else:
            assessment = "⭐ Poor"
        report.append(f"Overall Quality: {assessment}")
        
        report_text = '\n'.join(report)
        
        # Save if output file specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            self.logger.info(f"Report saved to {output_file}")
        
        return report_text