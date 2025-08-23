"""Smart section mapping for document standardization."""

from typing import Dict, List, Optional, Set, Tuple, Any
import re
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from ..extractors import ExtractedContent, ContentSection


@dataclass
class SectionMapping:
    """Represents a mapping between source and target sections."""
    
    source_section: str
    target_section: str
    confidence: float
    mapping_type: str = "direct"  # direct, merged, split, derived
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StandardizationTemplate:
    """Template for standardized document structure."""
    
    required_sections: List[str]
    optional_sections: List[str] = field(default_factory=list)
    section_order: List[str] = field(default_factory=list)
    section_descriptions: Dict[str, str] = field(default_factory=dict)
    merge_patterns: Dict[str, List[str]] = field(default_factory=dict)


class SmartSectionMapper:
    """Intelligent section mapping for document standardization.
    
    Maps content from various document formats to standardized section structures
    while preserving information and improving organization.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common section patterns
        self.section_patterns = {
            'introduction': [
                'intro', 'introduction', 'overview', 'about', 'getting_started',
                'what_is', 'summary', 'description', 'purpose'
            ],
            'installation': [
                'install', 'installation', 'setup', 'getting_started', 'quickstart',
                'requirements', 'dependencies', 'prerequisites'
            ],
            'usage': [
                'usage', 'how_to_use', 'examples', 'tutorial', 'guide', 'demo',
                'basic_usage', 'quick_start', 'getting_started'
            ],
            'api_reference': [
                'api', 'reference', 'documentation', 'methods', 'functions',
                'classes', 'endpoints', 'commands'
            ],
            'configuration': [
                'config', 'configuration', 'settings', 'options', 'parameters',
                'customization', 'preferences'
            ],
            'troubleshooting': [
                'troubleshooting', 'faq', 'problems', 'issues', 'errors',
                'debugging', 'common_problems', 'known_issues'
            ],
            'contributing': [
                'contributing', 'development', 'contribute', 'developers',
                'building', 'testing', 'guidelines'
            ],
            'changelog': [
                'changelog', 'history', 'releases', 'versions', 'updates',
                'changes', 'news', 'whats_new'
            ],
            'license': [
                'license', 'licensing', 'copyright', 'legal', 'terms'
            ]
        }
        
        # Standard templates
        self.templates = {
            'technical_documentation': StandardizationTemplate(
                required_sections=['introduction', 'installation', 'usage'],
                optional_sections=['configuration', 'api_reference', 'troubleshooting', 
                                 'contributing', 'changelog', 'license'],
                section_order=['introduction', 'installation', 'usage', 'configuration',
                             'api_reference', 'troubleshooting', 'contributing', 'license'],
                section_descriptions={
                    'introduction': 'Overview and purpose of the project',
                    'installation': 'How to install and set up the project',
                    'usage': 'Basic usage examples and tutorials',
                    'configuration': 'Configuration options and settings',
                    'api_reference': 'Detailed API documentation',
                    'troubleshooting': 'Common issues and solutions',
                    'contributing': 'Guidelines for contributors',
                    'license': 'License and legal information'
                }
            ),
            'user_guide': StandardizationTemplate(
                required_sections=['introduction', 'getting_started'],
                optional_sections=['advanced_usage', 'configuration', 'troubleshooting',
                                 'faq', 'support'],
                section_order=['introduction', 'getting_started', 'advanced_usage',
                             'configuration', 'troubleshooting', 'faq', 'support']
            ),
            'api_documentation': StandardizationTemplate(
                required_sections=['introduction', 'api_reference'],
                optional_sections=['authentication', 'examples', 'error_codes',
                                 'rate_limits', 'changelog'],
                section_order=['introduction', 'authentication', 'api_reference',
                             'examples', 'error_codes', 'rate_limits', 'changelog']
            )
        }
    
    def map_sections(self, extracted_content: ExtractedContent, 
                    template_name: str = 'technical_documentation') -> List[SectionMapping]:
        """Map extracted sections to standardized structure.
        
        Args:
            extracted_content: Extracted document content
            template_name: Name of standardization template to use
            
        Returns:
            List of section mappings
        """
        if template_name not in self.templates:
            self.logger.warning(f"Unknown template '{template_name}', using default")
            template_name = 'technical_documentation'
        
        template = self.templates[template_name]
        mappings = []
        
        # Get source sections
        source_sections = set(extracted_content.sections.keys())
        unmapped_sections = source_sections.copy()
        
        # Direct mappings first
        for target_section in template.required_sections + template.optional_sections:
            mapping = self._find_direct_mapping(target_section, source_sections)
            if mapping:
                mappings.append(mapping)
                unmapped_sections.discard(mapping.source_section)
        
        # Handle unmapped sections
        for section in unmapped_sections:
            mapping = self._find_best_mapping(section, template)
            if mapping:
                mappings.append(mapping)
        
        # Create derived sections if needed
        derived_mappings = self._create_derived_sections(
            extracted_content, template, mappings
        )
        mappings.extend(derived_mappings)
        
        return mappings
    
    def _find_direct_mapping(self, target_section: str, 
                           source_sections: Set[str]) -> Optional[SectionMapping]:
        """Find direct mapping between source and target sections."""
        target_patterns = self.section_patterns.get(target_section, [target_section])
        
        best_match = None
        best_confidence = 0.0
        
        for source_section in source_sections:
            confidence = self._calculate_section_similarity(
                source_section, target_patterns
            )
            
            if confidence > best_confidence and confidence >= 0.7:
                best_confidence = confidence
                best_match = source_section
        
        if best_match:
            return SectionMapping(
                source_section=best_match,
                target_section=target_section,
                confidence=best_confidence,
                mapping_type="direct"
            )
        
        return None
    
    def _find_best_mapping(self, source_section: str, 
                          template: StandardizationTemplate) -> Optional[SectionMapping]:
        """Find the best mapping for an unmapped source section."""
        all_targets = template.required_sections + template.optional_sections
        
        best_match = None
        best_confidence = 0.0
        
        for target_section in all_targets:
            target_patterns = self.section_patterns.get(target_section, [target_section])
            confidence = self._calculate_section_similarity(
                source_section, target_patterns
            )
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = target_section
        
        # Only create mapping if confidence is reasonable
        if best_match and best_confidence >= 0.5:
            return SectionMapping(
                source_section=source_section,
                target_section=best_match,
                confidence=best_confidence,
                mapping_type="derived"
            )
        
        # If no good match, suggest a generic section
        return SectionMapping(
            source_section=source_section,
            target_section="additional_information",
            confidence=0.3,
            mapping_type="derived",
            metadata={"original_title": source_section}
        )
    
    def _calculate_section_similarity(self, source_section: str, 
                                    target_patterns: List[str]) -> float:
        """Calculate similarity between source section and target patterns."""
        source_normalized = self._normalize_section_name(source_section)
        
        max_similarity = 0.0
        
        for pattern in target_patterns:
            pattern_normalized = self._normalize_section_name(pattern)
            
            # Exact match
            if source_normalized == pattern_normalized:
                return 1.0
            
            # Substring match
            if pattern_normalized in source_normalized or source_normalized in pattern_normalized:
                similarity = 0.8
            else:
                # Word overlap similarity
                source_words = set(source_normalized.split('_'))
                pattern_words = set(pattern_normalized.split('_'))
                
                if source_words and pattern_words:
                    overlap = len(source_words.intersection(pattern_words))
                    total = len(source_words.union(pattern_words))
                    similarity = overlap / total * 0.7
                else:
                    similarity = 0.0
            
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _normalize_section_name(self, section_name: str) -> str:
        """Normalize section name for comparison."""
        # Convert to lowercase
        normalized = section_name.lower()
        
        # Replace various separators with underscore
        normalized = re.sub(r'[-\s\.\/\\]+', '_', normalized)
        
        # Remove special characters
        normalized = re.sub(r'[^\w_]', '', normalized)
        
        # Remove redundant underscores
        normalized = re.sub(r'_+', '_', normalized)
        
        # Strip leading/trailing underscores
        normalized = normalized.strip('_')
        
        return normalized
    
    def _create_derived_sections(self, extracted_content: ExtractedContent,
                               template: StandardizationTemplate,
                               existing_mappings: List[SectionMapping]) -> List[SectionMapping]:
        """Create derived sections based on content analysis."""
        derived_mappings = []
        mapped_targets = {m.target_section for m in existing_mappings}
        
        # Check if we need to create essential sections
        for required_section in template.required_sections:
            if required_section not in mapped_targets:
                # Try to derive from title or content
                if required_section == 'introduction' and extracted_content.title:
                    derived_mappings.append(SectionMapping(
                        source_section="__derived_from_title__",
                        target_section=required_section,
                        confidence=0.6,
                        mapping_type="derived",
                        metadata={"source": "title", "title": extracted_content.title}
                    ))
        
        return derived_mappings
    
    def create_standardized_structure(self, mappings: List[SectionMapping],
                                    extracted_content: ExtractedContent,
                                    template_name: str = 'technical_documentation') -> Dict[str, str]:
        """Create standardized document structure from mappings.
        
        Args:
            mappings: Section mappings
            extracted_content: Original extracted content
            template_name: Template name for ordering
            
        Returns:
            Dict of standardized sections
        """
        template = self.templates.get(template_name)
        if not template:
            template = self.templates['technical_documentation']
        
        # Group mappings by target section
        target_to_sources = defaultdict(list)
        for mapping in mappings:
            target_to_sources[mapping.target_section].append(mapping)
        
        standardized_sections = {}
        
        # Process sections in template order
        for target_section in template.section_order:
            if target_section in target_to_sources:
                content_parts = []
                
                for mapping in target_to_sources[target_section]:
                    if mapping.source_section == "__derived_from_title__":
                        # Handle title-derived content
                        content_parts.append(f"# {extracted_content.title}\n")
                    elif mapping.source_section in extracted_content.sections:
                        content_parts.append(extracted_content.sections[mapping.source_section])
                
                if content_parts:
                    standardized_sections[target_section] = '\n\n'.join(content_parts)
        
        # Add any unmapped target sections that have mappings
        for target_section, source_mappings in target_to_sources.items():
            if target_section not in standardized_sections:
                content_parts = []
                for mapping in source_mappings:
                    if mapping.source_section in extracted_content.sections:
                        content_parts.append(extracted_content.sections[mapping.source_section])
                
                if content_parts:
                    standardized_sections[target_section] = '\n\n'.join(content_parts)
        
        return standardized_sections
    
    def get_available_templates(self) -> List[str]:
        """Get list of available standardization templates."""
        return list(self.templates.keys())
    
    def add_custom_template(self, name: str, template: StandardizationTemplate):
        """Add a custom standardization template."""
        self.templates[name] = template
        self.logger.info(f"Added custom template: {name}")
    
    def get_mapping_summary(self, mappings: List[SectionMapping]) -> Dict[str, Any]:
        """Get summary of section mappings."""
        summary = {
            'total_mappings': len(mappings),
            'mapping_types': defaultdict(int),
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'target_sections': set(),
            'source_sections': set()
        }
        
        for mapping in mappings:
            summary['mapping_types'][mapping.mapping_type] += 1
            summary['target_sections'].add(mapping.target_section)
            summary['source_sections'].add(mapping.source_section)
            
            if mapping.confidence >= 0.8:
                summary['confidence_distribution']['high'] += 1
            elif mapping.confidence >= 0.5:
                summary['confidence_distribution']['medium'] += 1
            else:
                summary['confidence_distribution']['low'] += 1
        
        # Convert sets to lists for JSON serialization
        summary['target_sections'] = list(summary['target_sections'])
        summary['source_sections'] = list(summary['source_sections'])
        summary['mapping_types'] = dict(summary['mapping_types'])
        
        return summary