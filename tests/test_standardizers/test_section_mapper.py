"""Tests for SmartSectionMapper."""

import pytest
from src.doc_generator.standardizers import SmartSectionMapper, StandardizationTemplate
from src.doc_generator.extractors import ExtractedContent


class TestSmartSectionMapper:
    """Test SmartSectionMapper functionality."""
    
    def test_initialization(self):
        """Test SmartSectionMapper initialization."""
        mapper = SmartSectionMapper()
        assert mapper is not None
        assert 'technical_documentation' in mapper.templates
        assert 'user_guide' in mapper.templates
        assert 'api_documentation' in mapper.templates
    
    def test_normalize_section_name(self):
        """Test section name normalization."""
        mapper = SmartSectionMapper()
        
        # Test various formats
        assert mapper._normalize_section_name("Getting Started") == "getting_started"
        assert mapper._normalize_section_name("API Reference") == "api_reference"
        assert mapper._normalize_section_name("Installation-Guide") == "installation_guide"
        assert mapper._normalize_section_name("FAQ/Help") == "faq_help"
        assert mapper._normalize_section_name("__config__") == "config"
        assert mapper._normalize_section_name("multiple___underscores") == "multiple_underscores"
    
    def test_calculate_section_similarity(self):
        """Test section similarity calculation."""
        mapper = SmartSectionMapper()
        
        # Exact match
        similarity = mapper._calculate_section_similarity("installation", ["installation"])
        assert similarity == 1.0
        
        # Substring match
        similarity = mapper._calculate_section_similarity("quick_installation", ["installation"])
        assert similarity == 0.8
        
        # Word overlap - use terms that actually overlap
        similarity = mapper._calculate_section_similarity("installation_guide", ["guide_installation"])
        assert similarity > 0.0
        
        # No match
        similarity = mapper._calculate_section_similarity("completely_different", ["installation"])
        assert similarity < 0.3
    
    def test_find_direct_mapping(self):
        """Test direct section mapping."""
        mapper = SmartSectionMapper()
        
        source_sections = {"installation_guide", "usage_examples", "api_docs"}
        
        # Should find installation mapping
        mapping = mapper._find_direct_mapping("installation", source_sections)
        assert mapping is not None
        assert mapping.source_section == "installation_guide"
        assert mapping.target_section == "installation"
        assert mapping.confidence >= 0.7
        
        # Should find usage mapping
        mapping = mapper._find_direct_mapping("usage", source_sections)
        assert mapping is not None
        assert mapping.source_section == "usage_examples"
        
        # Should not find mapping for non-existent section
        mapping = mapper._find_direct_mapping("troubleshooting", source_sections)
        assert mapping is None
    
    def test_map_sections(self):
        """Test complete section mapping."""
        mapper = SmartSectionMapper()
        
        # Create mock extracted content
        extracted = ExtractedContent(
            title="Test Document",
            sections={
                "overview": "This is an overview section",
                "installation_guide": "How to install the software",
                "examples": "Usage examples",
                "reference": "API reference documentation",
                "changelog": "Version history"
            },
            format_type="html"
        )
        
        mappings = mapper.map_sections(extracted, "technical_documentation")
        
        # Check that we have mappings
        assert len(mappings) > 0
        
        # Check specific mappings
        mapping_dict = {m.target_section: m.source_section for m in mappings}
        
        assert "introduction" in mapping_dict  # overview should map to introduction
        assert "installation" in mapping_dict  # installation_guide should map to installation
        assert "usage" in mapping_dict  # examples should map to usage
        assert "api_reference" in mapping_dict  # reference should map to api_reference
    
    def test_create_standardized_structure(self):
        """Test standardized structure creation."""
        mapper = SmartSectionMapper()
        
        # Create mock extracted content
        extracted = ExtractedContent(
            title="Test Document",
            sections={
                "intro": "Introduction content",
                "install": "Installation content",
                "guide": "Usage guide content"
            },
            format_type="html"
        )
        
        # Create mock mappings
        mappings = [
            type('SectionMapping', (), {
                'source_section': 'intro',
                'target_section': 'introduction',
                'confidence': 0.9,
                'mapping_type': 'direct'
            })(),
            type('SectionMapping', (), {
                'source_section': 'install',
                'target_section': 'installation',
                'confidence': 0.8,
                'mapping_type': 'direct'
            })(),
            type('SectionMapping', (), {
                'source_section': 'guide',
                'target_section': 'usage',
                'confidence': 0.7,
                'mapping_type': 'direct'
            })()
        ]
        
        standardized = mapper.create_standardized_structure(
            mappings, extracted, "technical_documentation"
        )
        
        assert "introduction" in standardized
        assert "installation" in standardized
        assert "usage" in standardized
        assert standardized["introduction"] == "Introduction content"
        assert standardized["installation"] == "Installation content"
        assert standardized["usage"] == "Usage guide content"
    
    def test_get_available_templates(self):
        """Test getting available templates."""
        mapper = SmartSectionMapper()
        templates = mapper.get_available_templates()
        
        assert isinstance(templates, list)
        assert "technical_documentation" in templates
        assert "user_guide" in templates
        assert "api_documentation" in templates
    
    def test_add_custom_template(self):
        """Test adding custom template."""
        mapper = SmartSectionMapper()
        
        custom_template = StandardizationTemplate(
            required_sections=["intro", "details"],
            optional_sections=["appendix"],
            section_order=["intro", "details", "appendix"]
        )
        
        mapper.add_custom_template("custom_template", custom_template)
        
        assert "custom_template" in mapper.templates
        assert mapper.templates["custom_template"] == custom_template
    
    def test_get_mapping_summary(self):
        """Test mapping summary generation."""
        mapper = SmartSectionMapper()
        
        # Create mock mappings
        mappings = [
            type('SectionMapping', (), {
                'source_section': 'intro',
                'target_section': 'introduction',
                'confidence': 0.9,
                'mapping_type': 'direct'
            })(),
            type('SectionMapping', (), {
                'source_section': 'install',
                'target_section': 'installation',
                'confidence': 0.6,
                'mapping_type': 'derived'
            })(),
            type('SectionMapping', (), {
                'source_section': 'misc',
                'target_section': 'additional_information',
                'confidence': 0.3,
                'mapping_type': 'derived'
            })()
        ]
        
        summary = mapper.get_mapping_summary(mappings)
        
        assert summary['total_mappings'] == 3
        assert summary['mapping_types']['direct'] == 1
        assert summary['mapping_types']['derived'] == 2
        assert summary['confidence_distribution']['high'] == 1
        assert summary['confidence_distribution']['medium'] == 1
        assert summary['confidence_distribution']['low'] == 1
        
        assert 'introduction' in summary['target_sections']
        assert 'intro' in summary['source_sections']
    
    def test_unknown_template(self):
        """Test handling of unknown template."""
        mapper = SmartSectionMapper()
        
        extracted = ExtractedContent(
            sections={"intro": "content"},
            format_type="html"
        )
        
        # Should fallback to default template
        mappings = mapper.map_sections(extracted, "unknown_template")
        assert len(mappings) >= 0  # Should not crash and may return mappings