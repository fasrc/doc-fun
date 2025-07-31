# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2025-07-29

### Added
- **ModuleRecommender System**: Intelligent HPC module recommendations based on topic keywords
  - Smart keyword mapping for different programming languages and tools
  - Relevance scoring based on topic analysis
  - Priority system preferring latest FASRC builds (fasrc02 > fasrc01)
  - Special handling for edge cases (R modules, statistics keywords)
  - Integration with terminology context for automatic module suggestions

- **Enhanced Prompt Templates**: 
  - Parameterized prompt system with {topic}, {format}, {organization} variables
  - Explicit module enforcement instructions to prevent outdated module usage
  - Template support for HTML and Markdown output formats
  - Runtime parameter overrides via --prompt-params flag

- **Code Examples Integration**:
  - Relevant code examples automatically included in documentation context
  - Language detection and filtering (Python, C, Fortran focus)
  - Relevance scoring based on file paths, descriptions, and topic keywords
  - Integration with existing code scanning functionality

- **New Command Examples**: Added comprehensive command_examples.txt with usage patterns

- **Enhanced Testing Suite**:
  - Complete test coverage for ModuleRecommender class (7 new tests)
  - Tests for enhanced DocumentationGenerator functionality (3 new tests)
  - Updated fixtures with comprehensive HPC module data
  - All existing tests maintained and passing (28 total tests)

### Enhanced
- **System Prompt Building**: Now supports parameterized templates with placeholder substitution
- **Terminology Context**: Enhanced with ModuleRecommender integration and code examples
- **Documentation Quality**: LLM now uses exact recommended modules instead of outdated versions
- **Template Structure**: More explicit section header requirements and formatting instructions

### Fixed
- **Section Detection**: Fixed malformatting issues where HTML was wrapped in markdown blocks
- **Module Accuracy**: Eliminated usage of outdated module versions (e.g., python/3.8.5-fasrc01)
- **Analysis Pipeline**: Fixed section analysis that was choosing "no examples" due to structure mismatches
- **File Corruption**: Resolved git restore issues and maintained code integrity
- **Output Directory**: Restored missing --output-dir flag functionality

### Technical Improvements
- **Simplified Language Support**: Focused on Python, C, and Fortran for better accuracy
- **Enhanced Scoring**: Improved relevance algorithms for both modules and code examples
- **Better Error Handling**: More robust handling of edge cases in module detection
- **Code Organization**: Cleaner separation of concerns between classes

### Documentation
- **Updated Command Examples**: Comprehensive usage patterns and workflows
- **Enhanced Test Coverage**: Full documentation of new functionality through tests
- **Improved Code Comments**: Better documentation of complex algorithms

### Breaking Changes
- **Language Support**: Code examples now limited to Python, C, and Fortran (removed problematic language detection)
- **Template Requirements**: Stricter section header requirements may affect custom templates

### Migration Notes
- Existing functionality remains compatible
- New ModuleRecommender requires HPC modules in terminology.yaml
- Enhanced prompt templates provide better output but may change existing generation patterns slightly