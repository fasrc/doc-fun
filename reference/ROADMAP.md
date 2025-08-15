# ROADMAP.md - doc-generator Project

## Project Overview
**doc-generator** is an AI-powered documentation generator that creates high-quality technical documentation using LLM models (OpenAI GPT and Claude). Originally designed for Faculty Arts and Sciences Research Computing (FASRC), it features an extensible plugin architecture for intelligent recommendations and multi-format output support.

## Current Features & Status

### ✅ Completed Features (v2.1.1)

#### Core Documentation Generation
- [x] OpenAI GPT integration with multiple model support (GPT-4, GPT-4o, GPT-5 family)
- [x] Claude API integration (Claude 3 Sonnet, Opus, Haiku)
- [x] Few-shot prompting with curated examples
- [x] Parameterized prompt templates with runtime customization
- [x] Multiple output formats (HTML, Markdown)
- [x] Multi-run generation with temperature variations

#### Plugin Architecture
- [x] Extensible recommendation engine system via entry points
- [x] Built-in ModuleRecommender for HPC module suggestions
- [x] Plugin discovery and management system
- [x] Priority-based plugin ordering
- [x] Analysis plugin framework for post-processing

#### Quality Assurance & Analysis
- [x] Algorithmic document analysis with section scoring
- [x] GPT-based quality evaluation for technical accuracy
- [x] Document compilation from best sections across variants
- [x] Link validation plugin for checking HTTP response codes
- [x] Comprehensive reporting in multiple formats (Markdown, HTML, JSON)

#### Document Comparison & Evaluation
- [x] Compare generated docs with existing documentation (URL or file)
- [x] Content similarity scoring using TF-IDF
- [x] Structural similarity analysis
- [x] Code example comparison
- [x] Composite scoring with detailed recommendations

#### CLI & Development Tools
- [x] Comprehensive CLI with 20+ command options
- [x] Code scanning for automatic example discovery
- [x] Batch processing capabilities
- [x] Model listing and provider auto-detection
- [x] Plugin listing and management commands

### 🚧 In Progress Features

#### README Generator Module (NEW)
- [ ] Generate README.md from code directory structures
- [ ] Depth-adaptive templates (top/mid/leaf levels)
- [ ] Quick Reference tables at each directory level
- [ ] Integration with existing --scan-code functionality

## Modular Architecture

### Core Modules
```
src/doc_generator/
├── core.py                 # DocumentationGenerator, DocumentAnalyzer, GPTQualityEvaluator
├── cli.py                  # Command-line interface and argument parsing
├── plugin_manager.py       # Plugin discovery and management
└── readme_generator.py     # NEW: README generation from code (pending)
```

### Provider System
```
providers/
├── base.py                 # Abstract base provider class
├── manager.py              # Provider auto-detection and management
├── openai_provider.py      # OpenAI GPT models integration
└── claude_provider.py      # Anthropic Claude models integration
```

### Plugin System
```
plugins/
├── base.py                 # RecommendationEngine base class
├── analysis_base.py        # AnalysisPlugin base class
└── modules.py              # Built-in ModuleRecommender plugin
```

### Analysis Modules
```
analysis/
├── compiler.py             # Document compilation from best sections
├── reporter.py             # Multi-format report generation
└── link_validator.py       # HTTP link validation
```

### Evaluator Modules
```
evaluator/
├── comparator.py           # Document comparison and similarity scoring
├── downloader.py           # URL content fetching for comparison
└── metrics.py              # Scoring algorithms and metrics
```

## Git Workflow & Version Control

### Branch Strategy
- **main**: Production-ready releases
- **feature/**: Feature development branches
  - `feature/doc-comparison-module` (merged)
  - `feature/claude-api-integration` (merged)
  - `feature/module-recommender-v1.1` (merged)
  - `feature/readme-generator` (upcoming)

### Commit Convention
When making commits with Claude Code:
```bash
git commit -m "Commit message here

🤖 Generated with Claude Code (https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Release Process
1. Feature development in feature branches
2. PR to main with comprehensive testing
3. Version bump in `src/doc_generator/__init__.py`
4. Update CHANGELOG.md
5. Tag release: `git tag v2.x.x`
6. Push tag: `git push origin v2.x.x`

### Current Version
- **Latest Release**: v2.1.1
- **Python Support**: 3.8+
- **Test Coverage**: 57 tests passing

## Outstanding TODO Items

### Immediate Tasks (README Generator)
- [ ] Create `src/doc_generator/readme_generator.py` module
- [ ] Extend CLI with --generate-readme, --recursive, --overwrite, --suffix flags
- [ ] Implement directory traversal with exclusion patterns
- [ ] Create depth-adaptive README templates
- [ ] Implement code analysis (header extraction, file classification)
- [ ] Build Quick Reference table generation
- [ ] Implement safe file writing with suffix/overwrite options
- [ ] Add GPT integration for enhanced descriptions (optional)
- [ ] Test with User_Codes directory structure

### Future Enhancements
- [ ] Additional recommendation plugins (datasets, workflows, containers)
- [ ] Web UI for documentation generation
- [ ] Real-time preview mode
- [ ] GitHub Actions integration
- [ ] Documentation versioning system
- [ ] Multi-language support (i18n)
- [ ] Custom CSS themes for HTML output
- [ ] Export to additional formats (PDF, EPUB, DocBook)

## Implementation Plan for README Generator

### Phase 1: Core Module Development
**Goal**: Create the foundational ReadmeGenerator module

#### 1.1 Module Setup
- [ ] Create `src/doc_generator/readme_generator.py`
- [ ] Define `ReadmeGenerator` class with initialization parameters
- [ ] Set up logging and configuration handling

#### 1.2 Directory Traversal
- [ ] Implement `scan_directory()` method with exclusion patterns
- [ ] Create recursive traversal logic
- [ ] Build directory tree structure representation

#### 1.3 Code Analysis
- [ ] Implement `analyze_code_file()` for metadata extraction
- [ ] Extract header comments from various languages (Python, Fortran, C, R, MATLAB)
- [ ] Parse Makefiles for compilation instructions
- [ ] Extract job parameters from .sbatch/.slurm files
- [ ] Capture key results from .out/.err files (optional)

### Phase 2: Template System
**Goal**: Implement depth-adaptive README generation

#### 2.1 Depth Detection
- [ ] Implement `determine_depth_level()` method
- [ ] Define criteria for top/mid/leaf level classification
- [ ] Handle edge cases (single file directories, mixed content)

#### 2.2 Content Templates
- [ ] Create top-level template (full sections, categories, overview)
- [ ] Create mid-level template (condensed sections, simple tables)
- [ ] Implement template selection logic

#### 2.3 Quick Reference Tables
- [ ] Implement `generate_quick_reference_table()` method
- [ ] Define column structures for different depth levels:
  - Top: `| Paradigm | Best For | Languages | Examples |`
  - Mid: `| Example | Purpose | Language | Nodes | Output |`
- [ ] Auto-populate table data from code analysis

### Phase 3: CLI Integration
**Goal**: Extend existing CLI with new flags

#### 3.1 CLI Extensions
- [ ] Add `--generate-readme` flag to trigger README generation
- [ ] Add `--recursive` flag for deep traversal
- [ ] Add `--overwrite` flag to replace existing READMEs
- [ ] Add `--suffix` flag for custom output naming

#### 3.2 Integration Points
- [ ] Modify `scan_code_examples()` function in `cli.py`
- [ ] Import and instantiate `ReadmeGenerator` when flags are present
- [ ] Handle output reporting and logging

### Phase 4: File Writing & Safety
**Goal**: Implement safe file operations

#### 4.1 Output Handling
- [ ] Implement `write_readme()` method
- [ ] Default behavior: create `README_generated.md`
- [ ] Overwrite mode: backup existing, write new `README.md`
- [ ] Custom suffix support: `README{suffix}.md`

#### 4.2 Safety Features
- [ ] Check for existing README files
- [ ] Implement backup mechanism
- [ ] Add dry-run option for preview
- [ ] Validate write permissions

### Phase 5: Testing & Refinement
**Goal**: Ensure robust functionality

#### 5.1 Testing
- [ ] Test with `/home/austin/Projects/User_Codes/` structure
- [ ] Verify depth-adaptive templates work correctly
- [ ] Test all language parsers
- [ ] Validate Quick Reference table generation
- [ ] Test edge cases (empty dirs, single files, deeply nested)

#### 5.2 Optimization
- [ ] Add GPT enhancement for descriptions (optional)
- [ ] Implement caching for repeated scans
- [ ] Optimize for large directory trees
- [ ] Add progress indicators for long operations

## Testing Strategy

### Test Coverage
- **Unit Tests**: 57 tests currently passing
- **Test Framework**: pytest with fixtures
- **Mock Strategy**: Mock LLM API calls for predictable testing
- **Test Organization**:
  ```
  tests/
  ├── test_doc_generator.py       # Core functionality tests
  ├── test_plugin_integration.py  # Plugin system tests
  ├── test_plugin_interface.py    # Plugin interface tests
  ├── test_plugin_manager.py      # Plugin management tests
  ├── test_provider_integration.py # Provider integration tests
  └── providers/
      ├── test_base.py
      ├── test_claude_provider.py
      ├── test_manager.py
      └── test_openai_provider.py
  ```

### README Generator Testing Plan
- [ ] Unit tests for each method in ReadmeGenerator class
- [ ] Integration tests with sample directory structures
- [ ] Edge case testing (empty dirs, permission errors, large trees)
- [ ] Template rendering tests for each depth level
- [ ] Table generation accuracy tests

## Configuration Files

### Core Configuration
- **prompt.yaml**: Main prompt templates and few-shot examples
- **terminology.yaml**: Domain-specific terms and HPC modules
- **pyproject.toml**: Package configuration and dependencies
- **pytest.ini**: Test configuration

### Environment Setup
- **.env**: API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY)
- **environment.yml**: Conda environment specification

## Technical Decisions

### Architecture
- **Module-based approach**: Standalone module in `src/doc_generator/readme_generator.py`
- **Integration**: Extends existing `--scan-code` functionality
- **Reuses**: Can leverage existing `DocumentationGenerator` for GPT capabilities

### Design Patterns Used
- **Plugin Architecture**: Entry points for extensibility
- **Provider Pattern**: Abstract base class for LLM providers
- **Factory Pattern**: Provider manager for auto-detection
- **Template Method**: Base classes for plugins define interface
- **Strategy Pattern**: Different analysis and compilation strategies

### Key Design Choices
1. **Input Discovery**: Exclusion-based with smart filters
   - Include: All code files, Makefiles, .sbatch, .out files
   - Exclude: `*.o`, `*.x`, `__pycache__`, `.git`, build artifacts

2. **Analysis Depth**: Surface-level for speed
   - Header comment extraction (first 10 lines)
   - Filename/directory context parsing
   - No deep AST analysis

3. **Output Safety**: Conservative defaults
   - Default: Generate `README_generated.md` alongside existing
   - Opt-in overwrite with `--overwrite` flag
   - Custom suffix with `--suffix` option

## File Structure
```
src/doc_generator/
   readme_generator.py  # New module
   cli.py               # Modified to add new flags
   __init__.py          # Updated to export ReadmeGenerator
```

## Example Usage
```bash
# Basic usage - safe generation
doc-gen --scan-code /path/to/User_Codes --generate-readme --recursive

# Overwrite existing READMEs
doc-gen --scan-code /path/to/User_Codes --generate-readme --recursive --overwrite

# Custom suffix
doc-gen --scan-code /path/to/User_Codes --generate-readme --recursive --suffix "_ai"
```

## Success Metrics
- [ ] Successfully generates READMEs for all directory levels
- [ ] Quick Reference tables accurately reflect directory contents
- [ ] Depth-adaptive templates provide appropriate detail
- [ ] No data loss (existing READMEs preserved by default)
- [ ] Performance: Can process User_Codes tree in < 30 seconds

## Dependencies
- Existing: `pathlib`, `logging`, `yaml`, `beautifulsoup4`
- Optional: GPT API for enhanced descriptions (reuse existing)

## Risk Mitigation
- **Data Loss**: Default to safe generation with `_generated` suffix
- **Performance**: Implement early exclusion patterns to skip non-code files
- **Accuracy**: Use surface-level analysis to avoid misinterpretation
- **Compatibility**: Test with multiple code languages and structures

## Future Enhancements (Post-MVP)
- [ ] Interactive mode for selective generation
- [ ] Diff viewer for comparing generated vs existing
- [ ] Configuration file support for custom templates
- [ ] Integration with git hooks for auto-update
- [ ] Support for additional documentation formats (RST, AsciiDoc)
- [ ] Multi-language README generation
- [ ] Statistics dashboard for code repository

## Development Workflow

### Local Development
```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Run specific test module
pytest tests/test_doc_generator.py -v

# Check code coverage
pytest --cov=doc_generator tests/
```

### Adding New Features
1. Create feature branch: `git checkout -b feature/feature-name`
2. Implement feature with tests
3. Update CHANGELOG.md
4. Create PR to main branch
5. Run full test suite
6. Merge after review

### Documentation Updates
- API docs in `docs/api/`
- User guides in `docs/guides/`
- Examples in `docs/examples/`
- Auto-generated with MkDocs

## Best Practices for Agentic Coding

### Planning & Architecture
- **Always discuss before implementing**: Review architectural decisions through Q&A
- **Create comprehensive roadmaps**: Document all decisions in ROADMAP.md
- **Explore options systematically**: Present multiple approaches with trade-offs
- **Use structured planning**: Break complex tasks into phases

### Code Understanding
- **Read before writing**: Always read existing code before modifications
- **Follow existing patterns**: Maintain consistency with current conventions
- **Check for duplicates**: Ensure functionality doesn't already exist
- **Understand the context**: Review related modules and dependencies

### Safety & Version Control
- **Default to non-destructive**: Use suffixes like `_generated` for new files
- **Feature branches**: Create dedicated branches for new features
- **Proper error handling**: Implement try-catch blocks and validation
- **Confirmation prompts**: Add user confirmations for critical operations
- **Atomic commits**: Make focused commits with clear messages

### Incremental Development
- **Small, testable chunks**: Build features incrementally
- **Phase validation**: Test each phase before proceeding
- **TODO tracking**: Maintain active TODO lists for visibility
- **Regular progress updates**: Keep user informed of progress

### Testing & Validation
- **Test-driven approach**: Write tests alongside implementation
- **Edge case coverage**: Test empty inputs, large datasets, permissions
- **Real data validation**: Test with actual user data structures
- **Mock external dependencies**: Use mocks for API calls in tests

### Documentation
- **Document decisions in ROADMAP.md**: Keep all architectural decisions centralized
- **Minimal inline comments**: Only add code comments when requested
- **Update affected docs**: Maintain README, CHANGELOG, API docs
- **Example usage**: Provide clear examples for new features

### Communication
- **Explicit action descriptions**: Clearly state what will be done
- **Clarification requests**: Ask when requirements are ambiguous
- **Options with trade-offs**: Present alternatives with pros/cons
- **Progress visibility**: Use TODO lists and status updates

### Performance Awareness
- **Consider scale**: Design for large datasets from the start
- **Optimize iteratively**: Profile first, optimize based on data
- **Cache when appropriate**: Implement caching for repeated operations
- **Resource management**: Clean up file handles, connections

## Performance Considerations

### Current Optimizations
- Lazy loading of LLM providers
- Cached prompt templates
- Parallel document analysis
- Efficient HTML parsing with BeautifulSoup

### README Generator Optimizations
- Early filtering of non-code files
- Directory-level caching
- Batch file reading
- Template pre-compilation
- Optional GPT enhancement (not required)

## Dependencies & Requirements

### Core Dependencies
```python
# pyproject.toml key dependencies
python = ">=3.8"
openai = ">=1.0.0"
anthropic = ">=0.18.0"
pyyaml = ">=6.0"
beautifulsoup4 = ">=4.12.0"
requests = ">=2.31.0"
python-dotenv = ">=1.0.0"
```

### Development Dependencies
```python
pytest = ">=7.4.0"
pytest-cov = ">=4.1.0"
black = ">=23.7.0"
mkdocs = ">=1.5.0"
```

## Notes
- Prioritize safety and non-destructive operations
- Focus on navigation/discovery use case (superficial analysis)
- Maintain consistency with existing doc-fun patterns
- Keep README generation separate from HTML doc generation
- Follow existing code style and testing patterns
- Ensure backward compatibility with existing features