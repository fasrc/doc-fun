# CLI API

The command-line interface provides comprehensive access to all doc-generator features, including documentation generation, README creation, document standardization, and analysis capabilities.

## CLI Module

::: doc_generator.cli
    options:
      show_source: true
      show_root_heading: true
      show_root_members_full_path: false
      show_category_heading: true
      heading_level: 3

## Operation Modes

Doc-generator supports four main operation modes:

1. **Topic Mode** (`--topic`): Generate HTML/Markdown documentation for specific topics
2. **README Mode** (`--readme`): Generate README.md files for code directories
3. **Standardization Mode** (`--standardize`): Standardize existing documentation to organizational templates
4. **Legacy Code Scanning** (`--scan-code`): Scan directories for code examples

## Main Operation Arguments

### Core Operation Modes

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--topic` | `str` | Generate HTML/Markdown documentation for specific topic | None |
| `--readme` | `str` | Generate README.md for directory using unified pipeline | None |
| `--standardize` | `str` | Standardize existing documentation file to organizational standards | None |
| `--scan-code` | `str` | Scan directory for code examples and update terminology | None |

### Shared Generation Options

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--runs` | `int` | Number of documentation variants to generate | `1` (README: `3`) |
| `--model` | `str` | Model to use (e.g., gpt-4o-mini, claude-3-5-sonnet) | Provider-specific |
| `--provider` | `str` | LLM provider: openai, claude, auto | `auto` |
| `--temperature` | `float` | Generation temperature (0.0-1.0) | `0.3` |
| `--output-dir` | `str` | Output directory for generated files | `output` |

### Topic Mode Options

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--format` | `str` | Output format: html, markdown, auto | `auto` |
| `--prompt-yaml-path` | `str` | Path to prompt YAML configuration | `prompts/generator/default.yaml` |
| `--quality-eval` | `flag` | Run GPT-based quality evaluation | `False` |
| `--compare-url` | `str` | Compare with existing documentation at URL | None |
| `--compare-file` | `str` | Compare with local documentation file | None |
| `--comparison-report` | `str` | Save comparison report to specified file | None |

### README Mode Options

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--recursive` | `flag` | Generate README files for all subdirectories | `False` |

### Standardization Options

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--template` | `str` | Template: technical_documentation, user_guide, api_documentation | `technical_documentation` |
| `--target-format` | `str` | Target format for standardized output: html, markdown | Inherits from `--format` |

### Legacy Code Scanning Options

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--generate-readme` | `flag` | Generate README files (legacy mode - use --readme instead) | `False` |
| `--max-scan-files` | `int` | Maximum files to scan for code examples | `50` |
| `--overwrite` | `flag` | Overwrite existing README.md files | `False` |
| `--suffix` | `str` | Custom suffix for generated README files | `_generated` |
| `--ai-enhance` | `flag` | Use AI to enhance README descriptions | `False` |

### Analysis and Configuration

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--analyze` | `flag` | Run document analysis after generation | `False` |
| `--analysis-prompt-path` | `str` | Path to analysis prompt configuration | `prompts/analysis/default.yaml` |
| `--report-format` | `str` | Format for analysis reports: markdown, html, json | `markdown` |
| `--shots` | `str` | Path to few-shot examples directory | None |
| `--examples-dir` | `str` | Directory containing few-shot examples | `shots` |
| `--terminology-path` | `str` | Path to terminology YAML file | `terminology.yaml` |

### Plugin Management

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--list-plugins` | `flag` | List all available plugins and exit | `False` |
| `--disable-plugins` | `list` | Disable specific plugins by name | `None` |
| `--enable-only` | `list` | Enable only specified plugins (disable all others) | `None` |

### Utility Commands

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--list-models` | `flag` | List available models and providers | `False` |
| `--cleanup` | `flag` | Remove all files and directories in ./output/ | `False` |
| `--info` | `flag` | Display detailed information about all options | `False` |
| `--verbose` | `flag` | Enable verbose logging output | `False` |
| `--quiet` | `flag` | Suppress non-essential output | `False` |
| `--version` | `flag` | Show package version | `False` |

## Usage Examples

### Topic Mode - Documentation Generation

```bash
# Basic topic documentation
doc-gen --topic "Python Programming" --output-dir ./docs

# Advanced generation with analysis
doc-gen \
  --topic "Machine Learning with GPU" \
  --runs 3 \
  --model gpt-4 \
  --temperature 0.5 \
  --analyze \
  --quality-eval \
  --verbose

# Markdown output with custom prompts
doc-gen \
  --topic "API Documentation" \
  --format markdown \
  --prompt-yaml-path ./prompts/custom/api-docs.yaml \
  --output-dir ./api-docs

# Compare with existing documentation
doc-gen \
  --topic "User Guide" \
  --compare-url https://existing-docs.com/guide \
  --comparison-report ./comparison.md
```

### README Mode - Directory Documentation

```bash
# Generate README for single directory
doc-gen --readme /path/to/project --runs 2 --analyze

# Recursive README generation
doc-gen --readme /path/to/project --recursive --output-dir ./output

# README with specific model
doc-gen --readme /path/to/project --model claude-3-5-sonnet
```

### Standardization Mode - Document Standardization

```bash
# Standardize HTML to Markdown
doc-gen --standardize existing-docs.html --target-format markdown

# Use specific template
doc-gen --standardize /path/to/doc.html --template api_documentation

# Standardize with custom output location
doc-gen --standardize legacy.md --output-dir ./standardized
```

### Legacy Code Scanning

```bash
# Scan directory for code examples
doc-gen --scan-code ./directory --generate-readme --recursive

# Scan with file limits
doc-gen --scan-code ./large-project --max-scan-files 100
```

### Plugin Management

```bash
# List available plugins
doc-gen --list-plugins

# Disable specific plugins
doc-gen --topic "Topic" --disable-plugins modules datasets

# Enable only specific plugins  
doc-gen --topic "Topic" --enable-only workflows templates
```

### Utility Operations

```bash
# List available models and providers
doc-gen --list-models

# Clean up output directory
doc-gen --cleanup

# Display detailed help information
doc-gen --info
```

## Provider and Model Support

### Available Providers

- **OpenAI**: GPT-3.5, GPT-4, GPT-4o model families
- **Anthropic**: Claude 3 (Haiku, Sonnet, Opus) models
- **Auto-detection**: Automatically selects provider based on available API keys

### Model Selection Examples

```bash
# OpenAI models
doc-gen --topic "Topic" --provider openai --model gpt-4o-mini
doc-gen --topic "Topic" --provider openai --model gpt-4

# Claude models
doc-gen --topic "Topic" --provider claude --model claude-3-5-sonnet-20240620
doc-gen --topic "Topic" --provider claude --model claude-3-haiku-20240307

# Auto-detection (default)
doc-gen --topic "Topic" --provider auto
```

## Output Structure

Generated files follow consistent naming patterns:

### Topic Mode
```
{topic}_{provider}_{model}_temp{temperature}_v{number}.{format}
Example: python_programming_openai_gpt4omini_temp03_v1.html
```

### README Mode
```
{directory}_readme_v{number}.md
Example: my_project_readme_v1.md
```

### Standardization Mode
```
{filename}_standardized.{format}
Example: legacy_docs_standardized.md
```

### Analysis Reports
```
{topic}_analysis_report.{format}
{topic}_gpt_evaluation_report.{format}
{topic}_comparison_report.{format}
```

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | General error |
| `2` | Invalid arguments |
| `3` | Configuration error |
| `4` | Plugin error |
| `5` | API error (OpenAI/Claude) |
| `6` | File operation error |
| `7` | Standardization error |