# CLI API

The command-line interface provides comprehensive access to all doc-generator features.

## CLI Module

::: doc_generator.cli
    options:
      show_source: true
      show_root_heading: true
      show_root_members_full_path: false
      show_category_heading: true
      heading_level: 3

## Command-Line Arguments

The CLI accepts the following arguments:

### Core Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--topic` | `str` | Topic for documentation generation | Required |
| `--output-dir` | `str` | Output directory for generated files | `output` |
| `--runs` | `int` | Number of documentation variants to generate | `1` |
| `--model` | `str` | OpenAI model to use | `gpt-4o-mini` |
| `--temperature` | `float` | Generation temperature (0.0-1.0) | `0.3` |

### Configuration Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--prompt-yaml` | `str` | Path to prompt template file | `prompts/generator/default.yaml` |
| `--terminology-path` | `str` | Path to terminology YAML file | `terminology.yaml` |
| `--examples-dir` | `str` | Directory containing few-shot examples | `examples` |

### Plugin Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--list-plugins` | `flag` | List all available plugins and exit | `False` |
| `--disable-plugins` | `str` | Comma-separated list of plugins to disable | `None` |
| `--enable-only` | `str` | Only enable specified plugins (comma-separated) | `None` |

### Analysis Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--analyze` | `flag` | Run document analysis after generation | `False` |
| `--quality-eval` | `flag` | Run GPT-based quality evaluation | `False` |
| `--scan-code` | `str` | Directory to scan for code examples | `None` |
| `--max-scan-files` | `int` | Maximum files to scan for code examples | `100` |

### Output Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--verbose` | `flag` | Enable verbose logging output | `False` |
| `--dry-run` | `flag` | Show what would be done without executing | `False` |

## Usage Examples

### Basic Usage
```bash
doc-gen --topic "Python Programming" --output-dir ./docs
```

### Advanced Generation
```bash
doc-gen \
  --topic "Machine Learning with GPU" \
  --runs 3 \
  --model gpt-4 \
  --temperature 0.5 \
  --analyze \
  --quality-eval \
  --verbose
```

### Plugin Management
```bash
# List available plugins
doc-gen --list-plugins

# Disable specific plugins
doc-gen --topic "Topic" --disable-plugins modules,datasets

# Enable only specific plugins  
doc-gen --topic "Topic" --enable-only workflows,templates
```

### Custom Configuration
```bash
doc-gen \
  --topic "Custom Documentation" \
  --prompt-yaml ./custom-prompts/api-docs.yaml \
  --terminology-path ./config/custom-terms.yaml \
  --examples-dir ./examples/specialized/
```

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | General error |
| `2` | Invalid arguments |
| `3` | Configuration error |
| `4` | Plugin error |
| `5` | API error (OpenAI) |