# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python-based documentation generation tool that uses OpenAI's API to create HTML documentation pages from natural language prompts. The system uses few-shot prompting and structured evaluation to generate consistent, high-quality technical documentation for Faculty Arts and Sciences Research Computing (FASRC).

## Key Commands

### Running the Application
```bash
python run.py  # Standalone script version
```
The script automatically runs through the basic documentation generation workflow.

### Working with the Jupyter Notebook
The primary interface is `doc-fun.ipynb` which provides:
- Interactive configuration (Cell 1: modify TOPIC, TOPIC_FILENAME, RUNS, MODEL, TEMPERATURE)
- Documentation generation workflow (Cells 1-6 for basic usage)
- Advanced analysis and evaluation (Cells 7-22 for quality assessment)

### Dependencies Installation
The notebook includes automatic dependency installation, but you can manually install:
```bash
pip install openai>=1.0.0 pyyaml python-dotenv beautifulsoup4 pandas tabulate
```

## Architecture

### Core Components

**DocumentationGenerator Class** (`doc-fun.ipynb` Cell 4)
- Main API interface for OpenAI GPT models
- Handles few-shot prompting with examples loaded from `examples/` directory
- Manages filename generation based on content analysis
- Supports multiple model variations and temperature settings

**DocumentAnalyzer Class** (`doc-fun.ipynb` Cell 10)
- Extracts and scores documentation sections using algorithmic metrics
- Evaluates completeness, structure, and formatting quality
- Provides objective scoring based on length, code examples, links, and organization

**GPTQualityEvaluator Class** (`doc-fun.ipynb` Cell 17)
- Uses GPT models to evaluate subjective quality metrics
- Assesses technical accuracy, writing style, and completeness
- Provides detailed explanations for quality scores

### File Structure

- `prompt.yaml`: Configuration for few-shot prompting and documentation structure
- `examples/`: HTML examples used for few-shot learning (matlab.html, mpi.html, etc.)
- `output/`: Generated documentation files and analysis reports
- `run.py`: Standalone command-line script that replicates basic notebook functionality
- `viewer.html`: HTML viewer for generated documentation

### Configuration System

The system uses YAML configuration (`prompt.yaml`) with:
- Documentation structure templates (Description, Installation, Usage, Examples, References)
- Few-shot examples for consistent naming conventions
- FASRC-specific terminology and formatting guidelines

### Quality Evaluation Pipeline

1. **Generation**: Creates multiple variations of documentation using different temperature settings
2. **Algorithmic Scoring**: Evaluates structural elements like length, code blocks, formatting
3. **GPT Evaluation**: Assesses technical accuracy, writing style, and completeness
4. **Combined Scoring**: Merges both evaluation methods with configurable weights
5. **Best Compilation**: Creates optimal documentation by selecting best sections across variations

## Environment Setup

Requires `.env` file with:
```
OPENAI_API_KEY=your-key-here
```

## Output Structure

Generated files follow the pattern:
- Individual versions: `{topic}_{model}_temp{temperature}_v{number}.html`
- Best compilations: `{topic}_best_compilation.html`
- Analysis reports: `{topic}_analysis_report.md` and `{topic}_gpt_evaluation_report.md`

## Development Notes

- **Variable Usage**: Use `TOPIC` for display/human-readable purposes, `TOPIC_FILENAME` for all file operations to ensure consistent underscore formatting
- The notebook provides both basic usage (Cells 1-6) and advanced analysis (Cells 7-22)
- Cost estimation helpers are included for managing OpenAI API usage
- HTML parsing uses BeautifulSoup for section extraction and analysis
- Supports batch generation for multiple topics
- Built-in comparison tools for evaluating different generated versions