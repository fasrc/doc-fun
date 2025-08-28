# Installation Guide

This guide covers different ways to install and set up doc-generator for development and production use.

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 2GB RAM minimum (4GB+ recommended for large documentation sets)
- **Storage**: 500MB for installation + space for generated documentation

### Required Dependencies
- `openai>=1.0.0` - OpenAI API client
- `pyyaml>=6.0` - YAML configuration parsing
- `python-dotenv>=0.19.0` - Environment variable management
- `beautifulsoup4>=4.11.0` - HTML parsing and analysis
- `pandas>=1.5.0` - Data analysis and reporting
- `tabulate>=0.9.0` - Table formatting

## Quick Installation

### Option 1: Development Installation (Recommended)

For development, testing, or if you want to modify the code:

```bash
# Clone the repository
git clone https://github.com/fasrc/doc-fun.git
cd doc-fun

# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Verify installation
doc-gen --version
doc-gen --help
```

### Option 2: Production Installation

For production use (when available on PyPI):

```bash
# Install from PyPI (future release)
pip install doc-generator

# Verify installation
doc-gen --version
```

## Environment Setup

### 1. OpenAI API Configuration

Create a `.env` file in your project directory:

```bash
# Create .env file
touch .env

# Add your OpenAI API key
echo "OPENAI_API_KEY=your-api-key-here" >> .env
```

**Getting an OpenAI API Key:**
1. Visit [OpenAI API Platform](https://platform.openai.com/)
2. Create an account or sign in
3. Navigate to API Keys section
4. Create a new secret key
5. Copy the key to your `.env` file

**Important Security Notes:**
- Never commit `.env` files to version control
- Keep your API key secure and private
- Monitor your API usage to avoid unexpected charges

### 2. Configuration Files

Set up the basic configuration structure:

```bash
# Create configuration directories
mkdir -p prompts/generator prompts/analysis
mkdir -p examples output

# Copy default configuration files (if they don't exist)
# These are included in the repository
```

**Key Configuration Files:**
- `prompts/generator/default.yaml` - Default prompt template
- `terminology.yaml` - HPC modules and terminology
- `examples/` - Few-shot learning examples

## Virtual Environment Setup

### Using venv (Recommended)

```bash
# Create virtual environment
python -m venv doc-generator-env

# Activate virtual environment
# On Linux/macOS:
source doc-generator-env/bin/activate
# On Windows:
doc-generator-env\Scripts\activate

# Install doc-generator
pip install -e .

# Deactivate when done
deactivate
```

### Using conda

```bash
# Create conda environment
conda create -n doc-generator python=3.10
conda activate doc-generator

# Install doc-generator
pip install -e .

# Deactivate when done
conda deactivate
```

## Verify Installation

### Basic Verification

```bash
# Check version
doc-gen --version

# List available plugins
doc-gen --list-plugins

# View help
doc-gen --help
```

### Test Generation

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run a simple test
doc-gen --topic "Test Installation" --runs 1 --output-dir ./test-output

# Check output
ls -la test-output/
```

### Run Test Suite

```bash
# Run all tests
python -m pytest -v

# Expected output: 57 tests should pass
# ========================= 57 passed in X.XXs =========================
```

## Development Setup

### Full Development Installation

```bash
# Clone repository
git clone https://github.com/fasrc/doc-fun.git
cd doc-fun

# Install with all development dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks (if available)
pre-commit install

# Run development checks
python -m pytest -v
python -m black src/ tests/
python -m flake8 src/ tests/
```

### IDE Configuration

#### VS Code

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./doc-generator-env/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black"
}
```

#### PyCharm

1. Set Python interpreter to your virtual environment
2. Enable pytest as test runner
3. Configure Black as code formatter
4. Enable flake8 for linting

## Troubleshooting

### Common Issues

#### Issue: `ModuleNotFoundError: No module named 'doc_generator'`

**Solution:**
```bash
# Make sure you're in the correct directory
cd /path/to/doc-fun

# Reinstall in editable mode
pip install -e .
```

#### Issue: `OpenAI API key not found`

**Solution:**
```bash
# Check if .env file exists
ls -la .env

# Verify API key is set
cat .env

# Set environment variable directly
export OPENAI_API_KEY="your-api-key-here"
```

#### Issue: `ImportError: No module named 'openai'`

**Solution:**
```bash
# Check if dependencies are installed
pip list | grep openai

# Reinstall dependencies
pip install -r requirements.txt
# OR
pip install -e .
```

#### Issue: Tests failing during installation

**Solution:**
```bash
# Check Python version
python --version  # Should be 3.8+

# Clean installation
pip uninstall doc-generator
pip cache purge
pip install -e .

# Run tests individually
python -m pytest tests/test_plugin_interface.py -v
```

### Getting Help

#### Check System Information

```bash
# Python version
python --version

# Pip version
pip --version

# Installed packages
pip list

# Doc-generator installation
pip show doc-generator
```

#### Debugging Steps

1. **Check Installation:**
   ```bash
   python -c "import doc_generator; print(doc_generator.__version__)"
   ```

2. **Check Plugin Discovery:**
   ```bash
   python -c "from importlib.metadata import entry_points; print(list(entry_points(group='doc_generator.plugins')))"
   ```

3. **Check Configuration:**
   ```bash
   doc-gen --list-plugins --verbose
   ```

#### Support Channels

- **GitHub Issues**: [Report bugs and request features](https://github.com/fasrc/doc-fun/issues)
- **Documentation**: [Read the guides](https://github.com/fasrc/doc-fun/docs)
- **FASRC Support**: Contact FASRC Research Computing

## Updating

### Update Development Installation

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -e ".[dev]"

# Run tests to verify
python -m pytest -v
```

### Update Production Installation

```bash
# Update from PyPI (when available)
pip install --upgrade doc-generator

# Verify update
doc-gen --version
```

## Docker Installation (Optional)

### Using Docker

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -e .

CMD ["doc-gen", "--help"]
```

```bash
# Build image
docker build -t doc-generator .

# Run container
docker run -e OPENAI_API_KEY="your-key" doc-generator \
  doc-gen --topic "Docker Test" --output-dir /app/output
```

---

## Next Steps

After successful installation:

1. **[Read Getting Started Guide](getting-started.md)** - Learn basic usage
2. **[Run Tests](testing.md)** - Understand the test suite
3. **[Create Plugins](creating-plugins.md)** - Extend functionality
4. **[Configure Templates](configuration.md)** - Customize prompts and terminology

**Installation Complete!**

Your doc-generator installation is ready. You can now generate AI-powered documentation with intelligent plugin recommendations.