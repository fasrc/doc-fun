# Troubleshooting

This guide helps you diagnose and resolve common issues with doc-generator.

## Common Issues

### Installation Problems

#### Issue: `ModuleNotFoundError: No module named 'doc_generator'`

**Symptoms:**
```bash
$ doc-gen --help
Traceback (most recent call last):
  File "/usr/local/bin/doc-gen", line 5, in <module>
    from doc_generator.cli import main
ModuleNotFoundError: No module named 'doc_generator'
```

**Solutions:**
```bash
# 1. Verify you're in the correct directory
cd /path/to/doc-fun

# 2. Check if package is installed
pip list | grep doc-generator

# 3. Reinstall in editable mode
pip uninstall doc-generator  # if installed
pip install -e .

# 4. Verify installation
python -c "import doc_generator; print(doc_generator.__version__)"
```

#### Issue: `ImportError: No module named 'openai'`

**Symptoms:**
```
ImportError: No module named 'openai'
```

**Solutions:**
```bash
# 1. Install missing dependencies
pip install openai>=1.0.0

# 2. Or reinstall with all dependencies
pip install -e ".[dev]"

# 3. Check requirements
pip check
```

### API and Authentication Issues

#### Issue: `OpenAI API key not found`

**Symptoms:**
```
Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable.
```

**Solutions:**
```bash
# 1. Check if .env file exists
ls -la .env

# 2. Create .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env

# 3. Set environment variable directly
export OPENAI_API_KEY="your-api-key-here"

# 4. Verify API key is loaded
python -c "import os; print('Key found' if os.getenv('OPENAI_API_KEY') else 'Key missing')"
```

#### Issue: `OpenAI API Error: Rate limit exceeded`

**Symptoms:**
```
OpenAI API Error: Rate limit exceeded for requests per minute. Please try again later.
```

**Solutions:**
```bash
# 1. Wait and retry (rate limits reset after time)
sleep 60
doc-gen --topic "Your Topic"

# 2. Use a slower approach with delays
doc-gen --topic "Topic 1"
sleep 10
doc-gen --topic "Topic 2"

# 3. Use a cheaper model for testing
doc-gen --topic "Test Topic" --model gpt-3.5-turbo

# 4. Reduce number of runs
doc-gen --topic "Your Topic" --runs 1
```

#### Issue: `OpenAI API Error: Invalid API key`

**Symptoms:**
```
OpenAI API Error: Incorrect API key provided
```

**Solutions:**
```bash
# 1. Verify API key format (should start with sk-)
echo $OPENAI_API_KEY

# 2. Check OpenAI dashboard for valid keys
# Visit: https://platform.openai.com/api-keys

# 3. Generate new API key if needed
# 4. Update .env file with correct key
```

### Plugin Issues

#### Issue: No Plugin Recommendations Appearing

**Symptoms:**
- Generated documentation lacks HPC module suggestions
- No plugin-specific content in output

**Diagnosis:**
```bash
# 1. Check if plugins are loaded
doc-gen --list-plugins

# Expected output should show modules plugin:
# Plugin: modules
#   Class: ModuleRecommender
#   Enabled: True
```

**Solutions:**
```bash
# 1. Verify terminology.yaml has HPC modules
head -20 terminology.yaml

# Should contain:
# hpc_modules:
#   - name: "python/3.12.8-fasrc01"
#     description: "Python 3.12..."

# 2. Test plugin directly
python -c "
from doc_generator.plugins.modules import ModuleRecommender
import yaml
with open('terminology.yaml') as f:
    term = yaml.safe_load(f)
rec = ModuleRecommender(terminology=term) 
print(rec.get_recommendations('Python'))
"

# 3. Check plugin manager
python -c "
from doc_generator.plugin_manager import PluginManager
import yaml
with open('terminology.yaml') as f:
    term = yaml.safe_load(f)
pm = PluginManager(terminology=term)
pm.load_plugins()
print(list(pm.engines.keys()))
"
```

#### Issue: Plugin Import Errors

**Symptoms:**
```
WARNING: Failed to load plugin 'my_plugin': No module named 'my_plugin'
```

**Solutions:**
```bash
# 1. Verify plugin package is installed
pip list | grep my-plugin

# 2. Check entry points registration
python -c "
from importlib.metadata import entry_points
eps = entry_points(group='doc_generator.plugins')
for ep in eps:
    print(f'{ep.name}: {ep.value}')
"

# 3. Reinstall plugin package
pip install -e /path/to/my-plugin-package
```

### Generation Quality Issues

#### Issue: Generated Documentation is Too Generic

**Symptoms:**
- Output lacks specific details
- Missing relevant technical information
- Generic boilerplate content

**Solutions:**
```bash
# 1. Use more specific topics
# Poor
doc-gen --topic "Python"

# Better  
doc-gen --topic "Python pandas DataFrame operations for time series analysis"

# 2. Use better model
doc-gen --topic "Specific Topic" --model gpt-4

# 3. Lower temperature for more focused output
doc-gen --topic "Technical Documentation" --temperature 0.1

# 4. Generate multiple variants and compare
doc-gen --topic "Important Topic" --runs 5 --analyze
```

#### Issue: Missing HPC-Specific Content

**Symptoms:**
- No module load commands
- Generic installation instructions
- Missing cluster-specific information

**Solutions:**
```bash
# 1. Ensure terminology.yaml is comprehensive
cat terminology.yaml | grep -A 5 hpc_modules

# 2. Use HPC-specific topics
doc-gen --topic "Running Python machine learning on SLURM cluster"

# 3. Check plugin status
doc-gen --list-plugins | grep modules

# 4. Test with verbose output
doc-gen --topic "HPC Topic" --verbose
```

### Performance Issues

#### Issue: Very Slow Generation

**Symptoms:**
- Long wait times (>5 minutes per generation)
- Timeouts or hanging

**Diagnosis:**
```bash
# Run with verbose output to see where it's stuck
doc-gen --topic "Test Topic" --verbose
```

**Solutions:**
```bash
# 1. Use faster model
doc-gen --topic "Your Topic" --model gpt-3.5-turbo

# 2. Reduce number of runs
doc-gen --topic "Your Topic" --runs 1

# 3. Disable external API plugins if you have them
doc-gen --topic "Your Topic" --disable-plugins github,datasets

# 4. Check internet connection and OpenAI API status
curl -I https://api.openai.com/v1/models
```

#### Issue: High API Costs

**Symptoms:**
- Unexpected high charges on OpenAI account
- Rapid token consumption

**Solutions:**
```bash
# 1. Use cheaper models for development
doc-gen --topic "Dev Topic" --model gpt-3.5-turbo

# 2. Reduce runs and temperature
doc-gen --topic "Topic" --runs 1 --temperature 0.3

# 3. Use more specific topics (reduces token usage)
doc-gen --topic "Very specific technical question"

# 4. Monitor usage
# Check OpenAI dashboard regularly
```

### Output and File Issues

#### Issue: Empty or Corrupted Output Files

**Symptoms:**
```bash
# Empty files
$ ls -la output/
-rw-r--r-- 1 user staff 0 date topic_file.html

# Or corrupted content
$ head output/topic_file.html
Error: [object Object]
```

**Solutions:**
```bash
# 1. Check for API errors in verbose mode
doc-gen --topic "Test Topic" --verbose

# 2. Verify output directory permissions
mkdir -p output
chmod 755 output

# 3. Test with simple topic
doc-gen --topic "Hello World" --runs 1

# 4. Check available disk space
df -h .
```

#### Issue: File Permission Errors

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: 'output/file.html'
```

**Solutions:**
```bash
# 1. Check directory permissions
ls -la output/

# 2. Fix permissions
chmod 755 output/
chmod 644 output/*.html

# 3. Use different output directory
doc-gen --topic "Topic" --output-dir /tmp/doc-test

# 4. Check if files are open in other applications
lsof output/*.html
```

## Debugging Tools

### Verbose Mode

```bash
# Enable detailed logging
doc-gen --topic "Debug Topic" --verbose

# This shows:
# - Plugin loading process
# - API request/response details
# - File operations
# - Error stack traces
```

### Dry Run Mode

```bash
# See what would happen without executing
doc-gen --topic "Test Topic" --dry-run --verbose

# Shows:
# - Configuration that would be used
# - Plugins that would be loaded
# - Files that would be created
```

### Direct API Testing

```python
# test_api.py - Test OpenAI API directly
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, API test"}],
        max_tokens=50
    )
    print("✓ API connection successful")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"✗ API error: {e}")
```

### Plugin Testing

```python
# test_plugins.py - Test plugin system
from doc_generator.plugin_manager import PluginManager
import yaml

try:
    # Load terminology
    with open('terminology.yaml') as f:
        terminology = yaml.safe_load(f)
    
    # Create plugin manager
    pm = PluginManager(terminology=terminology)
    pm.load_plugins()
    
    print(f"✓ Loaded {len(pm.engines)} plugins:")
    for name, plugin in pm.engines.items():
        print(f"  - {name}: {plugin.__class__.__name__}")
    
    # Test recommendations
    recommendations = pm.get_recommendations("Python")
    print(f"✓ Got {len(recommendations)} total recommendations")
    
except Exception as e:
    print(f"✗ Plugin error: {e}")
    import traceback
    traceback.print_exc()
```

## System Diagnostics

### Environment Check Script

```bash
#!/bin/bash
# diagnose.sh - System diagnostic script

echo "=== doc-generator Diagnostics ==="
echo

echo "1. Python Environment:"
python --version
which python
echo

echo "2. Package Installation:"
pip list | grep -E "(doc-generator|openai|pyyaml)"
echo

echo "3. API Key Status:"
if [ -z "$OPENAI_API_KEY" ]; then 
    echo "OPENAI_API_KEY not set"
else 
    echo "OPENAI_API_KEY is set"
fi
echo

echo "4. Configuration Files:"
for file in .env terminology.yaml prompts/generator/default.yaml; do
    if [ -f "$file" ]; then
        echo "$file exists ($(wc -l < "$file") lines)"
    else
        echo "$file missing"
    fi
done
echo

echo "5. Output Directory:"
if [ -d "output" ]; then
    echo "output/ exists ($(ls output/ | wc -l) files)"
    ls -la output/ | head -5
else
    echo "output/ directory missing"
fi
echo

echo "6. Plugin Status:"
doc-gen --list-plugins 2>&1 | head -10
echo

echo "7. Basic Test:"
export OPENAI_API_KEY="${OPENAI_API_KEY:-test-key-for-dry-run}"
doc-gen --topic "Test Topic" --dry-run --verbose 2>&1 | tail -5
```

### Performance Monitoring

```python
# performance_monitor.py
import time
import psutil
import os
from doc_generator import DocumentationGenerator

def monitor_generation(topic: str, **kwargs):
    """Monitor system resources during generation."""
    
    # Start monitoring
    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Starting generation: {topic}")
    print(f"Initial memory: {start_memory:.1f} MB")
    
    # Generate documentation
    generator = DocumentationGenerator()
    results = generator.generate_documentation(query=topic, **kwargs)
    
    # End monitoring
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    duration = end_time - start_time
    memory_used = end_memory - start_memory
    
    print(f"Generation completed in {duration:.1f} seconds")
    print(f"Memory used: {memory_used:.1f} MB")
    print(f"Generated {len(results)} variants")
    
    return results

# Usage
results = monitor_generation("Python programming", runs=2)
```

## 📞 Getting Help

### Self-Help Checklist

Before seeking help, try this checklist:

- [ ] Read error messages carefully
- [ ] Check this troubleshooting guide
- [ ] Run with `--verbose` flag
- [ ] Test with simple topic
- [ ] Verify API key and internet connection
- [ ] Check file permissions
- [ ] Try with fresh virtual environment

### Support Resources

1. **GitHub Issues**: [Report bugs and get help](https://github.com/fasrc/doc-fun/issues)
2. **Documentation**: [Read all guides](../guides/getting-started.md)
3. **Examples**: [Check usage examples](basic.md)
4. **FASRC Support**: Contact the original development team

### Creating Bug Reports

When reporting issues, include:

```markdown
**Environment:**
- OS: [e.g., macOS 13.2]
- Python: [e.g., 3.10.8]
- doc-generator: [e.g., 1.1.0]

**Command Used:**
```bash
doc-gen --topic "Your Topic" --model gpt-4
```

**Error Output:**
```
[Paste full error output here]
```

**Expected Behavior:**
[What you expected to happen]

**Additional Context:**
[Any other relevant information]
```

### Quick Fixes Reference

| Issue | Quick Fix |
|-------|-----------|
| Module not found | `pip install -e .` |
| API key missing | `export OPENAI_API_KEY="your-key"` |
| Permission denied | `chmod 755 output/` |
| Plugin not loading | `doc-gen --list-plugins` |
| Slow generation | `--model gpt-3.5-turbo --runs 1` |
| Generic output | Use more specific topics |
| Rate limit | Wait 60 seconds and retry |
| Empty output | Check `--verbose` for errors |

---

Most issues can be resolved by carefully reading error messages and following the solutions above. For persistent problems, don't hesitate to seek help through the support channels!

---

**Need more help?** Check out the [Getting Started Guide](../guides/getting-started.md) for basic usage or the [Installation Guide](../guides/installation.md) for setup issues.