# Cell 1: Configuration Variables
# ============================================
# CHANGE THESE VARIABLES TO CUSTOMIZE OUTPUT
# ============================================

TOPIC = "PyTorch"  # <-- Change this! Examples: "TensorFlow", "OpenMP", "MATLAB", "Conda", "Git"
RUNS = 5  # Number of variations to generate (1-10 recommended)
MODEL = 'gpt-4'  # Options: 'gpt-4', 'gpt-3.5-turbo'
TEMPERATURE = 0.7  # Creativity (0.0 = deterministic, 1.0 = very creative)

# Query template - modify if you want different phrasing
QUERY_TEMPLATE = "Create a knowledge base article with regards to using {topic} on the FASRC cluster, using the tone of graduate level Academic Computing documentation."

# Alternative query templates you can use:
# QUERY_TEMPLATE = "Generate HTML documentation for {topic} following academic computing standards"
# QUERY_TEMPLATE = "Write a technical reference page for {topic} on HPC clusters"
# QUERY_TEMPLATE = "Create a comprehensive guide for using {topic} in a research computing environment"

print(f"📌 Configuration set:")
print(f"   Topic: {TOPIC}")
print(f"   Runs: {RUNS}")
print(f"   Model: {MODEL}")
print(f"   Temperature: {TEMPERATURE}")


# Cell 2: Install Dependencies (run once)
# ============================================
import subprocess
import sys

def install_dependencies():
    """Install required packages if not already installed."""
    packages = ['openai>=1.0.0', 'pyyaml', 'python-dotenv']
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Uncomment to install
# install_dependencies()


# Cell 3: Import Libraries and Load Configuration
# ============================================
import os
import re
import glob
import yaml
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Optional
from datetime import datetime

# Load environment variables
load_dotenv()

# Check if API key is set
if not os.getenv('OPENAI_API_KEY'):
    print("⚠️  Warning: OPENAI_API_KEY not found in environment variables!")
    print("   Please create a .env file with: OPENAI_API_KEY=your-key-here")
else:
    print("✅ OpenAI API key loaded successfully")


# Cell 4: Define the DocumentationGenerator Class
# ============================================
# [Paste the entire DocumentationGenerator class here from the previous artifact]
# ... (class definition) ...


# Cell 5: Initialize Generator
# ============================================
print("🔧 Initializing documentation generator...")

try:
    generator = DocumentationGenerator(
        prompt_yaml_path='prompt.yaml',
        examples_dir='examples/'
    )
    print("✅ Generator initialized successfully")
    print(f"📁 Found {len(generator.examples)} examples")
except Exception as e:
    print(f"❌ Error initializing generator: {e}")


# Cell 6: Generate Documentation for Single Topic
# ============================================
# This cell uses the TOPIC variable defined in Cell 1

# Build the query from template
query = QUERY_TEMPLATE.format(topic=TOPIC)

print(f"\n{'='*60}")
print(f"📝 Generating documentation for: {TOPIC}")
print(f"📋 Query: {query}")
print(f"🔄 Generating {RUNS} variations...")
print(f"{'='*60}\n")

# Track generation time
start_time = datetime.now()

# Generate the documentation
files = generator.generate_documentation(
    query=query,
    runs=RUNS,
    model=MODEL,
    temperature=TEMPERATURE
)

# Calculate elapsed time
elapsed = (datetime.now() - start_time).total_seconds()

print(f"\n{'='*60}")
print(f"✅ Generation complete!")
print(f"⏱️  Time taken: {elapsed:.2f} seconds")
print(f"📁 Generated {len(files)} files:")
for file in files:
    print(f"   - {file}")
print(f"{'='*60}")


# Cell 7: Batch Generation for Multiple Topics (Optional)
# ============================================
# Uncomment and run this cell to generate docs for multiple topics at once

# TOPICS_LIST = ["PyTorch", "TensorFlow", "OpenMP", "MATLAB", "Conda"]
# 
# for topic in TOPICS_LIST:
#     query = QUERY_TEMPLATE.format(topic=topic)
#     print(f"\n📝 Generating documentation for: {topic}")
#     
#     files = generator.generate_documentation(
#         query=query,
#         runs=1,  # Just one version per topic for batch
#         model=MODEL,
#         temperature=TEMPERATURE
#     )
#     
#     print(f"✅ Generated: {', '.join(files)}")


# Cell 8: Preview Generated Files (Optional)
# ============================================
# This cell lets you preview the generated HTML files

from IPython.display import HTML, display
import os

# Get the most recently generated file
output_dir = Path('output')
if output_dir.exists():
    html_files = sorted(output_dir.glob(f'{TOPIC.lower()}*.html'))
    if html_files:
        latest_file = html_files[-1]
        print(f"📄 Previewing: {latest_file.name}")
        print("="*60)
        
        with open(latest_file, 'r') as f:
            content = f.read()
            # Show first 1000 characters
            print(content[:1000] + "..." if len(content) > 1000 else content)
            
        # Optionally display as rendered HTML (uncomment if in Jupyter)
        # display(HTML(content))
    else:
        print(f"No files found for topic: {TOPIC}")
else:
    print("Output directory not found!")


# Cell 9: Compare Multiple Versions (Optional)
# ============================================
# This cell helps you compare different generated versions

def compare_versions(topic: str):
    """Compare key differences between generated versions."""
    output_dir = Path('output')
    files = sorted(output_dir.glob(f'{topic.lower()}_*.html'))
    
    if len(files) < 2:
        print("Need at least 2 versions to compare")
        return
    
    print(f"📊 Comparing {len(files)} versions of {topic} documentation:\n")
    
    for i, file in enumerate(files, 1):
        with open(file, 'r') as f:
            content = f.read()
            
        # Extract some metrics
        word_count = len(content.split())
        line_count = len(content.splitlines())
        has_examples = 'example' in content.lower()
        has_code_blocks = '<code>' in content or '<pre>' in content
        
        print(f"Version {i} ({file.name}):")
        print(f"  - Words: {word_count}")
        print(f"  - Lines: {line_count}")
        print(f"  - Has examples: {'Yes' if has_examples else 'No'}")
        print(f"  - Has code blocks: {'Yes' if has_code_blocks else 'No'}")
        print()

# Run comparison
# compare_versions(TOPIC)
