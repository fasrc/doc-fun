# Imports at top of file
import os
import re
import subprocess
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict
from datetime import datetime

# Cell 1: Configuration Variables
# ============================================
# CHANGE THESE VARIABLES TO CUSTOMIZE OUTPUT
# ============================================

TOPIC = "PyTorch"  # <-- Change this! Examples: "TensorFlow", "OpenMP", "MATLAB", "Conda", "Git"
TOPIC_FILENAME = TOPIC.lower().replace(' ', '_')  # Filename-safe version for file creation
RUNS = 5  # Number of variations to generate (1-10 recommended)
MODEL = 'gpt-4'  # Options: 'gpt-4', 'gpt-3.5-turbo'
TEMPERATURE = 0.7  # Creativity (0.0 = deterministic, 1.0 = very creative)

# Query template - modify if you want different phrasing
QUERY_TEMPLATE = "Create a knowledge base article with regards to using {topic} on the FASRC cluster, using the tone of graduate level Academic Computing documentation."

# Alternative query templates you can use:
# QUERY_TEMPLATE = "Generate HTML documentation for {topic} following academic computing standards"
# QUERY_TEMPLATE = "Write a technical reference page for {topic} on HPC clusters"
# QUERY_TEMPLATE = "Create a comprehensive guide for using {topic} in a research computing environment"

print("📌 Configuration set:")
print(f"   Topic: {TOPIC}")
print(f"   Filename format: {TOPIC_FILENAME}")
print(f"   Runs: {RUNS}")
print(f"   Model: {MODEL}")
print(f"   Temperature: {TEMPERATURE}")


# Cell 2: Install Dependencies (run once)
# ============================================
def install_dependencies():
    """Install required packages if not already installed."""
    packages = ['openai>=1.0.0', 'pyyaml', 'python-dotenv']
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Uncomment to install
# install_dependencies()


# Cell 3: Load Configuration
# ============================================

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
class DocumentationGenerator:
    def __init__(self, prompt_yaml_path: str = 'prompt.yaml', examples_dir: str = 'examples/'):
        """Initialize the documentation generator with configuration."""
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.examples_dir = Path(examples_dir)
        self.prompt_config = self._load_prompt_config(prompt_yaml_path)
        self.examples = self._load_examples()
        
    def _load_prompt_config(self, path: str) -> dict:
        """Load the prompt configuration from YAML file."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: {path} not found. Using default configuration.")
            return {
                'system_prompt': 'You are a technical documentation expert.',
                'documentation_structure': ['Description', 'Installation', 'Usage', 'Examples', 'References']
            }
    
    def _load_examples(self) -> List[Dict[str, str]]:
        """Load few-shot examples from YAML files."""
        examples = []
        
        # Ensure examples directory exists
        self.examples_dir.mkdir(exist_ok=True)
        
        # Load YAML examples
        yaml_files = sorted(self.examples_dir.glob('*.yaml'))
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r') as f:
                    msgs = yaml.safe_load(f)
                    if isinstance(msgs, list):
                        examples.extend(msgs)
                    else:
                        examples.append(msgs)
            except Exception as e:
                print(f"Error loading {yaml_file}: {e}")
        
        # Load HTML examples if needed for reference
        html_files = sorted(self.examples_dir.glob('*.html'))
        for html_file in html_files:
            try:
                with open(html_file, 'r') as f:
                    content = f.read()
                    # Add as assistant example showing the expected format
                    examples.append({
                        'role': 'assistant',
                        'content': content,
                        'metadata': {'filename': html_file.name}
                    })
            except Exception as e:
                print(f"Error loading {html_file}: {e}")
        
        return examples
    
    def _extract_topic_from_query(self, query: str) -> str:
        """Extract the main topic from the query for filename generation."""
        # Try to extract topic using various patterns
        patterns = [
            r'documentation for (\w+)',
            r'using (\w+)',
            r'about (\w+)',
            r'for (\w+) documentation',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).lower().replace(' ', '_')
        
        # Fallback: use first significant word
        words = query.split()
        for word in words:
            if len(word) > 3 and word.lower() not in ['create', 'make', 'generate', 'write']:
                return word.lower().replace(' ', '_')
        
        return 'documentation'
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt from configuration."""
        base_prompt = self.prompt_config.get('system_prompt', 
            'You are a technical documentation expert creating HTML knowledge base articles.')
        
        # Add structure information if available
        if 'documentation_structure' in self.prompt_config:
            structure = self.prompt_config['documentation_structure']
            base_prompt += f"\n\nEach article should follow this structure:\n"
            base_prompt += "\n".join(f"- {section}" for section in structure)
        
        # Add any terms/definitions
        if 'terms' in self.prompt_config:
            base_prompt += "\n\nKey terms:\n"
            for term, definition in self.prompt_config['terms'].items():
                base_prompt += f"- {term}: {definition}\n"
        
        return base_prompt
    
    def generate_documentation(self, query: str, runs: int = 5, 
                             model: str = 'gpt-4', 
                             temperature: float = 0.7) -> List[str]:
        """Generate multiple documentation pages based on the query."""
        topic = self._extract_topic_from_query(query)
        generated_files = []
        
        # Build messages
        system_prompt = self._build_system_prompt()
        
        for i in range(runs):
            try:
                messages = [
                    {'role': 'system', 'content': system_prompt}
                ]
                
                # Add few-shot examples
                for example in self.examples:
                    # Only add role and content, skip metadata
                    messages.append({
                        'role': example['role'],
                        'content': example['content']
                    })
                
                # Add the actual query
                messages.append({'role': 'user', 'content': query})
                
                # Make API call
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                )
                
                content = response.choices[0].message.content.strip()
                
                # Generate filename based on topic, model, temperature, and iteration
                # Clean model name (remove special characters)
                model_name = model.replace('-', '').replace('.', '')
                temp_str = str(temperature).replace('.', '')

                if runs == 1:
                    filename = f'{topic}_{model_name}_temp{temp_str}.html'
                else:
                    filename = f'{topic}_{model_name}_temp{temp_str}_v{i+1}.html'                
                # Save the response
                output_dir = Path('output')
                output_dir.mkdir(exist_ok=True)
                
                filepath = output_dir / filename
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                generated_files.append(str(filepath))
                print(f"✓ Generated: {filepath}")
                
            except Exception as e:
                print(f"✗ Error generating documentation (run {i+1}): {e}")
        
        return generated_files


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

print("\n" + "=" * 60)
print(f"📝 Generating documentation for: {TOPIC}")
print(f"📋 Query: {query}")
print(f"🔄 Generating {RUNS} variations...")
print("=" * 60 + "\n")

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

print("\n" + "=" * 60)
print("✅ Generation complete!")
print(f"⏱️  Time taken: {elapsed:.2f} seconds")
print(f"📁 Generated {len(files)} files:")
for file in files:
    print(f"   - {file}")
print("=" * 60)


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

# Note: IPython imports removed for standalone script compatibility

# Get the most recently generated file
output_dir = Path('output')
if output_dir.exists():
    html_files = sorted(output_dir.glob(f'{TOPIC_FILENAME}*.html'))
    if html_files:
        latest_file = html_files[-1]
        print(f"📄 Previewing: {latest_file.name}")
        print("=" * 60)
        
        with open(latest_file, 'r') as f:
            content = f.read()
            # Show first 1000 characters
            print(content[:1000] + "..." if len(content) > 1000 else content)
            
        # Note: HTML display not available in standalone script
        # Use a web browser to view the generated HTML files
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
    files = sorted(output_dir.glob(f'{topic.lower().replace(" ", "_")}_*.html'))
    
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
