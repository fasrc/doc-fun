#!/usr/bin/env python3
"""
Documentation Generator - Standalone Script
Converted from Jupyter notebook to generate HTML documentation using OpenAI GPT models.

This script provides comprehensive documentation generation with analysis and evaluation capabilities.
"""

import os
import re
import json
import time
import yaml
import argparse
import subprocess
import sys
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from bs4 import BeautifulSoup

try:
    import pandas as pd
except ImportError:
    pd = None


def install_dependencies():
    """Install required packages if not already installed."""
    packages = ['openai>=1.0.0', 'pyyaml', 'python-dotenv', 'beautifulsoup4', 'pandas', 'tabulate']
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to install {package}")


class DocumentationGenerator:
    """Main class for generating documentation using OpenAI GPT models."""
    
    def __init__(self, prompt_yaml_path: str = 'prompt.yaml', examples_dir: str = 'examples/',
                 terminology_path: str = 'terminology.yaml'):
        """Initialize the documentation generator with configuration."""
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.examples_dir = Path(examples_dir)
        self.prompt_config = self._load_prompt_config(prompt_yaml_path)
        self.terminology = self._load_terminology(terminology_path)
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
    
    def _load_terminology(self, path: str) -> dict:
        """Load terminology configuration from YAML file."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: {path} not found. No terminology loaded.")
            return {}
    
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
    
    def _extract_topic_keywords(self, topic: str) -> List[str]:
        """Extract meaningful keywords from the topic for module matching."""
        import re
        
        # Convert to lowercase and extract words
        words = re.findall(r'\b\w+\b', topic.lower())
        
        # Filter out common stop words that aren't useful for module matching
        stop_words = {
            'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'using', 'how', 'what', 'when', 'where', 'why', 'computing', 'cluster',
            'fasrc', 'cannon', 'hpc', 'high', 'performance'
        }
        
        # Keep words that are 2+ characters and not stop words
        keywords = [word for word in words if len(word) >= 2 and word not in stop_words]
        
        return keywords
    
    def _find_relevant_modules(self, topic_keywords: List[str]) -> List[Dict]:
        """Find modules relevant to the topic keywords."""
        if not self.terminology.get('hpc_modules') or not topic_keywords:
            return []
        
        relevant_modules = []
        
        for module in self.terminology['hpc_modules']:
            score = 0
            
            # Check for keyword matches in module name
            module_name_lower = module['name'].lower()
            for keyword in topic_keywords:
                if keyword in module_name_lower:
                    score += 3  # High weight for name matches
            
            # Check for keyword matches in module description
            module_desc_lower = module['description'].lower()
            for keyword in topic_keywords:
                if keyword in module_desc_lower:
                    score += 2  # Medium weight for description matches
            
            # Check for keyword matches in category
            module_category_lower = module['category'].lower()
            for keyword in topic_keywords:
                if keyword in module_category_lower:
                    score += 1  # Lower weight for category matches
            
            if score > 0:
                module_with_score = module.copy()
                module_with_score['relevance_score'] = score
                relevant_modules.append(module_with_score)
        
        # Sort by relevance score (highest first) and limit results
        relevant_modules.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_modules[:15]  # Limit to top 15 to avoid token overflow
    
    def _build_terminology_context(self, topic: str) -> str:
        """Build context-aware terminology based on the topic."""
        if not self.terminology:
            return ""
        
        context_parts = []
        topic_keywords = self._extract_topic_keywords(topic)
        
        # Find relevant modules based on topic keywords
        relevant_modules = self._find_relevant_modules(topic_keywords)
        
        if relevant_modules:
            context_parts.append("Available HPC Modules (relevant to this topic):")
            for module in relevant_modules:
                context_parts.append(f"- {module['name']}: {module['description']}")
        else:
            # Fallback: include some essential modules if no matches found
            if 'hpc_modules' in self.terminology:
                essential_modules = [m for m in self.terminology['hpc_modules'] 
                                   if m['category'] in ['programming', 'compiler']][:8]
                if essential_modules:
                    context_parts.append("Available HPC Modules (essential):")
                    for module in essential_modules:
                        context_parts.append(f"- {module['name']}: {module['description']}")
        
        # Include cluster commands for most topics
        if 'cluster_commands' in self.terminology:
            context_parts.append("\nCommon SLURM Commands:")
            for cmd in self.terminology['cluster_commands'][:6]:  # Limit to most important
                context_parts.append(f"- {cmd['name']}: {cmd['description']}")
                if 'usage' in cmd:
                    context_parts.append(f"  Usage: {cmd['usage']}")
        
        # Include relevant filesystems
        if 'filesystems' in self.terminology:
            context_parts.append("\nFASRC Filesystems:")
            for fs in self.terminology['filesystems']:
                context_parts.append(f"- {fs['name']}: {fs['description']}")
        
        # Include partition information for GPU/parallel topics
        if 'partitions' in self.terminology:
            gpu_parallel_keywords = ['gpu', 'parallel', 'mpi', 'cuda', 'computing']
            if any(keyword in topic_keywords for keyword in gpu_parallel_keywords):
                context_parts.append("\nCluster Partitions:")
                for partition in self.terminology['partitions']:
                    context_parts.append(f"- {partition['name']}: {partition['description']}")
        
        return "\n".join(context_parts)
    
    def _build_system_prompt(self, topic: str = "") -> str:
        """Build the system prompt with terminology context."""
        base_prompt = self.prompt_config.get('system_prompt', 
            'You are a technical documentation expert creating HTML knowledge base articles.')
        
        # Add structure information if available
        if 'documentation_structure' in self.prompt_config:
            structure = self.prompt_config['documentation_structure']
            base_prompt += f"\n\nEach article should follow this structure:\n"
            base_prompt += "\n".join(f"- {section}" for section in structure)
        
        # Add terminology context
        terminology_context = self._build_terminology_context(topic)
        if terminology_context:
            base_prompt += f"\n\nRelevant HPC Environment Information:\n{terminology_context}"
            base_prompt += "\n\nWhen writing documentation, reference these specific modules, commands, and resources where appropriate. Use exact module names as listed above."
        
        # Add any terms/definitions from prompt config
        if 'terms' in self.prompt_config:
            base_prompt += "\n\nAdditional Key Terms:\n"
            for term, definition in self.prompt_config['terms'].items():
                base_prompt += f"- {term}: {definition}\n"
        
        return base_prompt
    
    def generate_documentation(self, query: str, runs: int = 5, 
                             model: str = 'gpt-4', 
                             temperature: float = 0.7,
                             topic_filename: str = None) -> List[str]:
        """Generate multiple documentation pages based on the query."""
        if topic_filename is None:
            topic_filename = self._extract_topic_from_query(query)
        
        # Extract topic for terminology context
        topic = self._extract_topic_from_query(query).replace('_', ' ')
        
        generated_files = []
        
        # Build messages with topic-aware system prompt
        system_prompt = self._build_system_prompt(topic)
        
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
                    filename = f'{topic_filename}_{model_name}_temp{temp_str}.html'
                else:
                    filename = f'{topic_filename}_{model_name}_temp{temp_str}_v{i+1}.html'                
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


class DocumentAnalyzer:
    """Analyze and extract sections from HTML documentation."""
    
    def __init__(self, section_headers: List[str] = None):
        self.section_headers = section_headers or [
            'Description', 'Installation', 'Usage', 'Examples', 'References'
        ]
        
    def extract_sections(self, html_content: str) -> Dict[str, str]:
        """Extract sections from HTML content based on headers."""
        soup = BeautifulSoup(html_content, 'html.parser')
        sections = {}
        
        # Find all headers (h1, h2, h3, etc.)
        headers = soup.find_all(['h1', 'h2', 'h3', 'h4'])
        
        for i, header in enumerate(headers):
            header_text = header.get_text().strip()
            
            # Check if this header matches any of our target sections
            for section_name in self.section_headers:
                if section_name.lower() in header_text.lower():
                    # Extract content between this header and the next
                    content_parts = []
                    
                    # Get all siblings until the next header
                    for sibling in header.find_next_siblings():
                        if sibling.name in ['h1', 'h2', 'h3', 'h4']:
                            break
                        content_parts.append(str(sibling))
                    
                    sections[section_name] = '\n'.join(content_parts)
                    break
        
        return sections
    
    def calculate_section_score(self, section_content: str, section_name: str) -> float:
        """Calculate a quality score for a section."""
        if not section_content:
            return 0.0
        
        soup = BeautifulSoup(section_content, 'html.parser')
        text = soup.get_text().strip()
        
        # Base score on multiple factors
        score = 0.0
        
        # Length (not too short, not too long)
        word_count = len(text.split())
        if section_name == "Description":
            ideal_length = 150
            score += max(0, 1 - abs(word_count - ideal_length) / ideal_length) * 20
        else:
            score += min(word_count / 100, 1) * 20  # Longer is generally better for other sections
        
        # Code examples (for Installation, Usage, Examples)
        if section_name in ["Installation", "Usage", "Examples"]:
            code_blocks = soup.find_all(['code', 'pre'])
            score += min(len(code_blocks) * 10, 30)
        
        # Lists (good for organization)
        lists = soup.find_all(['ul', 'ol'])
        score += min(len(lists) * 5, 15)
        
        # Links (good for References)
        if section_name == "References":
            links = soup.find_all('a')
            score += min(len(links) * 10, 30)
        else:
            links = soup.find_all('a')
            score += min(len(links) * 2, 10)
        
        # Formatting variety (bold, italic, etc.)
        formatting_tags = soup.find_all(['strong', 'em', 'b', 'i'])
        score += min(len(formatting_tags) * 2, 10)
        
        # Clarity (sentences not too long)
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        if 10 <= avg_sentence_length <= 25:
            score += 15
        
        return min(score, 100)  # Cap at 100


class GPTQualityEvaluator:
    """Evaluate documentation quality using GPT for subjective metrics."""
    
    def __init__(self, client, model='gpt-4'):
        self.client = client
        self.model = model
        
    def create_evaluation_prompt(self, section_content: str, section_name: str, 
                               topic: str, criteria: str) -> str:
        """Create a prompt for evaluating specific quality criteria."""
        
        prompts = {
            "technical_accuracy": f"""
Evaluate the technical accuracy of this {section_name} section about {topic}.
Consider:
- Are the commands, code examples, and technical details correct?
- Are version numbers, dependencies, and requirements accurate?
- Are there any outdated or incorrect technical statements?
- Would following these instructions actually work?

Section content:
{section_content}

Provide a score from 0-100 and a brief explanation (2-3 sentences).
Format: {{"score": NUMBER, "explanation": "..."}}
""",
            
            "writing_style": f"""
Evaluate the writing style and tone of this {section_name} section for academic/research computing documentation.
Consider:
- Is the tone appropriately professional and academic?
- Is it clear and accessible for graduate-level users?
- Does it avoid being too casual or too dense?
- Is the language consistent and well-structured?

Section content:
{section_content}

Provide a score from 0-100 and a brief explanation (2-3 sentences).
Format: {{"score": NUMBER, "explanation": "..."}}
""",
            
            "completeness": f"""
Evaluate the completeness of this {section_name} section about {topic}.
Consider:
- Does it cover all essential information for this section type?
- Are there important details or steps missing?
- For {section_name}, what key elements should be present?
- Does it answer the questions users would typically have?

Section content:
{section_content}

Provide a score from 0-100 and a brief explanation (2-3 sentences).
Format: {{"score": NUMBER, "explanation": "..."}}
"""
        }
        
        return prompts.get(criteria, "")
    
    def parse_gpt_response(self, response: str) -> Tuple[float, str]:
        """Parse the GPT response to extract score and explanation."""
        try:
            # Try to parse as JSON first
            result = json.loads(response)
            return result['score'], result['explanation']
        except:
            # Fallback: extract number and text
            import re
            score_match = re.search(r'\b(\d+)\b', response)
            score = float(score_match.group(1)) if score_match else 50.0
            
            # Extract explanation (everything after the score)
            explanation = response.split(str(int(score)), 1)[-1].strip()
            return score, explanation
    
    def evaluate_section(self, section_content: str, section_name: str, 
                        topic: str, criteria: List[str] = None) -> Dict[str, Dict]:
        """Evaluate a section on multiple criteria."""
        if criteria is None:
            criteria = ['technical_accuracy', 'writing_style', 'completeness']
        
        results = {}
        
        for criterion in criteria:
            if not section_content.strip():
                results[criterion] = {
                    'score': 0,
                    'explanation': 'Section is empty'
                }
                continue
            
            # Truncate very long sections
            max_chars = 3000
            content = section_content[:max_chars] + "..." if len(section_content) > max_chars else section_content
            
            prompt = self.create_evaluation_prompt(content, section_name, topic, criterion)
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert technical documentation reviewer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=150
                )
                
                score, explanation = self.parse_gpt_response(response.choices[0].message.content)
                results[criterion] = {
                    'score': score,
                    'explanation': explanation
                }
                
            except Exception as e:
                results[criterion] = {
                    'score': 0,
                    'explanation': f'Error: {str(e)}'
                }
            
            # Rate limiting
            time.sleep(0.5)
        
        return results


def load_and_analyze_versions(topic_filename: str, model: str, temperature: str, num_versions: int = 5):
    """Load all versions and extract their sections."""
    analyzer = DocumentAnalyzer()
    output_dir = Path('output')
    all_sections = {}
    
    for version in range(1, num_versions + 1):
        # Construct filename
        if num_versions == 1:
            filename = f'{topic_filename}_{model}_temp{temperature}.html'
        else:
            filename = f'{topic_filename}_{model}_temp{temperature}_v{version}.html'
        
        filepath = output_dir / filename
        
        if not filepath.exists():
            print(f"⚠️  File not found: {filepath}")
            continue
            
        # Load and extract sections
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        sections = analyzer.extract_sections(content)
        all_sections[f'Version {version}'] = sections
        
        print(f"✅ Loaded Version {version}: {filename}")
        print(f"   Found sections: {', '.join(sections.keys())}")
    
    return all_sections, analyzer


def analyze_sections(all_sections: Dict[str, Dict[str, str]], analyzer: DocumentAnalyzer) -> List[Dict]:
    """Analyze and score all sections across versions."""
    scores_data = []
    
    for section_name in analyzer.section_headers:
        section_scores = {}
        
        for version, sections in all_sections.items():
            if section_name in sections:
                score = analyzer.calculate_section_score(
                    sections[section_name], 
                    section_name
                )
                section_scores[version] = score
            else:
                section_scores[version] = 0
        
        # Find best version for this section
        if section_scores:
            best_version = max(section_scores, key=section_scores.get)
            best_score = section_scores[best_version]
        else:
            best_version = "N/A"
            best_score = 0
        
        scores_data.append({
            'Section': section_name,
            **section_scores,
            'Best Version': best_version,
            'Best Score': best_score
        })
    
    return scores_data


def compile_best_document(all_sections: Dict[str, Dict[str, str]], 
                         scores_data: List[Dict],
                         topic: str,
                         model: str,
                         temperature: float,
                         manual_overrides: Dict[str, str] = None) -> str:
    """Compile the best sections into a final document."""
    
    # Allow manual overrides if needed
    manual_overrides = manual_overrides or {}
    
    # Start building the final HTML
    html_parts = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f'<title>{topic} Documentation - Best Compilation</title>',
        '<style>',
        'body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; ',
        '       line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; }',
        'h1, h2, h3 { color: #2c3e50; }',
        'code { background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }',
        'pre { background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }',
        '.metadata { background-color: #e8f4f8; padding: 15px; border-radius: 5px; ',
        '            margin-bottom: 30px; font-size: 0.9em; }',
        '.section { margin-bottom: 40px; }',
        '.version-note { color: #7f8c8d; font-size: 0.85em; font-style: italic; }',
        '</style>',
        '</head>',
        '<body>',
        f'<h1>{topic} Documentation</h1>',
        '<div class="metadata">',
        f'<strong>Compiled from best sections across {len(all_sections)} versions</strong><br>',
        f'Generated using: {model} (Temperature: {temperature})<br>',
        f'Compilation date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '</div>'
    ]
    
    # Add each section
    for row in scores_data:
        section_name = row['Section']
        best_version = manual_overrides.get(section_name, row['Best Version'])
        
        if best_version != "N/A" and best_version in all_sections:
            if section_name in all_sections[best_version]:
                html_parts.append('<div class="section">')
                html_parts.append(f'<h2>{section_name}</h2>')
                html_parts.append(f'<span class="version-note">From {best_version}</span>')
                html_parts.append(all_sections[best_version][section_name])
                html_parts.append('</div>')
    
    # Close HTML
    html_parts.extend([
        '<div class="metadata" style="margin-top: 50px;">',
        '<strong>Section Sources:</strong><br>',
    ])
    
    # Add section source summary
    for row in scores_data:
        section_name = row['Section']
        best_version = manual_overrides.get(section_name, row['Best Version'])
        score = row['Best Score']
        html_parts.append(f'{section_name}: {best_version} (Score: {score:.1f})<br>')
    
    html_parts.extend([
        '</div>',
        '</body>',
        '</html>'
    ])
    
    return '\n'.join(html_parts)


def generate_analysis_report(topic: str, model: str, temperature: float, 
                           all_sections: Dict, scores_data: List[Dict]) -> str:
    """Generate a detailed analysis report of the compilation process."""
    
    report = []
    report.append(f"# Documentation Analysis Report for {topic}")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Model: {model}, Temperature: {temperature}")
    report.append(f"Analyzed {len(all_sections)} versions\n")
    
    report.append("## Section Scores\n")
    report.append("| Section | " + " | ".join(all_sections.keys()) + " | Best Version | Best Score |")
    report.append("|---------|" + "---|" * (len(all_sections) + 2))
    
    for row in scores_data:
        line = f"| {row['Section']} | "
        for version in all_sections.keys():
            score = row.get(version, 0)
            line += f"{score:.1f} | "
        line += f"{row['Best Version']} | {row['Best Score']:.1f} |"
        report.append(line)
    
    report.append("\n## Key Findings\n")
    
    # Find most consistent version
    version_totals = {}
    for version in all_sections.keys():
        total = sum(row.get(version, 0) for row in scores_data)
        version_totals[version] = total
    
    if version_totals:
        best_overall = max(version_totals, key=version_totals.get)
        report.append(f"- **Best Overall Version**: {best_overall} (Total Score: {version_totals[best_overall]:.1f})")
    
    # Find sections with high variance
    report.append("\n### Section Quality Variance")
    for row in scores_data:
        version_scores = [row.get(v, 0) for v in all_sections.keys()]
        if version_scores:
            variance = max(version_scores) - min(version_scores)
            if variance > 20:
                report.append(f"- **{row['Section']}**: High variance ({variance:.1f} points) - quality varies significantly between versions")
    
    report.append("\n## Recommendations")
    report.append("- Review sections with high variance manually")
    report.append("- Consider regenerating sections with scores below 50")
    report.append("- The compiled document uses the best scoring section from each category")
    
    return '\n'.join(report)


def compare_versions(topic_filename: str, model: str, temperature: str, runs: int):
    """Compare key differences between generated versions."""
    output_dir = Path('output')
    
    # Build filename pattern
    model_clean = model.replace('-', '').replace('.', '')
    temp_str = str(temperature).replace('.', '')
    
    if runs == 1:
        pattern = f'{topic_filename}_{model_clean}_temp{temp_str}.html'
        files = [output_dir / pattern] if (output_dir / pattern).exists() else []
    else:
        files = [output_dir / f'{topic_filename}_{model_clean}_temp{temp_str}_v{i}.html' 
                for i in range(1, runs + 1) if (output_dir / f'{topic_filename}_{model_clean}_temp{temp_str}_v{i}.html').exists()]
    
    if len(files) < 2:
        print("Need at least 2 versions to compare")
        return
    
    print(f"📊 Comparing {len(files)} versions of documentation:\n")
    
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


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Generate technical documentation using OpenAI GPT models')
    parser.add_argument('--topic', required=True, help='Topic for documentation generation')
    parser.add_argument('--runs', type=int, default=5, help='Number of variations to generate (1-10)')
    parser.add_argument('--model', default='gpt-4o-mini', help='OpenAI model to use')
    parser.add_argument('--temperature', type=float, default=0.7, help='Creativity level (0.0-1.0)')
    parser.add_argument('--query-template', default="Create a knowledge base article with regards to using {topic} on the FASRC cluster, using the tone of graduate level Academic Computing documentation.", 
                        help='Query template (use {topic} placeholder)')
    parser.add_argument('--analyze', action='store_true', help='Run analysis on generated files')
    parser.add_argument('--compare', action='store_true', help='Compare generated versions')
    parser.add_argument('--install-deps', action='store_true', help='Install required dependencies')
    parser.add_argument('--skip-generation', action='store_true', help='Skip generation and only run analysis')
    parser.add_argument('--terminology', default='terminology.yaml', help='Path to terminology configuration file')
    parser.add_argument('--scan-code-examples', metavar='PATH', help='Scan filesystem path for code examples and update terminology file')
    parser.add_argument('--update-code-examples', action='store_true', help='Update existing code examples in terminology file')
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        print("📦 Installing dependencies...")
        install_dependencies()
        print("✅ Dependencies installed")
    
    # Load environment variables
    load_dotenv()
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables!")
        print("   Please create a .env file with: OPENAI_API_KEY=your-key-here")
        return 1
    
    # Generate topic filename
    topic_filename = args.topic.lower().replace(' ', '_')
    
    generated_files = []
    
    if not args.skip_generation:
        # Initialize generator
        print("🔧 Initializing documentation generator...")
        try:
            generator = DocumentationGenerator(
                prompt_yaml_path='prompt.yaml',
                examples_dir='examples/',
                terminology_path=args.terminology
            )
            print("✅ Generator initialized successfully")
            print(f"📁 Found {len(generator.examples)} examples")
        except Exception as e:
            print(f"❌ Error initializing generator: {e}")
            return 1
        
        # Build the query from template
        query = args.query_template.format(topic=args.topic)
        
        print(f"\n{'='*60}")
        print(f"📝 Generating documentation for: {args.topic}")
        print(f"📋 Query: {query}")
        print(f"🔄 Generating {args.runs} variations...")
        print(f"{'='*60}\n")
        
        # Track generation time
        start_time = datetime.now()
        
        # Generate the documentation
        generated_files = generator.generate_documentation(
            query=query,
            runs=args.runs,
            model=args.model,
            temperature=args.temperature,
            topic_filename=topic_filename
        )
        
        # Calculate elapsed time
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\n{'='*60}")
        print(f"✅ Generation complete!")
        print(f"⏱️  Time taken: {elapsed:.2f} seconds")
        print(f"📁 Generated {len(generated_files)} files:")
        for file in generated_files:
            print(f"   - {file}")
        print(f"{'='*60}")
    
    # Run comparison if requested
    if args.compare:
        print(f"\n📊 Running comparison...")
        compare_versions(topic_filename, args.model, args.temperature, args.runs)
    
    # Run analysis if requested
    if args.analyze:
        print(f"\n🔍 Running analysis...")
        
        # Load and analyze versions
        model_clean = args.model.replace('-', '').replace('.', '')
        temp_str = str(args.temperature).replace('.', '')
        
        all_sections, analyzer = load_and_analyze_versions(
            topic_filename=topic_filename,
            model=model_clean,
            temperature=temp_str,
            num_versions=args.runs
        )
        
        if not all_sections:
            print("⚠️  No files found for analysis")
            return 1
        
        # Analyze sections
        scores_data = analyze_sections(all_sections, analyzer)
        
        print("\n📊 Section Analysis Results:")
        print("="*80)
        
        # Create a simple table if pandas is not available
        if pd is not None:
            df = pd.DataFrame(scores_data)
            print(df.to_string(index=False))
        else:
            # Simple table format
            print(f"{'Section':<12} {'Best Version':<15} {'Best Score':<10}")
            print("-" * 40)
            for row in scores_data:
                print(f"{row['Section']:<12} {row['Best Version']:<15} {row['Best Score']:<10.1f}")
        
        print("="*80)
        
        print("\n🏆 Best Version for Each Section:")
        for row in scores_data:
            print(f"   {row['Section']}: {row['Best Version']} (Score: {row['Best Score']:.1f})")
        
        # Generate best compilation
        best_document_html = compile_best_document(
            all_sections, scores_data, args.topic, args.model, args.temperature
        )
        
        # Save the best compilation
        best_output_path = Path('output') / f'{topic_filename}_best_compilation.html'
        with open(best_output_path, 'w', encoding='utf-8') as f:
            f.write(best_document_html)
        
        # Generate analysis report
        report_content = generate_analysis_report(
            args.topic, args.model, args.temperature, all_sections, scores_data
        )
        report_path = Path('output') / f'{topic_filename}_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n📄 Analysis complete!")
        print(f"✅ Best compilation: {best_output_path}")
        print(f"📊 Analysis report: {report_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())