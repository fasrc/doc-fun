"""
Documentation Generator Core Module
Version 2.6.0-dev

Advanced documentation generation with plugin-based recommendation system.
"""

__version__ = "2.6.0-dev"

import os
import re
import json
import time
import yaml
import hashlib
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from bs4 import BeautifulSoup

from .plugin_manager import PluginManager
from .providers import ProviderManager, CompletionRequest
from .config import get_settings
from .exceptions import DocGeneratorError, ConfigurationError, ProviderError, FileOperationError
from .error_handler import ErrorHandler, retry_on_failure, handle_gracefully
from .cache import cached, get_cache_manager
from .command_tracker import CommandTracker
from .utils import get_output_directory
# Backward compatibility imports - Classes moved to separate modules in Phase 2
from .generator import DocumentationGenerator
from .analyzer import DocumentAnalyzer
from .quality_evaluator import GPTQualityEvaluator
from .code_scanner import CodeExampleScanner

try:
    import pandas as pd
except ImportError:
    pd = None











def load_and_analyze_versions(topic_filename: str, model: str, temperature: str, num_versions: int = 5, output_dir: str = 'output'):
    """Load all versions and extract their sections."""
    analyzer = DocumentAnalyzer()
    output_path = Path(output_dir)
    all_sections = {}
    
    for version in range(1, num_versions + 1):
        # Construct filename
        if num_versions == 1:
            filename = f'{topic_filename}_{model}_temp{temperature}.html'
        else:
            filename = f'{topic_filename}_{model}_temp{temperature}_v{version}.html'
        
        filepath = output_path / filename
        
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


def compare_versions(topic_filename: str, model: str, temperature: str, runs: int, output_dir: str = 'output'):
    """Compare key differences between generated versions."""
    output_path = Path(output_dir)
    
    # Build filename pattern
    model_clean = model.replace('-', '').replace('.', '')
    temp_str = str(temperature).replace('.', '')
    
    if runs == 1:
        pattern = f'{topic_filename}_{model_clean}_temp{temp_str}.html'
        files = [output_path / pattern] if (output_path / pattern).exists() else []
    else:
        files = [output_path / f'{topic_filename}_{model_clean}_temp{temp_str}_v{i}.html' 
                for i in range(1, runs + 1) if (output_path / f'{topic_filename}_{model_clean}_temp{temp_str}_v{i}.html').exists()]
    
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
    parser.add_argument('--generator-prompt', default='./prompts/generator/default.yaml', help='Path to generator prompt configuration file')
    parser.add_argument('--analysis-prompt', default='./prompts/analysis/default.yaml', help='Path to analysis prompt configuration file')
    parser.add_argument('--output-dir', default='output', help='Output directory for generated files (default: output)')
    
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
    
    # Handle code example scanning
    if args.scan_code_examples:
        print(f"🔍 Scanning code examples from: {args.scan_code_examples}")
        scanner = CodeExampleScanner()
        try:
            code_examples = scanner.scan_directory(args.scan_code_examples)
            scanner.update_terminology_file(args.terminology, code_examples)
            print("✅ Code example scanning complete")
            return 0
        except Exception as e:
            print(f"❌ Error scanning code examples: {e}")
            return 1
    
    # Handle code example updates
    if args.update_code_examples:
        print(f"🔄 Checking for code example updates...")
        scanner = CodeExampleScanner()
        try:
            updated_files = scanner.check_for_updates(args.terminology)
            if updated_files:
                print(f"📝 Found {len(updated_files)} updated files:")
                for file_path in updated_files:
                    print(f"   - {file_path}")
                
                # Ask user if they want to rescan
                response = input("Rescan updated files? [y/N]: ").lower()
                if response == 'y':
                    # Rescan only the directories containing updated files
                    scan_dirs = set(Path(f).parent for f in updated_files)
                    all_examples = []
                    for scan_dir in scan_dirs:
                        examples = scanner.scan_directory(str(scan_dir))
                        all_examples.extend(examples)
                    scanner.update_terminology_file(args.terminology, all_examples)
                    print("✅ Code examples updated")
            else:
                print("✅ All code examples are up to date")
            return 0
        except Exception as e:
            print(f"❌ Error checking code example updates: {e}")
            return 1
    
    # Generate topic filename
    topic_filename = args.topic.lower().replace(' ', '_')
    
    # Get the proper output directory (with timestamp if using default)
    logger = logging.getLogger(__name__)
    output_dir = get_output_directory(args.output_dir, logger)
    
    generated_files = []
    
    if not args.skip_generation:
        # Initialize generator
        print("🔧 Initializing documentation generator...")
        try:
            generator = DocumentationGenerator(
                prompt_yaml_path=args.generator_prompt,
                shots_dir='shots/',
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
            topic_filename=topic_filename,
            output_dir=output_dir
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
        compare_versions(topic_filename, args.model, args.temperature, args.runs, output_dir)
    
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
            num_versions=args.runs,
            output_dir=output_dir
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
        best_output_path = Path(output_dir) / f'{topic_filename}_best_compilation.html'
        with open(best_output_path, 'w', encoding='utf-8') as f:
            f.write(best_document_html)
        
        # Save command file for the best compilation
        command_file = CommandTracker.save_command_file(str(best_output_path))
        if command_file:
            print(f"✓ Command saved: {Path(command_file).name}")
        
        # Generate analysis report
        report_content = generate_analysis_report(
            args.topic, args.model, args.temperature, all_sections, scores_data
        )
        report_path = Path(output_dir) / f'{topic_filename}_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n📄 Analysis complete!")
        print(f"✅ Best compilation: {best_output_path}")
        print(f"📊 Analysis report: {report_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())