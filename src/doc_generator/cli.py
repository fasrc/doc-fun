#!/usr/bin/env python3
"""
Command-line interface for doc_generator package.
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
from dotenv import load_dotenv

from .core import DocumentationGenerator, DocumentAnalyzer, GPTQualityEvaluator, CodeExampleScanner
from .config import get_settings
from .exceptions import DocGeneratorError, DocumentStandardizerError
from . import __version__
from .utils import get_output_directory
from .agents.token_machine import TokenMachine, AnalysisDepth, analyze_operation


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration using settings."""
    settings = get_settings()
    
    # Override log level if verbose is specified
    if verbose:
        level = logging.DEBUG
    else:
        level = getattr(logging, settings.log_level.upper())
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('doc_generator')




def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog='doc-gen',
        description='Generate technical documentation using AI with plugin-based recommendations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Documentation Generation (--topic mode)
  doc-gen --topic "Python Programming" --runs 3 --analyze
  doc-gen --topic "CUDA Programming" --model gpt-4 --temperature 0.7
  doc-gen --topic "Machine Learning" --prompt-yaml ./prompts/custom.yaml
  
  # README Generation (--readme mode)
  doc-gen --readme /path/to/directory --runs 2 --analyze
  doc-gen --readme /path/to/directory --recursive --output-dir ./output
  doc-gen --readme /path/to/directory --model claude-3-5-sonnet
  
  # Document Standardization (--standardize mode)
  doc-gen --standardize existing-docs.html --format markdown
  doc-gen --standardize /path/to/doc.html --template api_documentation
  doc-gen --standardize legacy.md --output-dir ./standardized
  
  # Token Analysis (--token-* modes)
  doc-gen --token-analyze "Generate Python documentation" --content ./example.py
  doc-gen --token-estimate "README generation for React project"
  doc-gen --token-report --period 7
  doc-gen --token-optimize "Batch documentation generation"
  
  # Legacy Code Scanning
  doc-gen --scan-code ./directory --generate-readme --recursive
  
  # Utility Commands
  doc-gen --list-models
  doc-gen --list-plugins
  doc-gen --cleanup
  doc-gen --info
        """
    )
    
    # === MAIN OPERATION MODES ===
    mode_group = parser.add_argument_group('Main Operation Modes', 'Choose one primary mode')
    mode_group.add_argument('--topic', 
                           help='Generate HTML/Markdown documentation for a specific topic')
    mode_group.add_argument('--readme', metavar='DIRECTORY',
                           help='Generate README.md for directory using unified pipeline')
    mode_group.add_argument('--standardize', metavar='FILE',
                           help='Standardize existing documentation to organizational standards')
    mode_group.add_argument('--scan-code', nargs='?', const='.', metavar='DIR',
                           help='Scan directory for code examples and update terminology')
    
    # === TOKEN ANALYSIS MODES ===
    token_group = parser.add_argument_group('Token Analysis Modes', 'Token usage analysis and optimization')
    token_group.add_argument('--token-analyze', metavar='OPERATION',
                            help='Analyze token usage for a specific operation')
    token_group.add_argument('--token-estimate', metavar='OPERATION',
                            help='Quick token estimation for an operation')
    token_group.add_argument('--token-report', action='store_true',
                            help='Generate token usage report')
    token_group.add_argument('--token-optimize', metavar='OPERATION',
                            help='Get optimization recommendations for an operation')
    
    # === TOKEN ANALYSIS OPTIONS ===
    token_opts_group = parser.add_argument_group('Token Analysis Options', 'Used with token analysis modes')
    token_opts_group.add_argument('--content', metavar='FILE_OR_TEXT',
                                 help='Content to analyze (file path or text)')
    token_opts_group.add_argument('--content-type', choices=['plain_text', 'code', 'markdown', 'json', 'yaml', 'html'],
                                 default='plain_text',
                                 help='Type of content being analyzed (default: plain_text)')
    token_opts_group.add_argument('--expected-output', type=int, metavar='CHARS',
                                 help='Expected output size in characters')
    token_opts_group.add_argument('--context-size', type=int, metavar='CHARS',
                                 help='Additional context size in characters')
    token_opts_group.add_argument('--period', type=int, default=7, metavar='DAYS',
                                 help='Report period in days (default: 7)')
    token_opts_group.add_argument('--analysis-depth', choices=['minimal', 'standard', 'comprehensive'],
                                 default='standard',
                                 help='Depth of token analysis (default: standard)')
    token_opts_group.add_argument('--max-cost', type=float, metavar='USD',
                                 help='Maximum cost constraint for model recommendations')
    token_opts_group.add_argument('--min-quality', type=float, metavar='SCORE',
                                 help='Minimum quality score (0-1) for model recommendations')
    
    # === SHARED GENERATION OPTIONS ===
    gen_group = parser.add_argument_group('Generation Options', 'Used with --topic or --readme')
    gen_group.add_argument('--runs', type=int, default=1,
                          help='Number of variants to generate (default: 1, README mode: 3)')
    gen_group.add_argument('--model', 
                          help='Model to use (e.g., gpt-4o-mini, claude-3-5-sonnet-20240620)')
    gen_group.add_argument('--provider', choices=['openai', 'claude', 'auto'], 
                          default='auto',
                          help='LLM provider to use (default: auto-detect)')
    gen_group.add_argument('--temperature', type=float, default=0.3,
                          help='Temperature for text generation (default: 0.3)')
    gen_group.add_argument('--output-dir', default='output',
                          help='Output directory (default: ./output/timestamp for README)')
    
    # === TOPIC MODE OPTIONS ===
    topic_group = parser.add_argument_group('Topic Mode Options', 'Used with --topic')
    topic_group.add_argument('--format', choices=['html', 'markdown', 'auto'],
                            default='auto',
                            help='Output format for documentation (default: auto-detect)')
    topic_group.add_argument('--prompt-yaml-path', '--prompt-yaml', 
                            default='./prompts/generator/default.yaml',
                            help='Path to prompt YAML configuration')
    topic_group.add_argument('--quality-eval', action='store_true',
                            help='Run GPT-based quality evaluation')
    topic_group.add_argument('--compare-url', metavar='URL',
                            help='Compare with existing documentation at URL')
    topic_group.add_argument('--compare-file', metavar='FILE',
                            help='Compare with local file')
    token_group.add_argument('--comparison-report', metavar='FILE',
                            help='Save comparison report to file')
    
    # === README MODE OPTIONS ===
    readme_group = parser.add_argument_group('README Mode Options', 'Used with --readme')
    readme_group.add_argument('--recursive', action='store_true',
                             help='Generate README files for all subdirectories')
    
    # === STANDARDIZATION OPTIONS ===
    std_group = parser.add_argument_group('Standardization Options', 'Used with --standardize')
    std_group.add_argument('--template', choices=['technical_documentation', 'user_guide', 'api_documentation'],
                          default='technical_documentation',
                          help='Standardization template to use (default: technical_documentation)')
    std_group.add_argument('--target-format', choices=['html', 'markdown'],
                          help='Target format for standardized output (overrides --format)')
    
    # === LEGACY CODE SCANNING ===
    legacy_group = parser.add_argument_group('Legacy Code Scanning', 'Used with --scan-code')
    legacy_group.add_argument('--generate-readme', action='store_true',
                             help='Generate README files (legacy mode - use --readme instead)')
    legacy_group.add_argument('--max-scan-files', type=int, default=50,
                             help='Maximum files to scan for code examples (default: 50)')
    legacy_group.add_argument('--overwrite', action='store_true',
                             help='Overwrite existing README.md files')
    legacy_group.add_argument('--suffix', metavar='SUFFIX',
                             help='Custom suffix for generated README files (default: _generated)')
    legacy_group.add_argument('--ai-enhance', action='store_true',
                             help='Use AI to enhance README descriptions')
    
    # === ANALYSIS AND CONFIGURATION ===
    analysis_group = parser.add_argument_group('Analysis and Configuration')
    analysis_group.add_argument('--analyze', action='store_true',
                               help='Run document analysis after generation')
    analysis_group.add_argument('--analysis-prompt-path', 
                               default='./prompts/analysis/default.yaml',
                               help='Path to analysis prompt configuration')
    analysis_group.add_argument('--report-format', choices=['markdown', 'html', 'json'],
                               default='markdown',
                               help='Format for analysis reports (default: markdown)')
    analysis_group.add_argument('--shots', metavar='PATH',
                               help='Path to few-shot examples directory')
    analysis_group.add_argument('--examples-dir', default='shots',
                               help='Directory containing few-shot examples (default: shots)')
    analysis_group.add_argument('--terminology-path', default='terminology.yaml',
                               help='Path to terminology YAML file (default: terminology.yaml)')
    
    # === PLUGIN MANAGEMENT ===
    plugin_group = parser.add_argument_group('Plugin Management')
    plugin_group.add_argument('--list-plugins', action='store_true',
                             help='List all available plugins and exit')
    plugin_group.add_argument('--disable-plugins', nargs='*', metavar='PLUGIN',
                             help='Disable specific plugins by name')
    plugin_group.add_argument('--enable-only', nargs='*', metavar='PLUGIN',
                             help='Enable only specified plugins (disable all others)')
    
    # === UTILITY COMMANDS ===
    utility_group = parser.add_argument_group('Utility Commands')
    utility_group.add_argument('--list-models', action='store_true',
                              help='List available models and providers')
    utility_group.add_argument('--cleanup', action='store_true',
                              help='Remove all files and directories in ./output/')
    utility_group.add_argument('--info', action='store_true',
                              help='Display detailed information about all options')
    utility_group.add_argument('--verbose', '-v', action='store_true',
                              help='Enable verbose logging')
    utility_group.add_argument('--quiet', '-q', action='store_true',
                              help='Suppress non-essential output')
    utility_group.add_argument('--version', action='version', version=f'doc-generator {__version__}')
    
    return parser.parse_args()


def list_plugins(generator: DocumentationGenerator) -> None:
    """List all available plugins."""
    print("Available Recommendation Engine Plugins:")
    print("=" * 50)
    
    engines = generator.plugin_manager.list_engines()
    
    if not engines:
        print("No plugins loaded.")
        return
    
    for engine in engines:
        print(f"\nPlugin: {engine['name']}")
        print(f"  Class: {engine['class']}")
        print(f"  Module: {engine['module']}")
        print(f"  Supported Types: {', '.join(engine['supported_types'])}")
        print(f"  Priority: {engine['priority']}")
        print(f"  Enabled: {engine['enabled']}")


def run_standardization(args, logger: logging.Logger) -> None:
    """Run document standardization."""
    from .standardizers import DocumentStandardizer
    from pathlib import Path
    
    logger.info(f"Standardizing document: {args.standardize}")
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Validate input file
        input_file = Path(args.standardize)
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            sys.exit(1)
        
        # Determine output format
        target_format = args.target_format or args.format
        if target_format == 'auto':
            # Auto-detect based on input file extension
            if input_file.suffix.lower() in ['.html', '.htm']:
                target_format = 'markdown'  # Convert HTML to Markdown by default
            else:
                target_format = 'html'  # Default to HTML
        
        # Initialize standardizer
        standardizer = DocumentStandardizer(
            provider=args.provider if args.provider != 'auto' else None,
            model=args.model,
            temperature=args.temperature,
            output_format=target_format
        )
        
        # Determine output directory and file
        output_dir = get_output_directory(args.output_dir, logger)
        
        # Generate output filename
        stem = input_file.stem
        if target_format == 'markdown':
            ext = '.md'
        elif target_format == 'html':
            ext = '.html'
        else:
            ext = '.txt'
        
        output_file = Path(output_dir) / f"{stem}_standardized{ext}"
        
        # Standardize document
        logger.info(f"Using template: {args.template}")
        logger.info(f"Target format: {target_format}")
        
        result = standardizer.standardize_file(
            file_path=input_file,
            output_path=output_file,
            target_format=target_format
        )
        
        # Report results
        if not args.quiet:
            print(f"\n📄 Document Standardization Results")
            print(f"📍 Input File: {input_file}")
            print(f"📂 Output File: {result.get('output_path', output_file)}")
            print(f"🔄 Format: {result['original_format']} → {result['target_format']}")
            print(f"📊 Sections Processed: {len(result['sections_processed'])}")
            
            if result['sections_processed']:
                print(f"📋 Sections: {', '.join(result['sections_processed'])}")
            
            metadata = result.get('metadata', {})
            if metadata.get('provider'):
                print(f"🤖 Provider: {metadata['provider']}")
                print(f"🧠 Model: {metadata.get('model', 'Unknown')}")
                if metadata.get('tokens_used'):
                    print(f"🎯 Tokens Used: {metadata['tokens_used']}")
        
        logger.info("Document standardization completed successfully")
        
    except Exception as e:
        logger.error(f"Error during standardization: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_readme_generation(args, logger: logging.Logger) -> None:
    """Run README generation using the unified pipeline."""
    from .readme_documentation_generator import ReadmeDocumentationGenerator
    
    logger.info(f"Generating README for directory: {args.readme}")
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Use --shots path if provided, otherwise use default
        shots_dir = args.shots if args.shots else args.examples_dir
        
        # Initialize the README generator
        generator = ReadmeDocumentationGenerator(
            prompt_yaml_path=args.prompt_yaml_path if 'readme' in args.prompt_yaml_path 
                            else './prompts/generator/user_codes_topic_readme.yaml',
            shots_dir=shots_dir,
            terminology_path=args.terminology_path,
            provider=args.provider,
            logger=logger
        )
        
        # Set README-specific analysis prompt if using default
        analysis_prompt_path = args.analysis_prompt_path
        if analysis_prompt_path == './prompts/analysis/default.yaml':
            analysis_prompt_path = './prompts/analysis/readme.yaml'
        
        # Determine output directory using helper function
        output_dir = get_output_directory(args.output_dir, logger)
        
        # Generate README with multiple runs and analysis
        results = generator.generate_readme_documentation(
            directory_path=args.readme,
            runs=args.runs if args.runs > 1 else 3,  # Default to 3 runs for quality
            model=args.model,
            temperature=args.temperature,
            analyze=args.analyze or args.runs > 1,  # Auto-analyze if multiple runs
            provider=args.provider if args.provider != 'auto' else None,
            output_dir=output_dir
        )
        
        # Process results
        generated_files = results['generated_files']
        logger.info(f"Generated {len(generated_files)} README variations")
        
        if not args.quiet:
            print(f"\n📁 README Generation Results for: {results['directory_info']['name']}")
            print(f"📍 Source Path: {results['directory_info']['path']}")
            print(f"📂 Output Directory: {output_dir}")
            print(f"📊 Depth Level: {results['depth_level']}")
            print(f"💻 Languages Found: {', '.join(results['directory_info']['languages']) or 'None'}")
            print(f"📂 Subdirectories: {len(results['directory_info']['subdirectories'])}")
            print(f"📄 Files: {len(results['directory_info']['files'])}")
            
            print("\n📝 Generated Files:")
            for file_path in generated_files:
                file_name = Path(file_path).name
                if 'best' in file_name:
                    print(f"  ⭐ {file_name} (best compilation)")
                else:
                    print(f"  📄 {file_name}")
            
            if results.get('analysis_results'):
                analysis = results['analysis_results']
                print("\n📈 Quality Analysis:")
                
                if analysis.get('overall_best'):
                    best = analysis['overall_best']
                    print(f"  🏆 Best Version: {Path(best['file']).name}")
                    print(f"  📊 Total Score: {best['total_score']:.2f}")
                
                if analysis.get('section_winners'):
                    print("\n🎯 Section Winners:")
                    for section, winner in analysis['section_winners'].items():
                        print(f"  • {section}: Version {winner['version'] + 1} (score: {winner['score']:.2f})")
        
        # Handle recursive generation if requested
        if args.recursive:
            logger.info("Processing subdirectories recursively...")
            subdirs = results['directory_info'].get('subdirectories', [])
            
            if subdirs:
                print(f"\n🔄 Processing {len(subdirs)} subdirectories recursively...")
                
                for i, subdir in enumerate(subdirs, 1):
                    subdir_path = subdir['path']
                    print(f"  [{i}/{len(subdirs)}] {subdir['name']}")
                    
                    try:
                        sub_results = generator.generate_readme_documentation(
                            directory_path=subdir_path,
                            runs=args.runs if args.runs > 1 else 2,  # Fewer runs for subdirs
                            model=args.model,
                            temperature=args.temperature,
                            analyze=args.analyze or args.runs > 1,
                            provider=args.provider if args.provider != 'auto' else None,
                            output_dir=subdir_path  # Save in the subdirectory
                        )
                        
                        if not args.quiet:
                            print(f"    ✅ Generated {len(sub_results['generated_files'])} files")
                            
                    except Exception as e:
                        logger.error(f"Error processing subdirectory {subdir_path}: {e}")
                        print(f"    ❌ Failed: {e}")
                        if args.verbose:
                            import traceback
                            traceback.print_exc()
            else:
                print("  No subdirectories found for recursive processing.")
        
        print(f"\n✅ README generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error generating README: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_comparison(args, results: list, logger: logging.Logger) -> None:
    """Run documentation comparison."""
    from .evaluator import DocumentationComparator
    
    comparator = DocumentationComparator()
    
    # Get the first generated file for comparison
    if not results:
        logger.error("No generated files to compare")
        return
    
    generated_file = results[0]  # results contains file paths directly
    if not generated_file:
        logger.error("Generated file path not found")
        return
    
    # Build full path  
    generated_path = Path(generated_file)
    
    try:
        if args.compare_url:
            logger.info(f"Comparing with URL: {args.compare_url}")
            # Use existing generated file for comparison
            comparison_results = comparator.compare_existing_files(
                str(generated_path),
                args.compare_url
            )
        elif args.compare_file:
            logger.info(f"Comparing with file: {args.compare_file}")
            comparison_results = comparator.compare_existing_files(
                str(generated_path),
                args.compare_file
            )
        else:
            return
        
        # Generate report
        report = comparator.generate_report(
            comparison_results,
            args.comparison_report
        )
        
        if not args.quiet:
            print("\n" + "="*60)
            print("COMPARISON RESULTS")
            print("="*60)
            print(f"Composite Score: {comparison_results['scores']['composite_score']:.2%}")
            print(f"Content Similarity: {comparison_results['scores']['content_similarity']:.2%}")
            print(f"Structural Similarity: {comparison_results['scores']['structural_similarity']:.2%}")
            print(f"Code Similarity: {comparison_results['scores']['code_similarity']:.2%}")
            
            print("\nRecommendations:")
            for i, rec in enumerate(comparison_results['recommendations'][:3], 1):
                print(f"  {i}. {rec}")
            
            if args.comparison_report:
                print(f"\nDetailed report saved to: {args.comparison_report}")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


def display_info() -> None:
    """Display detailed information about all CLI options."""
    info_text = f"""
{'='*80}
📚 DOC-GENERATOR DETAILED OPTION GUIDE
{'='*80}

Version: {__version__}

This guide provides comprehensive information about all available options,
including usage examples and best practices.

{'='*80}
🎯 MAIN OPERATION MODES
{'='*80}

--topic "Topic Name"
    Generate technical documentation for a specific topic.
    
    📝 Description:
    Creates HTML or Markdown documentation about any technical topic using
    AI-powered generation with few-shot examples.
    
    ✨ Examples:
    doc-gen --topic "Machine Learning" --runs 3 --analyze
    doc-gen --topic "Quantum Computing" --model gpt-4o --temperature 0.7
    
    💡 Best Practices:
    - Use quotes for multi-word topics
    - Combine with --runs 3+ for quality through variation
    - Add --analyze to get best compilation from multiple runs

--readme /path/to/directory
    Generate README.md for a directory structure.
    
    📝 Description:
    Analyzes directory contents and creates comprehensive README documentation
    with automatic structure detection and contextualization.
    
    ✨ Examples:
    doc-gen --readme ./my-project --runs 3 --analyze
    doc-gen --readme /home/user/code --recursive --output-dir ./docs
    
    💡 Best Practices:
    - Default 3 runs for quality (automatic with --readme)
    - Use --recursive for nested directory documentation
    - Analysis is auto-enabled with multiple runs

--scan-code /path/to/directory
    Scan directory for code examples and update terminology.
    
    📝 Description:
    Analyzes code files to extract patterns, update terminology database,
    and optionally generate README files (legacy mode).
    
    ✨ Examples:
    doc-gen --scan-code ./src --max-scan-files 100
    doc-gen --scan-code . --generate-readme --recursive
    
    ⚠️ Note: --generate-readme is deprecated, use --readme instead

{'='*80}
🔧 GENERATION OPTIONS
{'='*80}

--runs N (default: 1, README mode: 3)
    Number of documentation variants to generate.
    
    📝 Description:
    Generates multiple versions of documentation. When N > 1, automatically
    enables analysis to create a "best compilation" combining the strongest
    sections from each variant.
    
    ✨ How it Works:
    - Run 1: machine_learning_readme_v1.md
    - Run 2: machine_learning_readme_v2.md  
    - Run 3: machine_learning_readme_v3.md
    - Analysis: machine_learning_readme_best.md (best of all versions)
    
    💡 Recommendations:
    - Use 3-5 runs for important documentation
    - Single run for quick drafts
    - More runs = better final quality through selection

--model MODEL_NAME
    Specify the AI model to use.
    
    📝 Available Models:
    OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
    Claude: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
    
    ✨ Examples:
    --model gpt-4o-mini         # Fast, cost-effective
    --model gpt-4o              # High quality, slower
    --model claude-3-5-sonnet   # Claude's best model
    
    💡 Model Selection:
    - gpt-4o-mini: Best for drafts and iterations
    - gpt-4o: Best for final documentation
    - claude-3-5-sonnet: Alternative high-quality option

--temperature FLOAT (default: 0.3)
    Control randomness/creativity (0.0-2.0).
    
    📝 Description:
    Lower values (0.0-0.5): More focused, deterministic
    Medium values (0.5-1.0): Balanced creativity
    Higher values (1.0-2.0): More creative, varied
    
    💡 Recommendations:
    - 0.3: Technical documentation (default)
    - 0.7: Creative examples and explanations
    - 1.0+: Brainstorming and exploration

--provider PROVIDER (default: auto)
    Choose LLM provider: openai, claude, or auto.
    
    📝 Description:
    Explicitly select provider or let system auto-detect based on
    available API keys.
    
    ✨ Examples:
    --provider openai  # Force OpenAI
    --provider claude  # Force Anthropic Claude
    --provider auto    # Auto-detect (default)

--output-dir PATH (default: ./output)
    Specify output directory for generated files.
    
    📝 Behavior:
    - Topic mode: Uses specified directory
    - README mode: Creates timestamped subdirectory if default
    
    ✨ Examples:
    --output-dir ./docs
    --output-dir /home/user/documentation

{'='*80}
📊 ANALYSIS & QUALITY
{'='*80}

--analyze
    Run document analysis after generation.
    
    📝 Description:
    Analyzes generated documentation for:
    - Section completeness and structure
    - Content quality metrics
    - Code example coverage
    - Best section selection (multi-run mode)
    
    🎯 Automatic Activation:
    Automatically enabled when --runs > 1
    
    📈 Output Files:
    - {{topic}}_analysis_report.md
    - {{topic}}_best_compilation.html (with multiple runs)

--quality-eval
    Run GPT-based quality evaluation.
    
    📝 Description:
    Uses AI to evaluate:
    - Technical accuracy
    - Writing clarity
    - Completeness
    - Example quality
    
    📈 Output:
    - {{topic}}_gpt_evaluation_report.md

--compare-url URL
    Compare generated docs with existing documentation.
    
    ✨ Example:
    --compare-url https://docs.example.com/topic

--compare-file PATH
    Compare with local documentation file.
    
    ✨ Example:
    --compare-file ./existing-docs/topic.html

{'='*80}
🔌 PLUGIN MANAGEMENT
{'='*80}

--list-plugins
    Display all available plugins and their status.
    
    📝 Shows:
    - Plugin names and descriptions
    - Supported types
    - Priority levels
    - Enabled/disabled status

--disable-plugins PLUGIN1 PLUGIN2
    Disable specific plugins by name.
    
    ✨ Example:
    --disable-plugins module_recommender link_validator

--enable-only PLUGIN1 PLUGIN2  
    Enable only specified plugins, disable all others.
    
    ✨ Example:
    --enable-only document_compiler

{'='*80}
🛠️ CONFIGURATION
{'='*80}

--prompt-yaml-path PATH
    Use custom prompt configuration file.
    
    📝 Available Prompts:
    ./prompts/generator/default.yaml          # Standard HTML generation
    ./prompts/generator/markdown.yaml         # Markdown format
    ./prompts/generator/readme.yaml           # README generation
    ./prompts/generator/perfect_readme.yaml   # High-quality README
    ./prompts/generator/perfect_readme_with_shot.yaml  # With example
    
    ✨ Example:
    --prompt-yaml-path ./prompts/generator/perfect_readme_with_shot.yaml

--shots PATH
    Directory containing few-shot examples.
    
    📝 Description:
    Few-shot examples guide the AI to match desired format and style.
    
    Default: ./shots

--terminology-path PATH  
    Path to terminology YAML file.
    
    📝 Description:
    Domain-specific terms and definitions for consistent language.
    
    Default: ./terminology.yaml

{'='*80}
🧹 UTILITY COMMANDS
{'='*80}

--list-models
    Show all available models and providers.
    
    📝 Shows:
    - Available providers (OpenAI, Claude)
    - Supported models for each provider
    - Configuration status (API keys)

--cleanup
    Remove all files and directories in ./output/.
    
    📝 Description:
    Interactive cleanup with:
    - Preview of files to be deleted
    - Confirmation prompt (must type 'yes')
    - Detailed success/failure reporting
    
    ⚠️ Warning: This action cannot be undone!

--info
    Display this detailed help information.

--verbose, -v
    Enable detailed logging output.
    
    📝 Shows:
    - API calls and responses
    - Processing steps
    - Error details and stack traces

--quiet, -q
    Suppress non-essential output.
    
    📝 Hides:
    - Progress messages
    - Informational output
    - Keeps only errors and results

--version
    Display version information.

{'='*80}
💡 COMMON WORKFLOWS
{'='*80}

1️⃣ High-Quality Documentation Generation:
   doc-gen --topic "Machine Learning" --runs 5 --analyze --model gpt-4o

2️⃣ Perfect README for Directory:
   doc-gen --readme ./my-project \\
           --prompt-yaml-path ./prompts/generator/perfect_readme_with_shot.yaml \\
           --runs 3 --analyze

3️⃣ Recursive README Generation:
   doc-gen --readme ./project --recursive --output-dir ./docs

4️⃣ Quick Draft Documentation:
   doc-gen --topic "API Reference" --model gpt-4o-mini

5️⃣ Compare with Existing Docs:
   doc-gen --topic "Python Guide" --compare-url https://docs.python.org

6️⃣ Clean Output Directory:
   doc-gen --cleanup

{'='*80}
📚 For more information, visit:
   GitHub: https://github.com/fasrc/doc-generator
   Docs: https://docs.rc.fas.harvard.edu/
{'='*80}
"""
    print(info_text)


def cleanup_output_directory(logger: logging.Logger) -> None:
    """Remove all files and directories in ./output/."""
    output_dir = Path('./output')
    
    if not output_dir.exists():
        logger.info("Output directory does not exist, nothing to clean")
        print("Output directory './output/' does not exist.")
        return
    
    # Get list of items to be deleted for user confirmation
    items = list(output_dir.iterdir())
    if not items:
        logger.info("Output directory is already empty")
        print("Output directory './output/' is already empty.")
        return
    
    # Show what will be deleted
    print("\n⚠️  WARNING: This will delete the following items:")
    print("-" * 50)
    
    file_count = 0
    dir_count = 0
    
    for item in sorted(items):
        if item.is_file():
            print(f"  📄 {item.name}")
            file_count += 1
        elif item.is_dir():
            print(f"  📁 {item.name}/")
            dir_count += 1
    
    print("-" * 50)
    print(f"Total: {file_count} files, {dir_count} directories")
    
    # Ask for confirmation
    print("\nAre you sure you want to delete all these items? This action cannot be undone.")
    response = input("Type 'yes' to confirm, or press Enter to cancel: ").strip().lower()
    
    if response != 'yes':
        logger.info("Cleanup cancelled by user")
        print("Cleanup cancelled.")
        return
    
    # Perform cleanup
    logger.info("Starting cleanup of output directory")
    print("\nCleaning up...")
    
    success_count = 0
    error_count = 0
    
    for item in items:
        try:
            if item.is_file():
                item.unlink()
                logger.debug(f"Deleted file: {item}")
            elif item.is_dir():
                shutil.rmtree(item)
                logger.debug(f"Deleted directory: {item}")
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to delete {item}: {e}")
            print(f"  ❌ Failed to delete {item.name}: {e}")
            error_count += 1
    
    # Report results
    if error_count == 0:
        logger.info(f"Successfully cleaned up {success_count} items")
        print(f"\n✅ Successfully deleted {success_count} items from ./output/")
    else:
        logger.warning(f"Cleanup completed with errors: {success_count} succeeded, {error_count} failed")
        print(f"\n⚠️  Cleanup completed with errors:")
        print(f"  ✅ {success_count} items deleted successfully")
        print(f"  ❌ {error_count} items failed to delete")


def scan_code_examples(args, logger: logging.Logger) -> None:
    """Scan directory for code examples and optionally generate README files."""
    logger.info(f"Scanning {args.scan_code} for code examples...")
    
    # Check if README generation is requested (legacy mode)
    if args.generate_readme:
        logger.warning("--generate-readme is deprecated. Use --readme /path/to/directory for the unified pipeline.")
        
        # Use legacy README generator for backward compatibility
        from .readme_generator import ReadmeGenerator
        
        logger.info("Generating README files using legacy pipeline...")
        
        # Initialize AI provider if enhancement is requested
        ai_provider = None
        if args.ai_enhance:
            from .providers import ProviderManager
            provider_manager = ProviderManager(logger=logger)
            ai_provider = provider_manager.get_provider(args.provider)
            if not ai_provider or not ai_provider.is_available():
                logger.warning("AI enhancement requested but no provider available")
                args.ai_enhance = False
        
        readme_gen = ReadmeGenerator(
            source_dir=Path(args.scan_code),
            recursive=args.recursive,
            overwrite=args.overwrite,
            suffix=args.suffix or "_generated",
            ai_provider=ai_provider if args.ai_enhance else None,
            logger=logger
        )
        
        try:
            generated_files = readme_gen.process_directory_tree()
            logger.info(f"Generated {len(generated_files)} README files")
            
            if not args.quiet:
                print("\nGenerated README Files (Legacy Mode):")
                for file_path in generated_files:
                    print(f"  - {file_path}")
                    
        except Exception as e:
            logger.error(f"Error generating README files: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    else:
        # Original code scanning functionality
        scanner = CodeExampleScanner()
        
        try:
            # Scan directory
            code_examples = scanner.scan_directory(
                args.scan_code, 
                max_files=args.max_scan_files
            )
            
            if not code_examples:
                logger.warning("No code examples found.")
                return
            
            logger.info(f"Found {len(code_examples)} code examples")
            
            # Update terminology file
            scanner.update_terminology_file(args.terminology_path, code_examples)
            logger.info(f"Updated terminology file: {args.terminology_path}")
            
            # Show summary
            languages = {}
            for example in code_examples:
                lang = example['language']
                languages[lang] = languages.get(lang, 0) + 1
            
            print("\nCode Examples Summary:")
            for lang, count in sorted(languages.items()):
                print(f"  {lang}: {count} files")
                
        except Exception as e:
            logger.error(f"Error scanning code examples: {e}")
            sys.exit(1)


def run_generation(args, logger: logging.Logger) -> None:
    """Run the main documentation generation process."""
    # Load environment variables
    load_dotenv()
    
    # Handle list-models command first (no generator needed)
    if args.list_models:
        from .providers import ProviderManager
        manager = ProviderManager(logger=logger)
        
        print("Available LLM Providers and Models:")
        print("=" * 50)
        
        available_models = manager.get_available_models()
        if not available_models:
            print("\nNo providers available. Check your API keys:")
            print("  - OPENAI_API_KEY for OpenAI models")
            print("  - ANTHROPIC_API_KEY for Claude models")
            return
        
        for provider_name, models in available_models.items():
            provider = manager.get_provider(provider_name)
            is_configured = provider.is_available() if provider else False
            status = "✅ CONFIGURED" if is_configured else "❌ NOT CONFIGURED (API key missing)"
            
            print(f"\n{provider_name.upper()}: {status}")
            for model in models:
                print(f"  - {model}")
        
        default_provider = manager.get_default_provider()
        if default_provider:
            print(f"\nDefault provider: {default_provider}")
            print(f"Default model: {manager.get_default_model()}")
        else:
            print(f"\nNo default provider available - configure API keys to use models")
        return
    
    try:
        # Initialize generator
        logger.info("Initializing documentation generator...")
        
        # Use --shots path if provided, otherwise use --examples-dir
        shots_dir = args.shots if args.shots else args.examples_dir
        
        generator = DocumentationGenerator(
            prompt_yaml_path=args.prompt_yaml_path,
            shots_dir=shots_dir,
            terminology_path=args.terminology_path,
            provider=args.provider,
            logger=logger
        )
        
        # List plugins if requested
        if args.list_plugins:
            list_plugins(generator)
            return
        
        # Filter plugins if requested
        if args.disable_plugins:
            for plugin_name in args.disable_plugins:
                if plugin_name in generator.plugin_manager.engines:
                    del generator.plugin_manager.engines[plugin_name]
                    logger.info(f"Disabled plugin: {plugin_name}")
        
        if args.enable_only:
            enabled_plugins = set(args.enable_only)
            disabled = []
            for plugin_name in list(generator.plugin_manager.engines.keys()):
                if plugin_name not in enabled_plugins:
                    del generator.plugin_manager.engines[plugin_name]
                    disabled.append(plugin_name)
            if disabled:
                logger.info(f"Disabled plugins: {', '.join(disabled)}")
        
        # Validate topic is provided for generation
        if not args.topic and not args.list_plugins:
            logger.error("--topic is required for documentation generation")
            sys.exit(1)
        
        # Skip generation if just listing plugins
        if args.list_plugins:
            return
            
        # Generate documentation
        logger.info(f"Generating documentation for topic: '{args.topic}'")
        logger.info(f"Model: {args.model}, Temperature: {args.temperature}, Runs: {args.runs}")
        
        # Determine output directory using helper function
        output_dir = get_output_directory(args.output_dir, logger)
        
        # Generate proper filename from topic
        topic_filename = args.topic.lower().replace(' ', '_')
        
        results = generator.generate_documentation(
            query=args.topic,
            runs=args.runs,
            model=args.model,
            temperature=args.temperature,
            topic_filename=topic_filename,
            output_dir=output_dir,
            provider=args.provider if args.provider != 'auto' else None
        )
        
        if not results:
            logger.error("No documentation generated")
            sys.exit(1)
        
        logger.info(f"Generated {len(results)} documentation variants")
        
        # Print output file paths
        if not args.quiet:
            print("\nGenerated Files:")
            for i, result in enumerate(results, 1):
                filename = Path(result).name  # result is a file path
                print(f"  {i}. {filename}")
        
        # Run comparison if requested
        if args.compare_url or args.compare_file:
            run_comparison(args, results, logger)
        
        # Run analysis pipeline for multiple runs
        if len(results) >= 2 or args.analyze:
            logger.info("Running analysis pipeline...")
            
            # Load analysis plugins with configuration
            analysis_config = {
                'reporter': {'formats': [args.report_format]},
                'link_validator': {'report_format': args.report_format}
            }
            generator.plugin_manager.load_analysis_plugins(config=analysis_config)
            
            # Prepare documents for analysis
            documents = []
            for result_path in results:
                try:
                    with open(result_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents.append({
                        'path': result_path,
                        'content': content
                    })
                except FileNotFoundError:
                    logger.error(f"Generated file not found: {result_path}")
            
            if documents:
                # Run analysis pipeline
                analysis_results = generator.plugin_manager.run_analysis_pipeline(
                    documents=documents,
                    topic=args.topic,
                    output_dir=Path(output_dir)
                )
                
                # Report results
                for plugin_name, plugin_results in analysis_results.items():
                    if 'error' in plugin_results:
                        logger.error(f"  ✗ {plugin_name}: {plugin_results['error']}")
                    else:
                        artifacts = plugin_results.get('artifacts', [])
                        if artifacts:
                            logger.info(f"  ✓ {plugin_name}: Generated {len(artifacts)} files")
                            if not args.quiet:
                                for artifact in artifacts:
                                    print(f"     - {Path(artifact).name}")
        
        # Legacy analysis if requested and no plugins available
        elif args.analyze:
            logger.info("Running basic document analysis...")
            analyzer = DocumentAnalyzer()
            
            for i, result in enumerate(results):
                # result is a file path, read content from file
                try:
                    with open(result, 'r', encoding='utf-8') as f:
                        content = f.read()
                    sections = analyzer.extract_sections(content)
                    logger.info(f"Variant {i+1}: Found {len(sections)} sections")
                except FileNotFoundError:
                    logger.error(f"Generated file not found: {result}")
        
        # Run quality evaluation if requested
        if args.quality_eval:
            logger.info("Running GPT-based quality evaluation...")
            try:
                evaluator = GPTQualityEvaluator(
                    generator.client,
                    analysis_prompt_path=args.analysis_prompt_path
                )
                
                for i, result in enumerate(results):
                    # result is a file path, read content from file
                    try:
                        with open(result, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except FileNotFoundError:
                        logger.error(f"Generated file not found: {result}")
                        continue
                    
                    if content:
                        # This is a simplified evaluation - full implementation would
                        # evaluate each section separately
                        logger.info(f"Evaluating variant {i+1}...")
                        
            except Exception as e:
                logger.warning(f"Quality evaluation failed: {e}")
        
        logger.info("Documentation generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_token_analysis(args, logger):
    """Run token usage analysis for a specific operation."""
    try:
        # Load content if provided
        content = None
        if args.content:
            content_path = Path(args.content)
            if content_path.exists():
                with open(content_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"Loaded content from {content_path}")
            else:
                # Treat as direct text input
                content = args.content
                logger.info("Using provided text content")
        
        # Initialize token machine
        config_path = Path('.claude/agents/token_machine.yaml')
        if config_path.exists():
            logger.info(f"Loading token machine config from {config_path}")
        else:
            logger.info("Using default token machine configuration")
            config_path = None
        
        depth = AnalysisDepth(args.analysis_depth)
        machine = TokenMachine(config_path=config_path, analysis_depth=depth)
        
        # Perform analysis
        analysis = machine.analyze(
            operation=args.token_analyze,
            content=content,
            content_type=args.content_type,
            context_size=args.context_size,
            expected_output_size=args.expected_output
        )
        
        # Display results
        print(f"\n=== Token Usage Analysis: {analysis.operation} ===")
        print(f"Timestamp: {analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Analysis Depth: {analysis.metadata.get('analysis_depth', 'standard').upper()}")
        
        print(f"\n--- Token Breakdown ---")
        print(f"Input Tokens:  {analysis.token_estimate.input_tokens:,}")
        print(f"Output Tokens: {analysis.token_estimate.output_tokens:,}")
        print(f"Total Tokens:  {analysis.token_estimate.total_tokens:,}")
        print(f"Confidence:    {analysis.token_estimate.confidence:.1%}")
        
        if analysis.token_estimate.breakdown:
            print(f"\n--- Detailed Breakdown ---")
            for component, tokens in analysis.token_estimate.breakdown.items():
                print(f"{component.replace('_', ' ').title()}: {tokens:,} tokens")
        
        print(f"\n--- Cost Analysis ---")
        if analysis.cost_estimates:
            print(f"{'Model':<20} {'Provider':<12} {'Total Cost':<12} {'Quality':<8} {'Speed':<8}")
            print("-" * 68)
            for estimate in analysis.cost_estimates[:5]:  # Show top 5
                print(f"{estimate.model_name:<20} {estimate.provider.value:<12} "
                      f"${estimate.total_cost:<11.4f} {estimate.quality_score:<7.1%} "
                      f"{estimate.speed_score:<7.1%}")
        
        if analysis.recommended_model:
            print(f"\nRecommended Model: {analysis.recommended_model}")
        
        print(f"\n--- Optimization Strategies ---")
        for i, strategy in enumerate(analysis.optimization_strategies[:3], 1):
            print(f"{i}. {strategy.name}")
            print(f"   {strategy.description}")
            print(f"   Potential Savings: {strategy.potential_savings:.0%}")
            print(f"   Implementation: {strategy.implementation_effort}")
            if strategy.recommended:
                print(f"   ✓ RECOMMENDED")
            print()
        
        if analysis.warnings:
            print(f"--- Warnings ---")
            for warning in analysis.warnings:
                print(f"⚠️  {warning}")
        
        logger.info("Token analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Token analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_token_estimate(args, logger):
    """Run quick token estimation."""
    try:
        machine = TokenMachine()
        
        # Load content if provided
        content = None
        if args.content:
            content_path = Path(args.content)
            if content_path.exists():
                with open(content_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                content = args.content
        
        # Get quick analysis
        result = analyze_operation(
            args.token_estimate,
            content=content,
            content_type=args.content_type,
            expected_output_size=args.expected_output,
            context_size=args.context_size
        )
        
        print(f"\n=== Quick Token Estimate: {args.token_estimate} ===")
        print(f"Total Tokens: {result['tokens']['total']:,}")
        print(f"Cost Range: ${result['cost_range']['min']:.4f} - ${result['cost_range']['max']:.4f}")
        
        if result['recommended_model']:
            print(f"Recommended: {result['recommended_model']}")
        
        if result['top_optimization']:
            print(f"Top Optimization: {result['top_optimization']}")
        
        if result['warnings']:
            print("Warnings:")
            for warning in result['warnings']:
                print(f"  ⚠️  {warning}")
        
        logger.info("Token estimation completed")
        
    except Exception as e:
        logger.error(f"Token estimation failed: {e}")
        sys.exit(1)


def run_token_report(args, logger):
    """Generate token usage report."""
    try:
        machine = TokenMachine()
        report = machine.generate_report(
            period_days=args.period,
            include_predictions=True
        )
        
        if "error" in report:
            print(f"❌ {report['error']}")
            return
        
        print(f"\n=== Token Usage Report ===")
        print(f"Period: {report['period']['days']} days")
        print(f"From: {report['period']['start'][:10]} to {report['period']['end'][:10]}")
        
        summary = report['summary']
        print(f"\n--- Summary ---")
        print(f"Total Operations: {summary['total_operations']:,}")
        print(f"Total Tokens: {summary['total_tokens']:,}")
        print(f"Estimated Cost: ${summary['estimated_cost']:.2f}")
        print(f"Avg Tokens/Operation: {summary['avg_tokens_per_operation']:,}")
        
        if report['model_distribution']:
            print(f"\n--- Model Usage ---")
            for model, count in sorted(report['model_distribution'].items(), 
                                     key=lambda x: x[1], reverse=True):
                print(f"{model}: {count} operations")
        
        if report['top_operations']:
            print(f"\n--- Top Operations by Token Usage ---")
            for op_info in report['top_operations']:
                print(f"{op_info['operation']}: {op_info['total_tokens']:,} tokens "
                      f"({op_info['count']} runs, avg: {op_info['avg_tokens']:,})")
        
        if report['optimization_opportunities']:
            print(f"\n--- Optimization Opportunities ---")
            for opp in report['optimization_opportunities']:
                print(f"• {opp['description']}")
                print(f"  Potential Savings: {opp['potential_savings']}")
                print(f"  Priority: {opp['priority'].upper()}")
        
        if 'predictions' in report:
            pred = report['predictions']
            print(f"\n--- Predictions ---")
            print(f"Next Week: {pred['next_week']['estimated_tokens']:,} tokens "
                  f"(${pred['next_week']['estimated_cost']:.2f})")
            print(f"Next Month: {pred['next_month']['estimated_tokens']:,} tokens "
                  f"(${pred['next_month']['estimated_cost']:.2f})")
            print(f"Growth Trend: {pred['growth_trend']}")
        
        logger.info("Token report generated successfully")
        
    except Exception as e:
        logger.error(f"Token report generation failed: {e}")
        sys.exit(1)


def run_token_optimize(args, logger):
    """Run token optimization analysis."""
    try:
        machine = TokenMachine()
        
        # Load content if provided
        content = None
        if args.content:
            content_path = Path(args.content)
            if content_path.exists():
                with open(content_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                content = args.content
        
        # Get comprehensive analysis focused on optimization
        analysis = machine.analyze(
            operation=args.token_optimize,
            content=content,
            content_type=args.content_type,
            context_size=args.context_size,
            expected_output_size=args.expected_output,
            depth=AnalysisDepth.COMPREHENSIVE
        )
        
        print(f"\n=== Token Optimization: {analysis.operation} ===")
        print(f"Current Usage: {analysis.token_estimate.total_tokens:,} tokens")
        
        if analysis.cost_estimates:
            current_cost = analysis.cost_estimates[0].total_cost
            print(f"Current Min Cost: ${current_cost:.4f}")
        
        print(f"\n--- Optimization Strategies ---")
        total_potential_savings = 0
        
        for i, strategy in enumerate(analysis.optimization_strategies, 1):
            savings_pct = strategy.potential_savings
            total_potential_savings += savings_pct
            
            if analysis.cost_estimates:
                savings_amount = current_cost * savings_pct
                print(f"{i}. {strategy.name} ({strategy.implementation_effort} effort)")
                print(f"   {strategy.description}")
                print(f"   💰 Potential Savings: {savings_pct:.0%} (${savings_amount:.4f})")
            else:
                print(f"{i}. {strategy.name} ({strategy.implementation_effort} effort)")
                print(f"   {strategy.description}")
                print(f"   💰 Potential Savings: {savings_pct:.0%}")
            
            if strategy.recommended:
                print(f"   ⭐ RECOMMENDED")
            
            if strategy.implementation_steps:
                print(f"   Implementation Steps:")
                for step in strategy.implementation_steps[:3]:  # Show first 3 steps
                    print(f"     • {step}")
                if len(strategy.implementation_steps) > 3:
                    print(f"     • ... and {len(strategy.implementation_steps) - 3} more")
            print()
        
        # Model recommendations
        if args.max_cost or args.min_quality:
            recommended = machine.recommend_model(
                max_cost=args.max_cost,
                min_quality=args.min_quality,
                max_tokens=analysis.token_estimate.total_tokens
            )
            
            if recommended:
                print(f"--- Model Recommendation ---")
                print(f"Based on your constraints: {recommended}")
                
                # Find the cost estimate for this model
                for estimate in analysis.cost_estimates:
                    if estimate.model_name == recommended:
                        print(f"Cost: ${estimate.total_cost:.4f}")
                        print(f"Quality: {estimate.quality_score:.1%}")
                        print(f"Speed: {estimate.speed_score:.1%}")
                        break
            else:
                print(f"--- Model Recommendation ---")
                print(f"❌ No model meets your constraints")
        
        if analysis.cost_estimates and total_potential_savings > 0:
            max_savings = current_cost * min(total_potential_savings, 0.8)  # Cap at 80%
            print(f"\n--- Summary ---")
            print(f"Maximum Potential Savings: ${max_savings:.4f} ({min(total_potential_savings, 0.8):.0%})")
            print(f"Optimized Cost Range: ${current_cost - max_savings:.4f} - ${current_cost:.4f}")
        
        logger.info("Token optimization analysis completed")
        
    except Exception as e:
        logger.error(f"Token optimization failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(args.verbose)
    
    try:
        # Handle info display
        if args.info:
            display_info()
            return
        
        # Handle cleanup mode
        if args.cleanup:
            cleanup_output_directory(logger)
            return
        
        # Handle token analysis modes
        if args.token_analyze:
            run_token_analysis(args, logger)
            return
        
        if args.token_estimate:
            run_token_estimate(args, logger)
            return
        
        if args.token_report:
            run_token_report(args, logger)
            return
        
        if args.token_optimize:
            run_token_optimize(args, logger)
            return
        
        # Handle README generation mode
        if args.readme:
            run_readme_generation(args, logger)
            return
        
        # Handle standardization mode
        if args.standardize:
            run_standardization(args, logger)
            return
        
        # Handle code scanning mode
        if args.scan_code:
            scan_code_examples(args, logger)
            return
        
        # Run main generation
        run_generation(args, logger)
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()