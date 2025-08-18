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
from .exceptions import DocGeneratorError
from . import __version__


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


def get_output_directory(output_dir: str, logger: logging.Logger) -> str:
    """
    Get the output directory path, creating a timestamped subdirectory if using default.
    
    Args:
        output_dir: The output directory from command line args
        logger: Logger instance
        
    Returns:
        The final output directory path to use
    """
    import time
    
    settings = get_settings()
    
    if output_dir == 'output':  # Default value, create timestamped directory
        timestamp = int(time.time())
        base_output = settings.paths.output_dir
        final_output_dir = base_output / str(timestamp)
        final_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created timestamped output directory: {final_output_dir}")
        return str(final_output_dir)
    else:
        final_output_dir = output_dir
        Path(final_output_dir).mkdir(parents=True, exist_ok=True)
    
    return final_output_dir


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
    mode_group.add_argument('--scan-code', nargs='?', const='.', metavar='DIR',
                           help='Scan directory for code examples and update terminology')
    
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
    topic_group.add_argument('--comparison-report', metavar='FILE',
                            help='Save comparison report to file')
    
    # === README MODE OPTIONS ===
    readme_group = parser.add_argument_group('README Mode Options', 'Used with --readme')
    readme_group.add_argument('--recursive', action='store_true',
                             help='Generate README files for all subdirectories')
    
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
        
        # Handle README generation mode
        if args.readme:
            run_readme_generation(args, logger)
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