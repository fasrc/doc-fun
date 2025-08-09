#!/usr/bin/env python3
"""
Command-line interface for doc_generator package.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

from .core import DocumentationGenerator, DocumentAnalyzer, GPTQualityEvaluator, CodeExampleScanner
from . import __version__


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('doc_generator')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate technical documentation using AI with plugin-based recommendations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  doc-gen --topic "Python Programming" --output-dir ./output
  
  # Multiple runs with custom model
  doc-gen --topic "Machine Learning" --runs 3 --model gpt-4 --temperature 0.7
  
  # Use custom prompt template
  doc-gen --topic "CUDA Programming" --prompt-yaml ./prompts/cuda.yaml
  
  # Full analysis pipeline
  doc-gen --topic "Parallel Computing" --runs 5 --analyze --quality-eval
        """
    )
    
    # Required arguments
    parser.add_argument('--topic', 
                       help='Topic to generate documentation for')
    
    # File path arguments
    parser.add_argument('--output-dir', default='output',
                       help='Output directory for generated files (default: output)')
    parser.add_argument('--prompt-yaml-path', '--prompt-yaml', 
                       default='./prompts/generator/default.yaml',
                       help='Path to prompt YAML configuration (default: ./prompts/generator/default.yaml)')
    parser.add_argument('--examples-dir', default='examples',
                       help='Directory containing few-shot examples (default: examples)')
    parser.add_argument('--terminology-path', default='terminology.yaml',
                       help='Path to terminology YAML file (default: terminology.yaml)')
    
    # Generation parameters
    parser.add_argument('--runs', type=int, default=1,
                       help='Number of documentation variants to generate (default: 1)')
    parser.add_argument('--model', 
                       help='Model to use (e.g., gpt-4o-mini, claude-3-5-sonnet-20240620). Auto-detects provider.')
    parser.add_argument('--provider', choices=['openai', 'claude', 'auto'], 
                       default='auto',
                       help='LLM provider to use (default: auto-detect)')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models and providers')
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='Temperature for text generation (default: 0.3)')
    
    # Analysis options
    parser.add_argument('--analyze', action='store_true',
                       help='Run document analysis after generation')
    parser.add_argument('--quality-eval', action='store_true',
                       help='Run GPT-based quality evaluation')
    parser.add_argument('--analysis-prompt-path', 
                       default='./prompts/analysis/default.yaml',
                       help='Path to analysis prompt configuration')
    
    # Plugin options
    parser.add_argument('--list-plugins', action='store_true',
                       help='List all available plugins and exit')
    parser.add_argument('--disable-plugins', nargs='*', metavar='PLUGIN',
                       help='Disable specific plugins by name')
    parser.add_argument('--enable-only', nargs='*', metavar='PLUGIN',
                       help='Enable only specified plugins (disable all others)')
    
    # Code scanning options
    parser.add_argument('--scan-code', nargs='?', const='.', metavar='DIR',
                       help='Scan directory for code examples and update terminology (default: current dir)')
    parser.add_argument('--max-scan-files', type=int, default=50,
                       help='Maximum files to scan for code examples (default: 50)')
    
    # Comparison options
    parser.add_argument('--compare-url', metavar='URL',
                       help='Compare generated documentation with existing documentation at URL')
    parser.add_argument('--compare-file', metavar='FILE',
                       help='Compare generated documentation with local file')
    parser.add_argument('--comparison-report', metavar='FILE',
                       help='Save comparison report to file')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress non-essential output')
    parser.add_argument('--version', action='version', version=f'doc-generator {__version__}')
    
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


def scan_code_examples(args, logger: logging.Logger) -> None:
    """Scan directory for code examples and update terminology."""
    logger.info(f"Scanning {args.scan_code} for code examples...")
    
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
        
        for provider, models in available_models.items():
            print(f"\n{provider.upper()}:")
            for model in models:
                print(f"  - {model}")
        
        print(f"\nDefault provider: {manager.get_default_provider()}")
        print(f"Default model: {manager.get_default_model()}")
        return
    
    try:
        # Initialize generator
        logger.info("Initializing documentation generator...")
        generator = DocumentationGenerator(
            prompt_yaml_path=args.prompt_yaml_path,
            examples_dir=args.examples_dir,
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
        
        # Generate proper filename from topic
        topic_filename = args.topic.lower().replace(' ', '_')
        
        results = generator.generate_documentation(
            query=args.topic,
            runs=args.runs,
            model=args.model,
            temperature=args.temperature,
            topic_filename=topic_filename,
            output_dir=args.output_dir,
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
        
        # Run analysis if requested
        if args.analyze:
            logger.info("Running document analysis...")
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