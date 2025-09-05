"""
Generate Command

Implements documentation generation functionality (replaces --topic mode).
"""

import sys
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from ..base import BaseCommand, CommandResult
from ...core import DocumentationGenerator, DocumentAnalyzer, GPTQualityEvaluator
from ...utils import get_output_directory
from ...exceptions import DocGeneratorError


class GenerateCommand(BaseCommand):
    """Command for generating technical documentation using AI."""
    
    @property
    def name(self) -> str:
        return "generate"
    
    @property
    def description(self) -> str:
        return "Generate technical documentation for a specific topic using AI"
    
    @property
    def aliases(self):
        return ["gen", "g"]
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add generate command arguments."""
        # Required positional argument
        parser.add_argument('topic', 
                          help='Topic for documentation generation (e.g., "Python Programming")')
        
        # Generation options
        parser.add_argument('--runs', type=int, default=1,
                          help='Number of variants to generate (default: 1)')
        parser.add_argument('--model', 
                          help='Model to use (e.g., gpt-4o-mini, claude-3-5-sonnet-20240620)')
        parser.add_argument('--provider', choices=['openai', 'claude', 'auto'], 
                          default='auto',
                          help='LLM provider to use (default: auto-detect)')
        parser.add_argument('--temperature', type=float, default=0.3,
                          help='Temperature for text generation (default: 0.3)')
        parser.add_argument('--output-dir', default='output',
                          help='Output directory (default: ./output)')
        
        # Format and prompts
        parser.add_argument('--format', choices=['html', 'markdown', 'auto'],
                          default='auto',
                          help='Output format for documentation (default: auto-detect)')
        parser.add_argument('--prompt-yaml-path', '--prompt-yaml', 
                          default='./prompts/generator/default.yaml',
                          help='Path to prompt YAML configuration')
        
        # Analysis and evaluation
        parser.add_argument('--analyze', action='store_true',
                          help='Run document analysis after generation')
        parser.add_argument('--quality-eval', action='store_true',
                          help='Run GPT-based quality evaluation')
        parser.add_argument('--compare-url', metavar='URL',
                          help='Compare with existing documentation at URL')
        parser.add_argument('--compare-file', metavar='FILE',
                          help='Compare with local file')
        
        # Configuration paths
        parser.add_argument('--shots', metavar='PATH',
                          help='Path to few-shot examples directory')
        parser.add_argument('--examples-dir', default='shots',
                          help='Directory containing few-shot examples (default: shots)')
        parser.add_argument('--terminology-path', default='terminology.yaml',
                          help='Path to terminology YAML file (default: terminology.yaml)')
        parser.add_argument('--analysis-prompt-path', 
                          default='./prompts/analysis/default.yaml',
                          help='Path to analysis prompt configuration')
        parser.add_argument('--report-format', choices=['markdown', 'html', 'json'],
                          default='markdown',
                          help='Format for analysis reports (default: markdown)')
        
        # Plugin management
        parser.add_argument('--disable-plugins', nargs='*', metavar='PLUGIN',
                          help='Disable specific plugins by name')
        parser.add_argument('--enable-only', nargs='*', metavar='PLUGIN',
                          help='Enable only specified plugins (disable all others)')
        
        # Utility flags
        parser.add_argument('--list-models', action='store_true',
                          help='List available models and providers')
        parser.add_argument('--list-plugins', action='store_true',
                          help='List all available plugins')
    
    def validate_args(self, args: Namespace) -> None:
        """Validate command arguments."""
        # Check if this is a utility command
        if args.list_models or args.list_plugins:
            return  # No further validation needed for utility commands
            
        # Validate topic is provided for actual generation
        if not args.topic:
            raise DocGeneratorError("Topic is required for documentation generation")
        
        # Validate temperature range
        if not (0.0 <= args.temperature <= 2.0):
            raise DocGeneratorError("Temperature must be between 0.0 and 2.0")
        
        # Validate runs count
        if args.runs < 1:
            raise DocGeneratorError("Number of runs must be at least 1")
        
        # Validate file paths if provided
        if args.compare_file:
            compare_file = self.validate_path(args.compare_file, must_exist=True, must_be_file=True)
            
        # Check prompt YAML exists
        if args.prompt_yaml_path:
            self.validate_path(args.prompt_yaml_path, must_exist=True, must_be_file=True)
        
        # Check analysis prompt exists if quality eval requested
        if args.quality_eval and args.analysis_prompt_path:
            self.validate_path(args.analysis_prompt_path, must_exist=True, must_be_file=True)
    
    def run(self, args: Namespace) -> CommandResult:
        """Execute documentation generation."""
        try:
            # Load environment variables
            load_dotenv()
            
            # Handle list-models command first (no generator needed)
            if args.list_models:
                return self._list_models()
            
            # Initialize generator
            self.logger.info("Initializing documentation generator...")
            
            # Use --shots path if provided, otherwise use --examples-dir
            shots_dir = args.shots if args.shots else args.examples_dir
            
            generator = DocumentationGenerator(
                prompt_yaml_path=args.prompt_yaml_path,
                shots_dir=shots_dir,
                terminology_path=args.terminology_path,
                provider=args.provider,
                logger=self.logger
            )
            
            # List plugins if requested
            if args.list_plugins:
                return self._list_plugins(generator)
            
            # Configure plugins
            self._configure_plugins(generator, args)
            
            # Generate documentation
            self.logger.info(f"Generating documentation for topic: '{args.topic}'")
            self.logger.info(f"Model: {args.model}, Temperature: {args.temperature}, Runs: {args.runs}")
            
            # Determine output directory using helper function
            output_dir = get_output_directory(args.output_dir, self.logger)
            
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
                return self.handle_error(
                    DocGeneratorError("No documentation generated"),
                    "Generation failed"
                )
            
            self.logger.info(f"Generated {len(results)} documentation variants")
            
            # Print output file paths with clickable URLs
            if not getattr(args, 'quiet', False):
                print("\n📝 Documentation Generation Complete")
                from ...cli import format_output_summary
                format_output_summary(Path(output_dir), results)
            
            # Run comparison if requested
            if args.compare_url or args.compare_file:
                self._run_comparison(args, results)
            
            # Run analysis pipeline
            self._run_analysis_pipeline(generator, args, results, output_dir)
            
            # Run quality evaluation if requested
            if args.quality_eval:
                self._run_quality_evaluation(generator, args, results)
            
            message = f"Successfully generated {len(results)} documentation variants for '{args.topic}'"
            return self.success(message, {
                'topic': args.topic,
                'variants_generated': len(results),
                'output_files': results
            })
            
        except Exception as e:
            return self.handle_error(e, "Documentation generation failed")
    
    def _list_models(self) -> CommandResult:
        """List available models and providers."""
        try:
            from ...providers import ProviderManager
            manager = ProviderManager(logger=self.logger)

            print("Available LLM Providers and Models:")
            print("=" * 50)

            available_models = manager.get_available_models()
            if not available_models:
                print("\nNo providers available. Check your API keys:")
                print("  - OPENAI_API_KEY for OpenAI models")
                print("  - ANTHROPIC_API_KEY for Claude models")
                return self.success("No providers configured")

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

            return self.success("Listed available models and providers")

        except Exception as e:
            return self.handle_error(e, "Failed to list models")

    def _list_plugins(self, generator: DocumentationGenerator) -> CommandResult:
        """List available plugins."""
        try:
            from ...cli import list_plugins  # Import the existing function
            list_plugins(generator)
            return self.success("Listed available plugins")
        except Exception as e:
            return self.handle_error(e, "Failed to list plugins")

    def _configure_plugins(self, generator: DocumentationGenerator, args: Namespace) -> None:
        """Configure plugin filtering based on arguments."""
        # Filter plugins if requested
        if args.disable_plugins:
            for plugin_name in args.disable_plugins:
                if plugin_name in generator.plugin_manager.engines:
                    del generator.plugin_manager.engines[plugin_name]
                    self.logger.info(f"Disabled plugin: {plugin_name}")

        if args.enable_only:
            enabled_plugins = set(args.enable_only)
            disabled = []
            for plugin_name in list(generator.plugin_manager.engines.keys()):
                if plugin_name not in enabled_plugins:
                    del generator.plugin_manager.engines[plugin_name]
                    disabled.append(plugin_name)
            if disabled:
                self.logger.info(f"Disabled plugins: {', '.join(disabled)}")

    def _run_comparison(self, args: Namespace, results: list) -> None:
        """Run document comparison if requested."""
        try:
            from ...cli import run_comparison  # Import existing function
            run_comparison(args, results, self.logger)
        except Exception as e:
            self.logger.warning(f"Comparison failed: {e}")

    def _run_analysis_pipeline(self, generator: DocumentationGenerator, args: Namespace, 
                             results: list, output_dir: str) -> None:
        """Run analysis pipeline for multiple runs."""
        if len(results) >= 2 or args.analyze:
            self.logger.info("Running analysis pipeline...")
            
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
                    self.logger.error(f"Generated file not found: {result_path}")
            
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
                        self.logger.error(f"  ✗ {plugin_name}: {plugin_results['error']}")
                    else:
                        artifacts = plugin_results.get('artifacts', [])
                        if artifacts:
                            self.logger.info(f"  ✓ {plugin_name}: Generated {len(artifacts)} files")
                            if not getattr(args, 'quiet', False):
                                for artifact in artifacts:
                                    print(f"     - {Path(artifact).name}")
            
        # Legacy analysis if requested and no plugins available
        elif args.analyze:
            self.logger.info("Running basic document analysis...")
            analyzer = DocumentAnalyzer()
            
            for i, result in enumerate(results):
                try:
                    with open(result, 'r', encoding='utf-8') as f:
                        content = f.read()
                    sections = analyzer.extract_sections(content)
                    self.logger.info(f"Variant {i+1}: Found {len(sections)} sections")
                except FileNotFoundError:
                    self.logger.error(f"Generated file not found: {result}")

    def _run_quality_evaluation(self, generator: DocumentationGenerator, 
                              args: Namespace, results: list) -> None:
        """Run GPT-based quality evaluation."""
        self.logger.info("Running GPT-based quality evaluation...")
        try:
            evaluator = GPTQualityEvaluator(
                generator.client,
                analysis_prompt_path=args.analysis_prompt_path
            )
            
            for i, result in enumerate(results):
                try:
                    with open(result, 'r', encoding='utf-8') as f:
                        content = f.read()
                except FileNotFoundError:
                    self.logger.error(f"Generated file not found: {result}")
                    continue
                
                if content:
                    self.logger.info(f"Evaluating variant {i+1}...")
                    # Note: This is a simplified evaluation - full implementation would
                    # evaluate each section separately
                    
        except Exception as e:
            self.logger.warning(f"Quality evaluation failed: {e}")