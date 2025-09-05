"""
Readme Command

Implements README.md generation functionality for directories.
"""

import sys
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from ..base import BaseCommand, CommandResult
from ...utils import get_output_directory
from ...exceptions import DocGeneratorError


class ReadmeCommand(BaseCommand):
    """Command for generating README.md files for directories."""
    
    @property
    def name(self) -> str:
        return "readme"
    
    @property
    def description(self) -> str:
        return "Generate README.md for directory using unified pipeline"
    
    @property
    def aliases(self):
        return ["r"]
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add README command arguments."""
        # Required positional argument
        parser.add_argument('directory', 
                          help='Directory path for README generation')
        
        # Generation options
        parser.add_argument('--runs', type=int, default=3,
                          help='Number of variants to generate (default: 3 for quality)')
        parser.add_argument('--model', 
                          help='Model to use (e.g., gpt-4o-mini, claude-3-5-sonnet-20240620)')
        parser.add_argument('--provider', choices=['openai', 'claude', 'auto'], 
                          default='auto',
                          help='LLM provider to use (default: auto-detect)')
        parser.add_argument('--temperature', type=float, default=0.3,
                          help='Temperature for text generation (default: 0.3)')
        parser.add_argument('--output-dir', default='output',
                          help='Output directory (default: ./output/timestamp)')
        
        # README-specific options
        parser.add_argument('--recursive', action='store_true',
                          help='Generate README files for all subdirectories')
        
        # Analysis and evaluation
        parser.add_argument('--analyze', action='store_true',
                          help='Run document analysis after generation (auto-enabled with --runs > 1)')
        parser.add_argument('--analysis-prompt-path', 
                          default='./prompts/analysis/default.yaml',
                          help='Path to analysis prompt configuration')
        parser.add_argument('--report-format', choices=['markdown', 'html', 'json'],
                          default='markdown',
                          help='Format for analysis reports (default: markdown)')
        
        # Configuration paths
        parser.add_argument('--prompt-yaml-path', '--prompt-yaml', 
                          default='./prompts/generator/user_codes_topic_readme.yaml',
                          help='Path to prompt YAML configuration (default: README prompt)')
        parser.add_argument('--shots', metavar='PATH',
                          help='Path to few-shot examples directory')
        parser.add_argument('--examples-dir', default='shots',
                          help='Directory containing few-shot examples (default: shots)')
        parser.add_argument('--terminology-path', default='terminology.yaml',
                          help='Path to terminology YAML file (default: terminology.yaml)')
        
        # Plugin management
        parser.add_argument('--disable-plugins', nargs='*', metavar='PLUGIN',
                          help='Disable specific plugins by name')
        parser.add_argument('--enable-only', nargs='*', metavar='PLUGIN',
                          help='Enable only specified plugins (disable all others)')
    
    def validate_args(self, args: Namespace) -> None:
        """Validate command arguments."""
        # Validate directory exists
        directory_path = Path(args.directory)
        if not directory_path.exists():
            raise DocGeneratorError(f"Directory not found: {args.directory}")
        
        if not directory_path.is_dir():
            raise DocGeneratorError(f"Path is not a directory: {args.directory}")
        
        # Validate temperature range
        if not (0.0 <= args.temperature <= 2.0):
            raise DocGeneratorError("Temperature must be between 0.0 and 2.0")
        
        # Validate runs count
        if args.runs < 1:
            raise DocGeneratorError("Number of runs must be at least 1")
        
        # Check prompt YAML exists
        if args.prompt_yaml_path:
            self.validate_path(args.prompt_yaml_path, must_exist=True, must_be_file=True)
        
        # Check analysis prompt exists if analyze requested
        if (args.analyze or args.runs > 1) and args.analysis_prompt_path:
            self.validate_path(args.analysis_prompt_path, must_exist=True, must_be_file=True)
    
    def run(self, args: Namespace) -> CommandResult:
        """Execute README generation."""
        try:
            # Load environment variables
            load_dotenv()
            
            # Initialize README generator
            self.logger.info(f"Generating README for directory: {args.directory}")
            
            from ...readme_documentation_generator import ReadmeDocumentationGenerator
            
            # Use --shots path if provided, otherwise use --examples-dir
            shots_dir = args.shots if args.shots else args.examples_dir
            
            # Use README-specific prompt if default is used
            prompt_yaml_path = args.prompt_yaml_path
            if 'readme' not in prompt_yaml_path.lower():
                prompt_yaml_path = './prompts/generator/user_codes_topic_readme.yaml'
            
            generator = ReadmeDocumentationGenerator(
                prompt_yaml_path=prompt_yaml_path,
                shots_dir=shots_dir,
                terminology_path=args.terminology_path,
                provider=args.provider,
                logger=self.logger
            )
            
            # Configure plugins
            self._configure_plugins(generator, args)
            
            # Set README-specific analysis prompt if using default
            analysis_prompt_path = args.analysis_prompt_path
            if analysis_prompt_path == './prompts/analysis/default.yaml':
                analysis_prompt_path = './prompts/analysis/readme.yaml'
            
            # Determine output directory using helper function
            output_dir = get_output_directory(args.output_dir, self.logger)
            
            # Generate README with multiple runs and analysis
            self.logger.info(f"Model: {args.model}, Temperature: {args.temperature}, Runs: {args.runs}")
            
            results = generator.generate_readme_documentation(
                directory_path=args.directory,
                runs=args.runs,
                model=args.model,
                temperature=args.temperature,
                analyze=args.analyze or args.runs > 1,  # Auto-analyze if multiple runs
                provider=args.provider if args.provider != 'auto' else None,
                output_dir=output_dir
            )
            
            # Process results
            generated_files = results['generated_files']
            if not generated_files:
                return self.handle_error(
                    DocGeneratorError("No README files generated"),
                    "Generation failed"
                )
            
            self.logger.info(f"Generated {len(generated_files)} README variations")
            
            # Display results if not quiet
            if not getattr(args, 'quiet', False):
                self._display_results(results, output_dir)
            
            # Handle recursive generation if requested
            if args.recursive:
                self._handle_recursive_generation(generator, args, results)
            
            message = f"Successfully generated {len(generated_files)} README files for '{results['directory_info']['name']}'"
            return self.success(message, {
                'directory': args.directory,
                'variants_generated': len(generated_files),
                'output_files': generated_files,
                'directory_info': results['directory_info']
            })
            
        except Exception as e:
            return self.handle_error(e, "README generation failed")
    
    def _configure_plugins(self, generator, args: Namespace) -> None:
        """Configure plugin filtering based on arguments."""
        # Filter plugins if requested
        if args.disable_plugins:
            for plugin_name in args.disable_plugins:
                if hasattr(generator, 'plugin_manager') and plugin_name in generator.plugin_manager.engines:
                    del generator.plugin_manager.engines[plugin_name]
                    self.logger.info(f"Disabled plugin: {plugin_name}")
        
        if args.enable_only:
            enabled_plugins = set(args.enable_only)
            disabled = []
            if hasattr(generator, 'plugin_manager'):
                for plugin_name in list(generator.plugin_manager.engines.keys()):
                    if plugin_name not in enabled_plugins:
                        del generator.plugin_manager.engines[plugin_name]
                        disabled.append(plugin_name)
                if disabled:
                    self.logger.info(f"Disabled plugins: {', '.join(disabled)}")
    
    def _display_results(self, results: dict, output_dir: str) -> None:
        """Display generation results to user."""
        print(f"\n📁 README Generation Results for: {results['directory_info']['name']}")
        print(f"📍 Source Path: {results['directory_info']['path']}")
        print(f"📂 Output Directory: {output_dir}")
        print(f"📊 Depth Level: {results['depth_level']}")
        print(f"💻 Languages Found: {', '.join(results['directory_info']['languages']) or 'None'}")
        print(f"📂 Subdirectories: {len(results['directory_info']['subdirectories'])}")
        print(f"📄 Files: {len(results['directory_info']['files'])}")
        
        print("\n📝 Generated Files:")
        # Prepare file types for better labeling
        file_types = {}
        for file_path in results['generated_files']:
            file_name = Path(file_path).name
            if 'best' in file_name.lower():
                file_types[file_path] = "⭐ Best compilation"
            else:
                file_types[file_path] = "📄 Generated"
        
        from ...cli import format_output_summary
        format_output_summary(Path(output_dir), results['generated_files'], file_types)
        
        # Display analysis results if available
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
    
    def _handle_recursive_generation(self, generator, args: Namespace, results: dict) -> None:
        """Handle recursive generation for subdirectories."""
        self.logger.info("Processing subdirectories recursively...")
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
                    
                    if not getattr(args, 'quiet', False):
                        print(f"    ✅ Generated {len(sub_results['generated_files'])} files")
                        
                except Exception as e:
                    self.logger.error(f"Error processing subdirectory {subdir_path}: {e}")
                    print(f"    ❌ Failed: {e}")
                    if getattr(args, 'verbose', False):
                        import traceback
                        traceback.print_exc()
        else:
            print("  No subdirectories found for recursive processing.")