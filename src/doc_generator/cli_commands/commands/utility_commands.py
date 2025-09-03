"""
Utility Commands

Implements utility commands like list-models, cleanup, and info.
"""

import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

from ..base import BaseCommand, CommandResult
from ... import __version__


class ListModelsCommand(BaseCommand):
    """Command for listing available models and providers."""
    
    @property
    def name(self) -> str:
        return "list-models"
    
    @property
    def description(self) -> str:
        return "List available models and providers"
    
    @property
    def aliases(self):
        return ["lm", "models"]
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add list-models command arguments."""
        pass  # No additional arguments needed
    
    def run(self, args: Namespace) -> CommandResult:
        """Execute list-models command."""
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


class CleanupCommand(BaseCommand):
    """Command for cleaning up the output directory."""
    
    @property
    def name(self) -> str:
        return "cleanup"
    
    @property
    def description(self) -> str:
        return "Remove all files and directories in ./output/"
    
    @property
    def aliases(self):
        return ["clean"]
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add cleanup command arguments."""
        parser.add_argument('--force', '-f', action='store_true',
                          help='Force cleanup without confirmation prompt')
    
    def run(self, args: Namespace) -> CommandResult:
        """Execute cleanup command."""
        try:
            output_dir = Path('./output')
            
            if not output_dir.exists():
                self.logger.info("Output directory does not exist, nothing to clean")
                print("Output directory './output/' does not exist.")
                return self.success("No cleanup needed - directory does not exist")
            
            # Get list of items to be deleted for user confirmation
            items = list(output_dir.iterdir())
            if not items:
                self.logger.info("Output directory is already empty")
                print("Output directory './output/' is already empty.")
                return self.success("No cleanup needed - directory is empty")
            
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
            
            # Ask for confirmation unless --force is used
            if not args.force:
                print("\nAre you sure you want to delete all these items? This action cannot be undone.")
                response = input("Type 'yes' to confirm, or press Enter to cancel: ").strip().lower()
                
                if response != 'yes':
                    self.logger.info("Cleanup cancelled by user")
                    print("Cleanup cancelled.")
                    return self.success("Cleanup cancelled by user")
            
            # Perform cleanup
            self.logger.info("Starting cleanup of output directory")
            print("\nCleaning up...")
            
            success_count = 0
            error_count = 0
            
            for item in items:
                try:
                    if item.is_file():
                        item.unlink()
                        self.logger.debug(f"Deleted file: {item}")
                    elif item.is_dir():
                        shutil.rmtree(item)
                        self.logger.debug(f"Deleted directory: {item}")
                    success_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to delete {item}: {e}")
                    print(f"  ❌ Failed to delete {item.name}: {e}")
                    error_count += 1
            
            # Report results
            if error_count == 0:
                self.logger.info(f"Successfully cleaned up {success_count} items")
                print(f"\n✅ Successfully deleted {success_count} items from ./output/")
                return self.success(f"Successfully cleaned up {success_count} items", {
                    'items_deleted': success_count
                })
            else:
                self.logger.warning(f"Cleanup completed with errors: {success_count} succeeded, {error_count} failed")
                print(f"\n⚠️  Cleanup completed with errors:")
                print(f"  ✅ {success_count} items deleted successfully")
                print(f"  ❌ {error_count} items failed to delete")
                return self.success(f"Cleanup completed with {error_count} errors", {
                    'items_deleted': success_count,
                    'items_failed': error_count
                })
                
        except Exception as e:
            return self.handle_error(e, "Cleanup failed")


class InfoCommand(BaseCommand):
    """Command for displaying detailed information about all CLI options."""
    
    @property
    def name(self) -> str:
        return "info"
    
    @property
    def description(self) -> str:
        return "Display detailed information about all options"
    
    @property
    def aliases(self):
        return ["help-detailed"]
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add info command arguments."""
        pass  # No additional arguments needed
    
    def run(self, args: Namespace) -> CommandResult:
        """Execute info command."""
        try:
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

generate "Topic Name"
    Generate technical documentation for a specific topic.
    
    📝 Description:
    Creates HTML or Markdown documentation about any technical topic using
    AI-powered generation with few-shot examples.
    
    ✨ Examples:
    doc-gen generate "Machine Learning" --runs 3 --analyze
    doc-gen generate "Quantum Computing" --model gpt-4o --temperature 0.7
    
    💡 Best Practices:
    - Use quotes for multi-word topics
    - Combine with --runs 3+ for quality through variation
    - Add --analyze to get best compilation from multiple runs

readme /path/to/directory
    Generate README.md for a directory structure.
    
    📝 Description:
    Analyzes directory contents and creates comprehensive README documentation
    with automatic structure detection and contextualization.
    
    ✨ Examples:
    doc-gen readme ./my-project --runs 3 --analyze
    doc-gen readme /home/user/code --recursive --output-dir ./docs
    
    💡 Best Practices:
    - Default 3 runs for quality (automatic with README mode)
    - Use --recursive for nested directory documentation
    - Analysis is auto-enabled with multiple runs

standardize FILE_OR_URL
    Standardize existing documentation to organizational standards.
    
    📝 Description:
    Takes existing documentation (file or URL) and standardizes it according
    to organizational templates and formatting guidelines.
    
    ✨ Examples:
    doc-gen standardize ./existing-doc.md --template user_guide
    doc-gen standardize https://example.com/docs --target-format markdown

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

{'='*80}
📈 ANALYSIS & QUALITY
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

--recursive (README mode only)
    Generate README files for all subdirectories.
    
    📝 Description:
    Processes all subdirectories recursively, generating README files
    for each one based on their content and structure.

{'='*80}
🧹 UTILITY COMMANDS
{'='*80}

list-models
    Show all available models and providers.
    
    📝 Shows:
    - Available providers (OpenAI, Claude)
    - Supported models for each provider
    - Configuration status (API keys)

cleanup
    Remove all files and directories in ./output/.
    
    📝 Description:
    Interactive cleanup with:
    - Preview of files to be deleted
    - Confirmation prompt (use --force to skip)
    - Detailed success/failure reporting
    
    ⚠️ Warning: This action cannot be undone!
    
    ✨ Options:
    --force, -f    Skip confirmation prompt

info
    Display this detailed help information.

{'='*80}
💡 COMMON WORKFLOWS
{'='*80}

1️⃣ High-Quality Documentation Generation:
   doc-gen generate "Machine Learning" --runs 5 --analyze --model gpt-4o

2️⃣ Perfect README for Directory:
   doc-gen readme ./my-project --runs 3 --analyze

3️⃣ Recursive README Generation:
   doc-gen readme ./project --recursive --output-dir ./docs

4️⃣ Quick Draft Documentation:
   doc-gen generate "API Reference" --model gpt-4o-mini

5️⃣ Standardize Existing Documentation:
   doc-gen standardize ./old-docs.md --template technical_documentation

6️⃣ Clean Output Directory:
   doc-gen cleanup --force

{'='*80}
📚 For more information, visit:
   GitHub: https://github.com/fasrc/doc-generator
   Docs: https://docs.rc.fas.harvard.edu/
{'='*80}
"""
            print(info_text)
            return self.success("Displayed detailed help information")

        except Exception as e:
            return self.handle_error(e, "Failed to display info")


class ListPluginsCommand(BaseCommand):
    """Command for listing available plugins."""
    
    @property
    def name(self) -> str:
        return "list-plugins"
    
    @property
    def description(self) -> str:
        return "List all available plugins"
    
    @property
    def aliases(self):
        return ["lp", "plugins"]
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add list-plugins command arguments."""
        pass  # No additional arguments needed
    
    def run(self, args: Namespace) -> CommandResult:
        """Execute list-plugins command."""
        try:
            # Import and initialize a generator to access plugin manager
            from ...core import DocumentationGenerator
            
            generator = DocumentationGenerator(
                logger=self.logger
            )
            
            print("Available Recommendation Engine Plugins:")
            print("=" * 50)
            
            engines = generator.plugin_manager.list_engines()
            
            if not engines:
                print("No plugins loaded.")
                return self.success("No plugins available")
            
            for engine in engines:
                print(f"\nPlugin: {engine['name']}")
                print(f"  Class: {engine['class']}")
                print(f"  Module: {engine['module']}")
                print(f"  Supported Types: {', '.join(engine['supported_types'])}")
                print(f"  Priority: {engine['priority']}")
                print(f"  Enabled: {engine['enabled']}")

            return self.success(f"Listed {len(engines)} available plugins")

        except Exception as e:
            return self.handle_error(e, "Failed to list plugins")