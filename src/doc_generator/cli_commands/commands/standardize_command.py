"""
Standardize Command

Implements documentation standardization functionality for files and URLs.
"""

import sys
import logging
import urllib.parse
import urllib.request
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional
from urllib.error import URLError

from dotenv import load_dotenv

from ..base import BaseCommand, CommandResult
from ...utils import get_output_directory
from ...exceptions import DocGeneratorError


class StandardizeCommand(BaseCommand):
    """Command for standardizing existing documentation to organizational standards."""
    
    @property
    def name(self) -> str:
        return "standardize"
    
    @property
    def description(self) -> str:
        return "Standardize existing documentation to organizational standards (accepts file path or URL)"
    
    @property
    def aliases(self):
        return ["std", "s"]
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add standardize command arguments."""
        # Required positional argument
        parser.add_argument('input_source', 
                          help='File path or URL to standardize')
        
        # Generation options
        parser.add_argument('--model', 
                          help='Model to use (e.g., gpt-4o-mini, claude-3-5-sonnet-20240620)')
        parser.add_argument('--provider', choices=['openai', 'claude', 'auto'], 
                          default='auto',
                          help='LLM provider to use (default: auto-detect)')
        parser.add_argument('--temperature', type=float, default=0.3,
                          help='Temperature for text generation (default: 0.3)')
        parser.add_argument('--output-dir', default='output',
                          help='Output directory (default: ./output)')
        
        # Standardization options
        parser.add_argument('--template', choices=['technical_documentation', 'user_guide', 'api_documentation'],
                          default='technical_documentation',
                          help='Standardization template to use (default: technical_documentation)')
        parser.add_argument('--target-format', choices=['html', 'markdown'],
                          help='Target format for standardized output (overrides --format)')
        parser.add_argument('--format', choices=['html', 'markdown', 'auto'],
                          default='auto',
                          help='Output format for documentation (default: auto-detect)')
        
        # Plugin management
        parser.add_argument('--disable-plugins', nargs='*', metavar='PLUGIN',
                          help='Disable specific plugins by name')
        parser.add_argument('--enable-only', nargs='*', metavar='PLUGIN',
                          help='Enable only specified plugins (disable all others)')
    
    def validate_args(self, args: Namespace) -> None:
        """Validate command arguments."""
        # Check if input is a URL or file path
        parsed = urllib.parse.urlparse(args.input_source)
        is_url = parsed.scheme in ('http', 'https')
        
        if not is_url:
            # Validate file exists
            input_file = Path(args.input_source)
            if not input_file.exists():
                raise DocGeneratorError(f"Input file not found: {args.input_source}")
            if not input_file.is_file():
                raise DocGeneratorError(f"Path is not a file: {args.input_source}")
        
        # Validate temperature range
        if not (0.0 <= args.temperature <= 2.0):
            raise DocGeneratorError("Temperature must be between 0.0 and 2.0")
    
    def run(self, args: Namespace) -> CommandResult:
        """Execute document standardization."""
        try:
            # Load environment variables
            load_dotenv()
            
            self.logger.info(f"Standardizing document: {args.input_source}")
            
            # Process input source (URL or file)
            content, input_name, file_path, is_url = self._process_input_source(args.input_source)
            
            # Determine output format
            target_format = self._determine_output_format(args, content, is_url, args.input_source)
            
            # Initialize standardizer
            from ...standardizers import DocumentStandardizer
            
            standardizer = DocumentStandardizer(
                provider=args.provider if args.provider != 'auto' else None,
                model=args.model,
                temperature=args.temperature,
                output_format=target_format
            )
            
            # Determine output directory and file
            output_dir = get_output_directory(args.output_dir, self.logger)
            
            # Generate output filename
            if target_format == 'markdown':
                ext = '.md'
            elif target_format == 'html':
                ext = '.html'
            else:
                ext = '.txt'
            
            output_file = Path(output_dir) / f"{input_name}_standardized{ext}"
            
            # Standardize document
            self.logger.info(f"Using template: {args.template}")
            self.logger.info(f"Target format: {target_format}")
            
            result = standardizer.standardize_document(
                content=content,
                file_path=file_path,
                target_format=target_format
            )
            
            # Write output file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result['standardized_content'])
            
            # Update result with output path
            result['output_path'] = str(output_file)
            
            # Display results if not quiet
            if not getattr(args, 'quiet', False):
                self._display_results(result, args.input_source, is_url, output_file)
            
            self.logger.info("Document standardization completed successfully")
            
            message = f"Successfully standardized document: {input_name}"
            return self.success(message, {
                'input_source': args.input_source,
                'output_file': str(output_file),
                'target_format': target_format,
                'template': args.template,
                'sections_processed': len(result['sections_processed'])
            })
            
        except Exception as e:
            return self.handle_error(e, "Document standardization failed")
    
    def _process_input_source(self, input_source: str) -> tuple[str, str, str, bool]:
        """
        Process input source (URL or file) and return content, name, path, and is_url flag.
        
        Returns:
            Tuple of (content, input_name, file_path, is_url)
        """
        # Check if input is a URL
        parsed = urllib.parse.urlparse(input_source)
        if parsed.scheme in ('http', 'https'):
            is_url = True
            self.logger.info(f"Input detected as URL: {input_source}")
            
            # Fetch content from URL
            try:
                with urllib.request.urlopen(input_source) as response:
                    content = response.read().decode('utf-8')
                input_name = parsed.path.split('/')[-1] or 'document'
                # Remove file extension for naming
                if '.' in input_name:
                    input_name = input_name.rsplit('.', 1)[0]
                file_path = input_source  # Use URL as file_path for format detection
                return content, input_name, file_path, is_url
                
            except URLError as e:
                raise DocGeneratorError(f"Failed to fetch URL: {e}")
            except UnicodeDecodeError as e:
                raise DocGeneratorError(f"Failed to decode content from URL: {e}")
        else:
            # Treat as file path
            input_file = Path(input_source)
            if not input_file.exists():
                raise DocGeneratorError(f"Input file not found: {input_file}")
            
            # Read file content
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            input_name = input_file.stem
            file_path = str(input_file)
            return content, input_name, file_path, False
    
    def _determine_output_format(self, args: Namespace, content: str, is_url: bool, input_source: str) -> str:
        """Determine the target output format based on arguments and input."""
        target_format = args.target_format or args.format
        
        if target_format == 'auto':
            # Auto-detect based on content or input
            if is_url:
                # For URLs, default to HTML unless content suggests otherwise
                if content and content.strip().startswith('#') and '\n#' in content:
                    target_format = 'markdown'  # Looks like Markdown
                else:
                    target_format = 'html'  # Default for URLs
            else:
                # For files, use extension
                input_file = Path(input_source)
                if input_file.suffix.lower() in ['.html', '.htm']:
                    target_format = 'markdown'  # Convert HTML to Markdown by default
                else:
                    target_format = 'html'  # Default to HTML
        
        return target_format
    
    def _display_results(self, result: dict, input_source: str, is_url: bool, output_file: Path) -> None:
        """Display standardization results to user."""
        print(f"\n📄 Document Standardization Results")
        if is_url:
            print(f"🔗 Input URL: {input_source}")
        else:
            print(f"📁 Input File: {input_source}")
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