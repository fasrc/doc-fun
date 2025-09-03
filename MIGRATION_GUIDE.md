# Phase 2 CLI Refactoring Migration Guide

## 📋 Overview

This guide helps developers understand and adapt to the Phase 2 CLI refactoring, which introduces a command pattern architecture while maintaining full backward compatibility.

## 🎯 What Changed

### Before (Phase 1)
```bash
# Old flag-based interface
doc-gen --topic "Machine Learning" --runs 3 --analyze
doc-gen --readme ./project --recursive
doc-gen --standardize document.html --target-format markdown
doc-gen --list-models
doc-gen --cleanup
```

### After (Phase 2) - New Command Pattern
```bash
# New command-based interface (recommended)
doc-gen generate "Machine Learning" --runs 3 --analyze
doc-gen readme ./project --recursive  
doc-gen standardize document.html --target-format markdown
doc-gen list-models
doc-gen cleanup

# Alternative shorter aliases
doc-gen gen "Machine Learning" --runs 3 --analyze
doc-gen r ./project --recursive
doc-gen std document.html --target-format markdown
doc-gen lm
doc-gen clean
```

### Backward Compatibility (Still Works)
```bash
# Old interface continues to work unchanged
doc-gen --topic "Machine Learning" --runs 3 --analyze  # ✅ Still works
doc-gen --readme ./project --recursive                 # ✅ Still works  
doc-gen --list-models                                  # ✅ Still works
```

## 🏗️ Architecture Changes

### Command Pattern Structure
```
src/doc_generator/cli_commands/
├── __init__.py                    # Package initialization
├── base.py                        # BaseCommand abstract class
├── registry.py                    # Command registration system
├── dispatcher.py                  # Command routing and execution
├── bootstrap.py                   # Command discovery and setup
├── main.py                        # New CLI entry point
└── commands/                      # Individual command implementations
    ├── __init__.py
    ├── generate_command.py         # --topic functionality
    ├── readme_command.py           # --readme functionality  
    ├── standardize_command.py      # --standardize functionality
    ├── test_command.py             # Testing utilities
    └── utility_commands.py         # list-models, cleanup, info, etc.
```

### Key Components

#### BaseCommand Abstract Class
```python
from src.doc_generator.cli_commands.base import BaseCommand, CommandResult

class MyCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "my-command"
    
    @property  
    def description(self) -> str:
        return "Description of my command"
    
    @property
    def aliases(self) -> List[str]:
        return ["mc", "my"]
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('--option', help='My option')
    
    def validate_args(self, args: Namespace) -> None:
        # Validate arguments
        pass
    
    def run(self, args: Namespace) -> CommandResult:
        # Implement command logic
        return self.success("Command completed successfully")
```

#### Command Registry
```python
from src.doc_generator.cli_commands.registry import CommandRegistry

registry = CommandRegistry()
registry.register(MyCommand)
```

#### Command Dispatcher
```python
from src.doc_generator.cli_commands.dispatcher import CommandDispatcher

dispatcher = CommandDispatcher(registry)
result = dispatcher.dispatch(['my-command', '--option', 'value'])
```

## 🔄 Migration Strategies

### For End Users
**No action required** - All existing CLI usage continues to work without changes.

### For Scripts and Automation
```bash
# Option 1: Keep using old format (recommended for stability)
doc-gen --topic "Machine Learning" --runs 3

# Option 2: Migrate to new format (recommended for new scripts)
doc-gen generate "Machine Learning" --runs 3

# Option 3: Gradual migration
# Old: doc-gen --list-models && doc-gen --topic "ML" --analyze
# New: doc-gen list-models && doc-gen generate "ML" --analyze
```

### For Developers Adding New Commands

#### Old Way (Phase 1)
```python
# Had to modify cli.py directly
def parse_args():
    parser.add_argument('--my-feature', help='My feature')
    
def main():
    if args.my_feature:
        run_my_feature(args)
```

#### New Way (Phase 2)
```python
# Create isolated command class
class MyFeatureCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "my-feature"
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('option', help='Required option')
        parser.add_argument('--flag', action='store_true')
    
    def run(self, args: Namespace) -> CommandResult:
        # Implement feature
        return self.success("Feature completed")

# Register in bootstrap.py
registry.register(MyFeatureCommand)
```

### For Testing

#### Old Testing Approach
```python
# Had to mock entire CLI system
with patch('sys.argv', ['doc-gen', '--topic', 'Test']):
    from doc_generator.cli import main
    main()
```

#### New Testing Approach  
```python
# Test individual commands directly
from src.doc_generator.cli_commands.commands.generate_command import GenerateCommand

def test_generate_command():
    cmd = GenerateCommand()
    args = Mock(topic="Test Topic", runs=1, temperature=0.3)
    result = cmd.run(args)
    assert result.success
    
# Or test through dispatcher
from src.doc_generator.cli_commands.main import main

def test_command_integration():
    result = main(['generate', 'Test Topic', '--runs', '2'])
    assert result == 0
```

## 📊 Performance Impact

Based on benchmarking results:
- **List Models**: 1.8% slower (within tolerance)
- **Info Display**: 0.9% faster  
- **Memory Usage**: ~40MB (reasonable)
- **Startup Time**: Equivalent (~460ms)

**Verdict**: No significant performance regression.

## 🐛 Troubleshooting

### Common Issues

#### 1. Command Not Found
```bash
$ doc-gen my-command
Error: Unknown command: my-command
```
**Solution**: Ensure command is registered in `bootstrap.py`

#### 2. Argument Validation Failed
```bash
$ doc-gen generate "Test" --temperature 5.0
Error: Argument validation failed: Temperature must be between 0.0 and 2.0
```
**Solution**: Check argument constraints in command's `validate_args()` method

#### 3. Import Errors
```python
ImportError: cannot import name 'GenerateCommand'
```
**Solution**: Use full import paths:
```python
from src.doc_generator.cli_commands.commands.generate_command import GenerateCommand
```

#### 4. Backward Compatibility Issues
If old CLI format stops working:
1. Check `src/doc_generator/cli.py` main() function
2. Verify command detection logic
3. Ensure legacy argument parsing is preserved

### Debugging Tips

#### Enable Verbose Logging
```bash
doc-gen --verbose generate "Test Topic"
doc-gen generate "Test Topic" --verbose  # New format
```

#### Test Command Registration
```python
from src.doc_generator.cli_commands.bootstrap import create_default_registry

registry = create_default_registry()
print(registry.list_commands())  # Should show all commands
```

#### Validate Command Parsing
```bash
doc-gen generate --help  # Test command-specific help
doc-gen --help            # Test global help
```

## 🧪 Testing Your Migration

### 1. Verify Backward Compatibility
```bash
# Test all your existing CLI usage still works
doc-gen --topic "Test" --runs 1
doc-gen --readme ./test-project
doc-gen --list-models
```

### 2. Test New Command Format
```bash  
# Try equivalent new commands
doc-gen generate "Test" --runs 1
doc-gen readme ./test-project
doc-gen list-models
```

### 3. Run Integration Tests
```bash
# Run the test suite
pytest tests/test_cli_commands.py -v
pytest tests/test_cli_simple.py -v  # Existing tests should pass
```

### 4. Performance Verification
```bash
# Run benchmark to check performance
python performance_benchmark.py
```

## 🚀 Best Practices

### For New Development
1. **Use the new command format** for new scripts and documentation
2. **Create focused commands** - each command should have a single responsibility
3. **Include proper validation** - validate arguments in `validate_args()`
4. **Add comprehensive tests** - test both success and error cases
5. **Document your commands** - provide clear descriptions and help text

### For Existing Code
1. **No rush to migrate** - old format will continue working
2. **Gradual adoption** - migrate when you're already modifying scripts
3. **Test thoroughly** - verify behavior is identical between old and new formats
4. **Update documentation** - show both old and new formats during transition

### For Command Development
```python
class NewCommand(BaseCommand):
    """Template for new commands."""
    
    @property
    def name(self) -> str:
        return "command-name"  # Use kebab-case
    
    @property
    def description(self) -> str:
        return "Clear description of what this command does"
    
    @property
    def aliases(self) -> List[str]:
        return ["short", "alias"]  # Provide convenient shortcuts
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add arguments with proper help text."""
        parser.add_argument('required_arg', help='Required argument')
        parser.add_argument('--optional', default='default', help='Optional argument')
    
    def validate_args(self, args: Namespace) -> None:
        """Validate arguments and throw DocGeneratorError if invalid."""
        if not args.required_arg:
            raise DocGeneratorError("Required argument missing")
    
    def run(self, args: Namespace) -> CommandResult:
        """Implement the command logic."""
        try:
            # Do work here
            result = self.do_work(args)
            return self.success(f"Completed successfully: {result}")
        except Exception as e:
            return self.handle_error(e, "Command failed")
```

## 📚 Additional Resources

### Documentation
- `src/doc_generator/cli_commands/base.py` - BaseCommand interface
- `src/doc_generator/cli_commands/registry.py` - Command registration
- `src/doc_generator/cli_commands/dispatcher.py` - Command routing
- `tests/test_cli_commands.py` - Example test patterns

### Example Commands
- `GenerateCommand` - Full-featured command with validation and analysis
- `ReadmeCommand` - Directory processing with recursive options  
- `StandardizeCommand` - File/URL input handling
- `ListModelsCommand` - Simple utility command
- `CleanupCommand` - Interactive command with confirmation

### Support
- Check existing tests for patterns and examples
- Review `CLAUDE.md` for updated CLI documentation
- Use `doc-gen --help` and `doc-gen <command> --help` for reference

## 🎯 Summary

The Phase 2 CLI refactoring provides:
- ✅ **100% Backward Compatibility** - All existing usage continues to work
- ✅ **Improved Architecture** - Clean command pattern with separation of concerns
- ✅ **Enhanced Testing** - Better test coverage and isolated command testing
- ✅ **Future Extensibility** - Easy to add new commands without modifying core CLI
- ✅ **Performance Maintained** - No significant performance regression

Migration is **optional but recommended** for new development. The new command pattern provides better organization, testing, and maintainability while preserving all existing functionality.