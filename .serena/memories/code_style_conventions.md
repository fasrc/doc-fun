# Code Style and Conventions

## Formatting and Style
- **Formatter**: Black with 100-character line length
- **Linter**: flake8 for style enforcement
- **Type Checker**: mypy with strict settings enabled

## Python Code Conventions
- **Type Hints**: Required for all function definitions
  - `disallow_untyped_defs = true`
  - `disallow_incomplete_defs = true`
- **Imports**: Follow PEP 8 import ordering
- **Naming**: Follow PEP 8 naming conventions
  - Classes: PascalCase (e.g., `DocumentationGenerator`)
  - Functions/variables: snake_case (e.g., `generate_documentation`)
  - Constants: UPPER_SNAKE_CASE (e.g., `DEFAULT_MODEL`)

## Documentation Standards
- **Docstrings**: Required for all public methods and classes
- **Format**: Google-style docstrings
- **Example from codebase**:
  ```python
  def __init__(self, prompt_yaml_path: str = None, shots_dir: str = None,
               terminology_path: str = None, provider: str = 'auto', 
               logger: Optional[logging.Logger] = None):
      """Initialize the documentation generator with configuration."""
  ```

## Type Annotations
- **Required**: All function parameters and return types
- **Optional Types**: Use `Optional[Type]` from typing module
- **Complex Types**: Use proper generic types (List, Dict, etc.)
- **Example**:
  ```python
  def load_examples(self) -> List[Dict[str, str]]:
  ```

## Error Handling
- **Custom Exceptions**: Use structured exception hierarchy
- **Context**: Include relevant context in exceptions
- **Logging**: Use logger for debugging and error reporting
- **Graceful Degradation**: Handle failures gracefully where possible

## Project-Specific Patterns
- **Configuration**: Use pydantic for settings validation
- **Caching**: Apply `@cached` decorator for expensive operations
- **Plugin System**: Inherit from base classes for extensibility
- **Provider Pattern**: Implement unified interfaces for different services

## File Organization
- **Module Structure**: Clear separation of concerns
- **Imports**: Group into standard library, third-party, local imports
- **Class Organization**: Public methods before private methods
- **Method Length**: Keep methods focused and reasonably short

## Testing Conventions
- **Test Files**: `test_*.py` pattern
- **Test Classes**: `Test*` pattern  
- **Test Functions**: `test_*` pattern
- **Fixtures**: Use pytest fixtures for setup/teardown
- **Mocking**: Use pytest-mock for external dependencies
- **Markers**: Use `@pytest.mark.unit` and `@pytest.mark.integration`