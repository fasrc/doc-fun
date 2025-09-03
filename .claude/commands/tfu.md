---
name: tfu
description: Advanced test fixer with rollback, interactive mode, and robust error recovery
---

# Test Fix Update (TFU) Command

An advanced intelligent test runner that provides automated test fixing with robust safety measures, rollback capabilities, and interactive control.

## Overview

The `tfu` command provides a comprehensive test fixing system that:
- Executes the complete test suite with detailed analysis
- Intelligently categorizes and fixes errors with safety measures
- Provides rollback capabilities for failed fixes
- Offers interactive mode for reviewing changes before applying
- Uses staged fixing approach to minimize risk
- Generates detailed reports and change logs

## Key Features

### 🔒 Safety First
- **Backup Creation**: Automatically creates backups before making changes
- **Rollback Capability**: Can undo changes if fixes cause new failures
- **Staging Mode**: Apply fixes incrementally and test after each change
- **Dry Run Mode**: Preview all proposed changes without applying them

### 🎯 Interactive Control
- **Review Mode**: Review and approve each fix before applying
- **Selective Fixing**: Choose which types of errors to fix
- **Manual Override**: Skip automatic fixes for specific cases
- **Progress Tracking**: Real-time progress updates with detailed logging

### 🚀 Advanced Analysis
- **Dependency Detection**: Understand fix dependencies and order them correctly
- **Conflict Analysis**: Detect when multiple fixes might conflict
- **Impact Assessment**: Analyze the scope of changes before applying
- **Root Cause Analysis**: Identify underlying causes, not just symptoms

## Process

### 1. Pre-Flight Safety Check
- Create timestamped backup of entire test directory
- Validate git working directory is clean (or create stash)
- Check for existing backup conflicts
- Verify test framework compatibility

### 2. Comprehensive Test Analysis
- Detect test framework (pytest, unittest, jest, etc.)
- Run full test suite with maximum verbosity
- Capture stdout, stderr, and exit codes
- Parse and categorize all errors, failures, and warnings
- Generate dependency graph for failed tests

### 3. Intelligent Fix Planning
- Analyze error relationships and dependencies
- Detect potential fix conflicts before applying
- Order fixes by complexity and risk level
- Calculate fix confidence scores
- Prepare rollback strategies for each fix

### 4. Safe Fix Application
Choose from multiple execution modes:

#### Interactive Mode (Default)
```
🔍 Found 5 test failures
📋 Proposed fixes:
  1. [HIGH CONFIDENCE] Add missing import 'requests' to test_api.py
  2. [MEDIUM] Update assertion in test_utils.py:45 (expected 'foo', got 'bar')  
  3. [LOW] Install missing package 'pytest-mock'
  
👉 Apply fix 1? [y/n/s(kip)/q(uit)/d(ry-run)]: 
```

#### Staging Mode
- Apply one fix at a time
- Run affected tests after each fix
- Rollback individual fix if it causes new failures
- Continue with next fix only if current one succeeds

#### Batch Mode
- Apply all high-confidence fixes automatically
- Prompt for medium and low confidence fixes
- Stop on first failure and offer rollback

### 5. Advanced Fix Strategies

#### Import Errors
- **Smart Import Resolution**: Try multiple import paths
- **Package Installation**: Use pip, conda, or poetry as appropriate
- **Version Compatibility**: Check version conflicts before installing
- **Import Ordering**: Respect PEP8 import ordering

#### Type Errors  
- **Type Inference**: Analyze expected types from context
- **Generic Handling**: Properly handle generic types and constraints
- **Optional Types**: Detect when Optional[] wrapper is needed
- **Protocol Compliance**: Ensure structural typing compatibility

#### Assertion Failures
- **Value Analysis**: Deep comparison of expected vs actual
- **Floating Point**: Handle floating point comparison tolerances
- **Collection Comparison**: Smart diff for lists, dicts, sets
- **Mock Verification**: Analyze mock call expectations

#### Configuration Issues
- **Framework Migration**: Update deprecated pytest/unittest patterns
- **Plugin Conflicts**: Detect and resolve plugin version conflicts
- **Environment Variables**: Set appropriate test environment variables
- **Path Resolution**: Fix relative path issues in tests

### 6. Rollback and Recovery

#### Automatic Rollback Triggers
- New test failures introduced by fixes
- Compilation/syntax errors after fixes
- Import resolution failures
- Critical test infrastructure damage

#### Manual Rollback Options
```bash
# Rollback last fix
tfu --rollback-last

# Rollback all changes from session
tfu --rollback-all

# Rollback specific fix by ID
tfu --rollback-fix 3

# Show rollback history
tfu --rollback-history
```

### 7. Comprehensive Reporting

#### Fix Session Report
```
🎯 TFU Session Report
===================
📅 Started: 2024-12-07 14:30:22
⏱️  Duration: 3m 45s
📊 Tests: 127 total, 5 failed → 0 failed

✅ Successful Fixes:
  1. Added missing import 'requests' to test_api.py
  2. Updated assertion tolerance in test_math.py:67
  3. Installed pytest-mock package

⚠️  Manual Review Recommended:
  1. Complex mock expectation in test_service.py:123
  
💾 Backup created: .tfu_backup_20241207_143022/
📝 Detailed log: .tfu_session.log
```

#### Change Tracking
- Detailed diff of all changes made
- Change attribution by fix type and confidence
- Revert instructions for each change
- Impact analysis on test coverage

## Configuration

### Configuration File: `.tfu.yml`
```yaml
# TFU Configuration
safety:
  backup_enabled: true
  backup_path: ".tfu_backups/"
  rollback_enabled: true
  staging_mode: true

fix_confidence:
  high_threshold: 0.9
  medium_threshold: 0.7
  auto_apply_high: true
  prompt_medium: true
  skip_low: false

frameworks:
  pytest:
    extra_args: ["-v", "--tb=short"]
    markers_to_skip: ["slow", "integration"]
  unittest:
    discovery_pattern: "test*.py"
    verbosity: 2

notifications:
  success_sound: true
  desktop_notifications: true
  slack_webhook: null
```

### Environment Variables
```bash
# Override default behavior
export TFU_MODE=interactive          # interactive|staging|batch
export TFU_CONFIDENCE_THRESHOLD=0.8  # minimum confidence for auto-apply
export TFU_BACKUP_PATH=./backups     # custom backup location
export TFU_DRY_RUN=true              # preview mode only
```

## Advanced Usage

### Command Line Options
```bash
# Basic usage
tfu

# Interactive mode with staging
tfu --mode interactive --staging

# Batch mode for CI/CD
tfu --mode batch --confidence-threshold 0.9

# Dry run to preview changes
tfu --dry-run

# Fix only specific error types
tfu --fix-types imports,assertions

# Skip specific tests
tfu --skip-tests "test_integration*"

# Custom backup location
tfu --backup-path ./my-backups/

# Restore from specific backup
tfu --restore .tfu_backup_20241207_143022/
```

### Integration Examples

#### Pre-commit Hook
```bash
#!/bin/bash
# Run TFU in batch mode before commit
tfu --mode batch --confidence-threshold 0.9 --quiet
if [ $? -eq 0 ]; then
    echo "✅ All tests fixed and passing"
else  
    echo "❌ TFU couldn't fix all issues - manual review required"
    exit 1
fi
```

#### CI/CD Integration
```yaml
# GitHub Actions
- name: Fix tests with TFU
  run: |
    tfu --mode batch --confidence-threshold 0.95 --no-backup
    pytest --tb=short
```

## Safety Measures

### Protected Operations
- **Fixture Preservation**: Never modify test fixtures without explicit approval
- **Mock Behavior**: Preserve mock configurations and call patterns
- **Test Data**: Protect test data files from modification
- **Coverage Thresholds**: Ensure fixes don't reduce test coverage

### Risk Assessment
Each fix is assigned a risk level:
- 🟢 **LOW**: Simple imports, basic assertions, formatting
- 🟡 **MEDIUM**: Type annotations, mock updates, configuration changes  
- 🔴 **HIGH**: Logic changes, fixture modifications, complex refactoring

### Review Requirements
Automatic escalation to manual review for:
- Changes affecting more than 5 test files
- Modifications to core test infrastructure  
- Fixes with confidence below configurable threshold
- Any change that reduces test coverage
- Modifications to shared fixtures or conftest.py

## Error Recovery

### Partial Fix Failure
If a fix partially succeeds but causes new issues:
1. Immediate rollback of the problematic change
2. Analysis of what went wrong
3. Suggest alternative fix approaches
4. Option to continue with remaining fixes

### Complete Session Failure
If the entire fix session fails:
1. Automatic rollback to session start state
2. Detailed failure analysis report
3. Recommendations for manual intervention
4. Preservation of analysis results for debugging

## Performance Optimization

### Parallel Processing
- Run independent fixes in parallel when safe
- Parallel test execution for validation
- Async package installation
- Concurrent backup creation

### Caching
- Cache test results between fix attempts
- Remember successful fix patterns
- Store package installation results
- Maintain fix confidence history

### Incremental Mode
- Only re-run affected tests after each fix
- Skip unchanged test files in validation
- Differential backup (only changed files)
- Smart dependency detection

## Notes

### Best Practices
- Always run in interactive mode for first-time use on a project
- Review backup contents before cleaning old backups
- Use staging mode for critical codebases
- Keep TFU configuration in version control
- Regular backup cleanup to manage disk space

### Limitations
- Cannot fix logical test errors requiring domain knowledge
- May struggle with complex mock configurations
- Limited support for property-based testing frameworks
- Some dynamic import patterns may not be detectable

### Compatibility
- Python 3.7+
- pytest 6.0+, unittest (built-in)
- Node.js 14+ for JavaScript/TypeScript projects
- Go 1.16+ for Go projects
- Rust 1.50+ for Rust projects

---

*TFU: Making test fixing safer, smarter, and more reliable.*