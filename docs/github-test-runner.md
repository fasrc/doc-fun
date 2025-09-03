# GitHub Test Runner and Analysis

A comprehensive Python function for running tests on GitHub Actions and performing detailed analysis of test results with intelligent insights and recommendations.

## Features

🔍 **Multi-Framework Support**
- Pytest, unittest, Jest, Mocha, Go test, Cargo test, Maven, Gradle
- Auto-detection of test frameworks
- Unified result parsing and analysis

📊 **Comprehensive Analysis**
- Test result categorization and failure analysis
- Performance profiling and slow test detection
- Coverage analysis and reporting
- Trend analysis (with historical data)
- Quality metrics and recommendations

🚀 **GitHub Integration**
- Native GitHub Actions environment detection
- Automated job summaries with rich formatting
- PR comments with detailed analysis
- Artifact management for test results
- Quality gate enforcement

🛡️ **Advanced Features**
- Flaky test detection
- Security test integration
- Performance benchmarking
- Multi-matrix test analysis
- Custom notification systems

## Quick Start

### Basic Usage

```python
from doc_generator.github_test_runner import run_github_tests

# Simple one-line usage
analysis = run_github_tests(
    test_path="tests/",
    enable_coverage=True,
    performance_threshold=3.0
)

print(f"Success rate: {analysis.suite_results.success_rate:.1f}%")
```

### GitHub Actions Integration

```yaml
name: Test Analysis
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: 3.10
    
    - name: Install dependencies
      run: pip install -e .[test]
    
    - name: Run test analysis
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      run: |
        python -m doc_generator.github_test_runner \
          --test-path tests/ \
          --github-token "$GITHUB_TOKEN" \
          --performance-threshold 3.0
```

## Core Components

### GitHubTestRunner Class

The main class that orchestrates test execution and analysis:

```python
from doc_generator.github_test_runner import GitHubTestRunner

runner = GitHubTestRunner(
    workspace_path=".",
    github_token=os.getenv("GITHUB_TOKEN"),
    enable_coverage=True,
    performance_threshold=5.0
)

# Run tests
suite_result = runner.run_tests("tests/", extra_args=["--maxfail=3"])

# Analyze results  
analysis = runner.analyze_results(suite_result)

# Generate reports
summary = runner.generate_github_summary(analysis)
```

### Data Structures

#### TestResult
Individual test execution result:
```python
@dataclass
class TestResult:
    name: str
    status: TestStatus  # PASSED, FAILED, SKIPPED, ERROR
    duration: float
    file_path: str
    line_number: Optional[int]
    class_name: Optional[str]
    error_message: Optional[str]
    failure_message: Optional[str]
```

#### TestSuiteResult
Complete test suite execution results:
```python
@dataclass
class TestSuiteResult:
    framework: TestFramework
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    coverage_percent: Optional[float]
    tests: List[TestResult]
    
    @property
    def success_rate(self) -> float
    @property  
    def is_healthy(self) -> bool
```

#### TestAnalysis
Comprehensive analysis results with insights:
```python
@dataclass
class TestAnalysis:
    suite_results: TestSuiteResult
    github_context: GitHubContext
    analysis_timestamp: datetime
    
    # Analysis Results
    failure_categories: Dict[str, List[TestResult]]
    flaky_tests: List[TestResult]
    slow_tests: List[TestResult]
    
    # Performance Metrics
    avg_test_duration: float
    slowest_test: Optional[TestResult]
    fastest_test: Optional[TestResult]
    
    # Insights
    recommendations: List[str]
    critical_issues: List[str]
```

## Framework Support

### Python - Pytest
```bash
# Auto-detected if pytest.ini or pyproject.toml exists
python -m doc_generator.github_test_runner --test-path tests/
```

Features:
- JUnit XML result parsing
- Coverage integration via `--cov`
- Performance timing per test
- Rich failure categorization

### Python - Unittest
```bash
# Auto-detected if test_*.py files exist
python -m doc_generator.github_test_runner --test-path tests/
```

Features:
- Text output parsing
- Basic test categorization
- Failure detection and analysis

### JavaScript - Jest
```bash
# Auto-detected if jest in package.json dependencies
python -m doc_generator.github_test_runner --test-path src/
```

Features:
- JSON result parsing
- Snapshot test analysis  
- Performance profiling
- Coverage integration

### Go
```bash
# Auto-detected if *_test.go files exist
python -m doc_generator.github_test_runner --test-path ./...
```

Features:
- JSON output parsing (`go test -json`)
- Package-level analysis
- Benchmark integration
- Performance profiling

### Rust - Cargo
```bash  
# Auto-detected if Cargo.toml exists
python -m doc_generator.github_test_runner --test-path ""
```

Features:
- Text output parsing
- Integration test support
- Documentation test analysis
- Performance metrics

## Analysis Features

### Failure Categorization

Tests failures are automatically categorized:

- **Import Errors**: Missing dependencies, module not found
- **Assertion Failures**: Test expectation mismatches
- **Type Errors**: Type-related issues and AttributeError
- **Timeout/Performance**: Slow or hanging tests
- **Configuration Issues**: Setup, fixture, or config problems
- **Unknown Errors**: Uncategorized failures

### Performance Analysis

```python
# Configure performance thresholds
runner = GitHubTestRunner(performance_threshold=2.0)
analysis = runner.analyze_results(suite_result)

print(f"Slow tests: {len(analysis.slow_tests)}")
print(f"Average duration: {analysis.avg_test_duration:.2f}s")

if analysis.slowest_test:
    print(f"Slowest: {analysis.slowest_test.name} ({analysis.slowest_test.duration:.2f}s)")
```

### Coverage Integration

```python
# Enable coverage collection
analysis = run_github_tests(
    test_path="tests/",
    enable_coverage=True
)

if analysis.suite_results.coverage_percent:
    print(f"Coverage: {analysis.suite_results.coverage_percent:.1f}%")
```

### Quality Gates

```python
def check_quality_gates(analysis: TestAnalysis) -> bool:
    """Implement custom quality gates."""
    suite = analysis.suite_results
    
    # Define criteria
    gates = [
        suite.success_rate >= 95.0,           # 95% success rate
        suite.failed == 0,                   # No failed tests
        len(analysis.slow_tests) <= 5,       # Max 5 slow tests
        not analysis.critical_issues,        # No critical issues
    ]
    
    # Optional coverage gate
    if suite.coverage_percent:
        gates.append(suite.coverage_percent >= 80.0)
    
    return all(gates)

# Usage
passed_gates = check_quality_gates(analysis)
print(f"Quality gates: {'✅ PASSED' if passed_gates else '❌ FAILED'}")
```

## GitHub Integration

### Environment Variables

The runner automatically detects GitHub Actions environment:

```bash
GITHUB_REPOSITORY    # Repository name (owner/repo)
GITHUB_REF          # Git reference
GITHUB_SHA          # Commit SHA
GITHUB_ACTOR        # User who triggered the workflow
GITHUB_WORKFLOW     # Workflow name
GITHUB_RUN_ID       # Unique run identifier
GITHUB_RUN_NUMBER   # Sequential run number
GITHUB_JOB          # Current job name
GITHUB_TOKEN        # GitHub API token (for PR comments)
```

### Job Summaries

Automatic GitHub Actions job summaries:

```python
# Enable job summary (default: True)
analysis = run_github_tests(post_summary=True)
```

Generated summary includes:
- 📊 Test results overview
- 🔍 Failure analysis breakdown  
- ⚡ Performance insights
- 🚨 Critical issues
- 💡 Actionable recommendations

### PR Comments

```python
# Post analysis as PR comment
runner = GitHubTestRunner(github_token=os.getenv("GITHUB_TOKEN"))
suite_result = runner.run_tests("tests/")
analysis = runner.analyze_results(suite_result)

# Post to PR
pr_number = 123
success = runner.post_pr_comment(analysis, pr_number)
```

### Artifacts

```yaml
- name: Upload test results
  uses: actions/upload-artifact@v3
  if: always()
  with:
    name: test-analysis
    path: |
      test-results.xml
      coverage.xml
      test-analysis.json
    retention-days: 30
```

## Advanced Usage

### Custom Analysis

```python
from doc_generator.github_test_runner import GitHubTestRunner

class CustomTestRunner(GitHubTestRunner):
    def _categorize_failures(self, tests):
        """Custom failure categorization."""
        categories = super()._categorize_failures(tests)
        
        # Add custom category
        categories["Database Errors"] = [
            test for test in tests 
            if test.status == TestStatus.FAILED 
            and "database" in (test.error_message or "").lower()
        ]
        
        return categories
    
    def _generate_recommendations(self, suite_result, failure_categories, slow_tests):
        """Custom recommendations."""
        recommendations = super()._generate_recommendations(
            suite_result, failure_categories, slow_tests
        )
        
        # Add custom recommendations
        if "Database Errors" in failure_categories:
            recommendations.append("Consider using test database fixtures or mocking.")
        
        return recommendations

# Usage
runner = CustomTestRunner()
analysis = runner.analyze_results(suite_result)
```

### Multi-Framework Projects

```python
def analyze_multi_framework_project():
    """Analyze project with multiple test frameworks."""
    test_configs = [
        ("tests/backend/", "Python Backend Tests"),
        ("tests/frontend/", "JavaScript Frontend Tests"),
        ("tests/integration/", "Integration Tests"),
    ]
    
    all_results = {}
    
    for test_path, description in test_configs:
        if Path(test_path).exists():
            print(f"Running {description}...")
            
            analysis = run_github_tests(
                test_path=test_path,
                save_results=False,
                post_summary=False
            )
            
            all_results[description] = analysis
    
    # Generate combined report
    generate_combined_report(all_results)

def generate_combined_report(results: dict):
    """Generate combined analysis report."""
    total_tests = sum(a.suite_results.total_tests for a in results.values())
    total_passed = sum(a.suite_results.passed for a in results.values())
    
    print(f"\n📊 Combined Results:")
    print(f"Total Tests: {total_tests}")
    print(f"Total Passed: {total_passed}")
    print(f"Overall Success Rate: {total_passed/total_tests*100:.1f}%")
    
    for description, analysis in results.items():
        suite = analysis.suite_results
        print(f"\n{description}:")
        print(f"  ✅ {suite.passed} ❌ {suite.failed} ⏭️ {suite.skipped}")
        print(f"  Success Rate: {suite.success_rate:.1f}%")
```

### Performance Benchmarking

```python
def run_performance_analysis():
    """Run performance-focused test analysis."""
    # Strict performance threshold
    analysis = run_github_tests(
        test_path="tests/performance/",
        performance_threshold=1.0,  # 1 second
        enable_coverage=False       # Skip for performance focus
    )
    
    # Performance grading
    avg_duration = analysis.avg_test_duration
    if avg_duration < 0.1:
        grade = "A+"
    elif avg_duration < 0.5:
        grade = "A" 
    elif avg_duration < 1.0:
        grade = "B"
    else:
        grade = "C"
    
    print(f"Performance Grade: {grade}")
    
    # Identify performance regressions
    if len(analysis.slow_tests) > 0:
        print(f"⚠️ {len(analysis.slow_tests)} slow tests detected")
        for test in analysis.slow_tests[:3]:
            print(f"  - {test.name}: {test.duration:.2f}s")
```

### Security Integration

```python
def run_security_enhanced_tests():
    """Run tests with security analysis integration."""
    import subprocess
    import json
    
    # Run security scans
    subprocess.run(["bandit", "-r", "src/", "-f", "json", "-o", "security.json"])
    subprocess.run(["safety", "check", "--json", "--output", "safety.json"])
    
    # Run regular tests
    analysis = run_github_tests("tests/")
    
    # Load security results
    security_issues = []
    
    try:
        with open("security.json") as f:
            bandit_results = json.load(f)
            security_issues.extend([
                f"Security: {r['test_name']} in {r['filename']}"
                for r in bandit_results.get("results", [])
            ])
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    
    # Enhanced recommendations
    if security_issues:
        analysis.recommendations.extend([
            f"Security issues found: {len(security_issues)} items need review",
            "Run 'bandit -r src/' for detailed security analysis"
        ])
    
    return analysis
```

## Command Line Interface

```bash
# Basic usage
python -m doc_generator.github_test_runner

# Custom configuration  
python -m doc_generator.github_test_runner \
  --test-path tests/unit/ \
  --workspace /path/to/project \
  --github-token "$GITHUB_TOKEN" \
  --performance-threshold 2.0 \
  --no-coverage \
  --pr-number 123

# Help
python -m doc_generator.github_test_runner --help
```

### CLI Arguments

- `--test-path`: Test directory path (default: "tests/")
- `--workspace`: Workspace root path (default: ".")  
- `--github-token`: GitHub API token for enhanced features
- `--performance-threshold`: Slow test threshold in seconds (default: 5.0)
- `--no-coverage`: Disable coverage collection
- `--no-save`: Don't save results to file
- `--no-summary`: Don't post GitHub Actions summary
- `--pr-number`: PR number for posting comments

## Best Practices

### 1. Quality Gates
```python
# Implement consistent quality standards
quality_criteria = {
    "min_success_rate": 95.0,
    "max_failed_tests": 0,
    "min_coverage": 80.0,
    "max_slow_tests": 5,
    "max_avg_duration": 2.0
}
```

### 2. Performance Monitoring
```python
# Track performance trends
def track_performance_trends(current_analysis, historical_data):
    current_avg = current_analysis.avg_test_duration
    previous_avg = historical_data.get("avg_duration", current_avg)
    
    if current_avg > previous_avg * 1.2:  # 20% slower
        return "PERFORMANCE_REGRESSION"
    elif current_avg < previous_avg * 0.8:  # 20% faster
        return "PERFORMANCE_IMPROVEMENT"
    else:
        return "PERFORMANCE_STABLE"
```

### 3. Failure Pattern Analysis
```python
# Identify recurring failure patterns
def analyze_failure_patterns(analysis_history):
    recurring_failures = {}
    
    for analysis in analysis_history:
        for category, failures in analysis.failure_categories.items():
            for failure in failures:
                key = f"{failure.name}:{category}"
                recurring_failures[key] = recurring_failures.get(key, 0) + 1
    
    # Identify flaky tests (failing >25% of time)
    flaky_candidates = {
        test: count for test, count in recurring_failures.items()
        if count > len(analysis_history) * 0.25
    }
    
    return flaky_candidates
```

### 4. CI/CD Integration
```yaml
# Comprehensive workflow example
name: Test Quality Pipeline

on: [push, pull_request]

jobs:
  test-analysis:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        test-type: [unit, integration, e2e]
    
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      
    - name: Run ${{ matrix.test-type }} tests
      run: |
        python -m doc_generator.github_test_runner \
          --test-path tests/${{ matrix.test-type }}/ \
          --performance-threshold ${{ matrix.test-type == 'e2e' && '30.0' || '3.0' }}
  
  quality-gate:
    needs: test-analysis
    runs-on: ubuntu-latest
    if: always()
    
    steps:
    - name: Quality Gate Check
      run: |
        # Download and analyze all test results
        # Implement quality gate logic
        # Fail pipeline if gates not met
```

## Troubleshooting

### Common Issues

**1. Framework Not Detected**
```bash
# Force framework detection
export TEST_FRAMEWORK=pytest
python -m doc_generator.github_test_runner
```

**2. Coverage Not Collected**
```bash
# Ensure coverage is installed
pip install coverage pytest-cov

# Check coverage configuration
cat .coveragerc  # or pyproject.toml
```

**3. Performance Threshold Too Strict**
```python
# Adjust threshold based on environment
threshold = 10.0 if os.getenv("CI") else 3.0
analysis = run_github_tests(performance_threshold=threshold)
```

**4. GitHub Token Issues**
```bash
# Verify token permissions
curl -H "Authorization: token $GITHUB_TOKEN" \
     https://api.github.com/user
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
analysis = run_github_tests(
    test_path="tests/",
    save_results=True  # Check test-analysis.json for details
)
```

## Contributing

To contribute improvements to the GitHub Test Runner:

1. **Add New Framework Support**
   - Implement `_run_<framework>` method
   - Add framework detection logic
   - Create result parser for framework output

2. **Enhance Analysis**
   - Add new failure categorization patterns
   - Implement additional performance metrics
   - Create custom recommendation engines

3. **Extend Integrations**
   - Add new CI/CD platform support
   - Implement additional notification channels
   - Create custom reporting formats

---

**Made with ❤️ for better testing workflows**