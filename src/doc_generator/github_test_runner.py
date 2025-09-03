"""
GitHub Test Runner and Analysis Module

This module provides comprehensive test execution and analysis capabilities
for GitHub Actions workflows, including test result parsing, failure analysis,
coverage reporting, and performance monitoring.
"""

import json
import time
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import xml.etree.ElementTree as ET
import re
import requests
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test execution status enumeration."""
    PASSED = "passed"
    FAILED = "failed" 
    SKIPPED = "skipped"
    ERROR = "error"
    XFAIL = "xfail"
    XPASS = "xpass"


class TestFramework(Enum):
    """Supported test frameworks."""
    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    MOCHA = "mocha"
    GO_TEST = "go_test"
    CARGO_TEST = "cargo_test"
    MAVEN = "maven"
    GRADLE = "gradle"


@dataclass
class TestResult:
    """Individual test result data."""
    name: str
    status: TestStatus
    duration: float
    file_path: str
    line_number: Optional[int] = None
    class_name: Optional[str] = None
    error_message: Optional[str] = None
    failure_message: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


@dataclass
class TestSuiteResult:
    """Test suite execution results."""
    framework: TestFramework
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    coverage_percent: Optional[float] = None
    tests: List[TestResult] = None
    
    def __post_init__(self):
        if self.tests is None:
            self.tests = []
    
    @property
    def success_rate(self) -> float:
        """Calculate test success rate."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100.0
    
    @property
    def is_healthy(self) -> bool:
        """Determine if test suite is in healthy state."""
        return self.failed == 0 and self.errors == 0


@dataclass
class GitHubContext:
    """GitHub Actions context information."""
    repository: str
    ref: str
    sha: str
    actor: str
    workflow: str
    run_id: str
    run_number: int
    job: str
    action: str
    event_name: str
    workspace: str
    token: Optional[str] = None


@dataclass
class TestAnalysis:
    """Comprehensive test analysis results."""
    suite_results: TestSuiteResult
    github_context: GitHubContext
    analysis_timestamp: datetime
    
    # Failure Analysis
    failure_categories: Dict[str, List[TestResult]]
    flaky_tests: List[TestResult]
    slow_tests: List[TestResult]
    
    # Performance Metrics  
    avg_test_duration: float
    slowest_test: Optional[TestResult]
    fastest_test: Optional[TestResult]
    
    # Trends (if historical data available)
    duration_trend: Optional[str]  # "improving", "degrading", "stable"
    success_rate_trend: Optional[str]
    
    # Recommendations
    recommendations: List[str]
    critical_issues: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary for serialization."""
        return {
            "suite_results": asdict(self.suite_results),
            "github_context": asdict(self.github_context),
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "failure_categories": {
                category: [asdict(test) for test in tests]
                for category, tests in self.failure_categories.items()
            },
            "flaky_tests": [asdict(test) for test in self.flaky_tests],
            "slow_tests": [asdict(test) for test in self.slow_tests],
            "avg_test_duration": self.avg_test_duration,
            "slowest_test": asdict(self.slowest_test) if self.slowest_test else None,
            "fastest_test": asdict(self.fastest_test) if self.fastest_test else None,
            "duration_trend": self.duration_trend,
            "success_rate_trend": self.success_rate_trend,
            "recommendations": self.recommendations,
            "critical_issues": self.critical_issues
        }


class GitHubTestRunner:
    """Main class for running and analyzing tests in GitHub Actions."""
    
    def __init__(self, 
                 workspace_path: str = ".",
                 github_token: Optional[str] = None,
                 enable_coverage: bool = True,
                 performance_threshold: float = 5.0):
        """
        Initialize GitHub Test Runner.
        
        Args:
            workspace_path: Path to the workspace root
            github_token: GitHub API token for enhanced functionality
            enable_coverage: Whether to collect coverage data
            performance_threshold: Threshold for identifying slow tests (seconds)
        """
        self.workspace_path = Path(workspace_path)
        self.github_token = github_token
        self.enable_coverage = enable_coverage
        self.performance_threshold = performance_threshold
        self.github_context = self._get_github_context()
        
    def _get_github_context(self) -> GitHubContext:
        """Extract GitHub Actions context from environment."""
        import os
        
        return GitHubContext(
            repository=os.getenv("GITHUB_REPOSITORY", "unknown/unknown"),
            ref=os.getenv("GITHUB_REF", "unknown"),
            sha=os.getenv("GITHUB_SHA", "unknown"),
            actor=os.getenv("GITHUB_ACTOR", "unknown"),
            workflow=os.getenv("GITHUB_WORKFLOW", "unknown"),
            run_id=os.getenv("GITHUB_RUN_ID", "0"),
            run_number=int(os.getenv("GITHUB_RUN_NUMBER", "0")),
            job=os.getenv("GITHUB_JOB", "unknown"),
            action=os.getenv("GITHUB_ACTION", "unknown"),
            event_name=os.getenv("GITHUB_EVENT_NAME", "unknown"),
            workspace=os.getenv("GITHUB_WORKSPACE", str(self.workspace_path)),
            token=self.github_token
        )
    
    def detect_test_framework(self) -> TestFramework:
        """Auto-detect the test framework being used."""
        # Check for Python frameworks
        if (self.workspace_path / "pytest.ini").exists() or \
           (self.workspace_path / "pyproject.toml").exists():
            return TestFramework.PYTEST
        
        if any(self.workspace_path.glob("**/test_*.py")):
            return TestFramework.UNITTEST
            
        # Check for JavaScript frameworks
        package_json = self.workspace_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    data = json.load(f)
                    deps = {**data.get("dependencies", {}), 
                           **data.get("devDependencies", {})}
                    if "jest" in deps:
                        return TestFramework.JEST
                    if "mocha" in deps:
                        return TestFramework.MOCHA
            except (json.JSONDecodeError, IOError):
                pass
        
        # Check for Go
        if any(self.workspace_path.glob("**/*_test.go")):
            return TestFramework.GO_TEST
            
        # Check for Rust
        if (self.workspace_path / "Cargo.toml").exists():
            return TestFramework.CARGO_TEST
            
        # Check for Java
        if (self.workspace_path / "pom.xml").exists():
            return TestFramework.MAVEN
        if (self.workspace_path / "build.gradle").exists() or \
           (self.workspace_path / "build.gradle.kts").exists():
            return TestFramework.GRADLE
            
        # Default to pytest for Python projects
        return TestFramework.PYTEST
    
    def run_tests(self, 
                  test_path: str = "tests/",
                  extra_args: List[str] = None) -> TestSuiteResult:
        """
        Execute tests and collect results.
        
        Args:
            test_path: Path to test directory or specific test file
            extra_args: Additional arguments to pass to test runner
            
        Returns:
            TestSuiteResult containing execution results
        """
        framework = self.detect_test_framework()
        extra_args = extra_args or []
        
        logger.info(f"Running tests with {framework.value} framework")
        
        if framework == TestFramework.PYTEST:
            return self._run_pytest(test_path, extra_args)
        elif framework == TestFramework.UNITTEST:
            return self._run_unittest(test_path, extra_args)
        elif framework == TestFramework.JEST:
            return self._run_jest(test_path, extra_args)
        elif framework == TestFramework.GO_TEST:
            return self._run_go_test(test_path, extra_args)
        elif framework == TestFramework.CARGO_TEST:
            return self._run_cargo_test(test_path, extra_args)
        else:
            raise ValueError(f"Unsupported test framework: {framework}")
    
    def _run_pytest(self, test_path: str, extra_args: List[str]) -> TestSuiteResult:
        """Run pytest and parse results."""
        cmd = ["python", "-m", "pytest", test_path, "--junitxml=test-results.xml"]
        
        if self.enable_coverage:
            cmd.extend(["--cov=src", "--cov-report=xml"])
        
        cmd.extend(extra_args)
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_path)
        duration = time.time() - start_time
        
        # Parse JUnit XML results
        tests = self._parse_junit_xml("test-results.xml")
        
        # Get coverage if enabled
        coverage_percent = None
        if self.enable_coverage:
            coverage_percent = self._parse_coverage_xml("coverage.xml")
        
        # Count results
        passed = sum(1 for t in tests if t.status == TestStatus.PASSED)
        failed = sum(1 for t in tests if t.status == TestStatus.FAILED)
        skipped = sum(1 for t in tests if t.status == TestStatus.SKIPPED)
        errors = sum(1 for t in tests if t.status == TestStatus.ERROR)
        
        return TestSuiteResult(
            framework=TestFramework.PYTEST,
            total_tests=len(tests),
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration=duration,
            coverage_percent=coverage_percent,
            tests=tests
        )
    
    def _run_unittest(self, test_path: str, extra_args: List[str]) -> TestSuiteResult:
        """Run unittest and parse results."""
        cmd = ["python", "-m", "unittest", "discover", "-s", test_path, "-v"]
        cmd.extend(extra_args)
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_path)
        duration = time.time() - start_time
        
        # Parse unittest output (simpler than JUnit XML)
        tests = self._parse_unittest_output(result.stdout, result.stderr)
        
        passed = sum(1 for t in tests if t.status == TestStatus.PASSED)
        failed = sum(1 for t in tests if t.status == TestStatus.FAILED)
        skipped = sum(1 for t in tests if t.status == TestStatus.SKIPPED)
        errors = sum(1 for t in tests if t.status == TestStatus.ERROR)
        
        return TestSuiteResult(
            framework=TestFramework.UNITTEST,
            total_tests=len(tests),
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration=duration,
            tests=tests
        )
    
    def _run_jest(self, test_path: str, extra_args: List[str]) -> TestSuiteResult:
        """Run Jest and parse results."""
        cmd = ["npm", "test", "--", "--json", "--outputFile=jest-results.json"]
        cmd.extend(extra_args)
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_path)
        duration = time.time() - start_time
        
        # Parse Jest JSON results
        tests = self._parse_jest_json("jest-results.json")
        
        passed = sum(1 for t in tests if t.status == TestStatus.PASSED)
        failed = sum(1 for t in tests if t.status == TestStatus.FAILED)
        skipped = sum(1 for t in tests if t.status == TestStatus.SKIPPED)
        errors = sum(1 for t in tests if t.status == TestStatus.ERROR)
        
        return TestSuiteResult(
            framework=TestFramework.JEST,
            total_tests=len(tests),
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration=duration,
            tests=tests
        )
    
    def _run_go_test(self, test_path: str, extra_args: List[str]) -> TestSuiteResult:
        """Run Go tests and parse results."""
        cmd = ["go", "test", "-json", test_path]
        cmd.extend(extra_args)
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_path)
        duration = time.time() - start_time
        
        # Parse Go test JSON output
        tests = self._parse_go_test_output(result.stdout)
        
        passed = sum(1 for t in tests if t.status == TestStatus.PASSED)
        failed = sum(1 for t in tests if t.status == TestStatus.FAILED)
        skipped = sum(1 for t in tests if t.status == TestStatus.SKIPPED)
        errors = sum(1 for t in tests if t.status == TestStatus.ERROR)
        
        return TestSuiteResult(
            framework=TestFramework.GO_TEST,
            total_tests=len(tests),
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration=duration,
            tests=tests
        )
    
    def _run_cargo_test(self, test_path: str, extra_args: List[str]) -> TestSuiteResult:
        """Run Rust/Cargo tests and parse results."""
        cmd = ["cargo", "test", "--", "--format", "json"]
        cmd.extend(extra_args)
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_path)
        duration = time.time() - start_time
        
        # Parse Cargo test output
        tests = self._parse_cargo_test_output(result.stdout)
        
        passed = sum(1 for t in tests if t.status == TestStatus.PASSED)
        failed = sum(1 for t in tests if t.status == TestStatus.FAILED)
        skipped = sum(1 for t in tests if t.status == TestStatus.SKIPPED)
        errors = sum(1 for t in tests if t.status == TestStatus.ERROR)
        
        return TestSuiteResult(
            framework=TestFramework.CARGO_TEST,
            total_tests=len(tests),
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration=duration,
            tests=tests
        )
    
    def _parse_junit_xml(self, xml_file: str) -> List[TestResult]:
        """Parse JUnit XML test results."""
        xml_path = self.workspace_path / xml_file
        if not xml_path.exists():
            return []
        
        tests = []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for testcase in root.findall(".//testcase"):
                name = testcase.get("name", "unknown")
                class_name = testcase.get("classname", "")
                duration = float(testcase.get("time", 0))
                file_path = testcase.get("file", "")
                line_number = testcase.get("line")
                
                # Determine status
                if testcase.find("failure") is not None:
                    status = TestStatus.FAILED
                    failure_elem = testcase.find("failure")
                    failure_message = failure_elem.text if failure_elem is not None else None
                    error_message = None
                elif testcase.find("error") is not None:
                    status = TestStatus.ERROR
                    error_elem = testcase.find("error")
                    error_message = error_elem.text if error_elem is not None else None
                    failure_message = None
                elif testcase.find("skipped") is not None:
                    status = TestStatus.SKIPPED
                    error_message = None
                    failure_message = None
                else:
                    status = TestStatus.PASSED
                    error_message = None
                    failure_message = None
                
                tests.append(TestResult(
                    name=name,
                    status=status,
                    duration=duration,
                    file_path=file_path,
                    line_number=int(line_number) if line_number else None,
                    class_name=class_name,
                    error_message=error_message,
                    failure_message=failure_message
                ))
                
        except ET.ParseError as e:
            logger.error(f"Failed to parse JUnit XML: {e}")
        
        return tests
    
    def _parse_coverage_xml(self, xml_file: str) -> Optional[float]:
        """Parse coverage XML and extract coverage percentage."""
        xml_path = self.workspace_path / xml_file
        if not xml_path.exists():
            return None
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Look for coverage attribute in root or coverage element
            coverage_elem = root.find(".//coverage")
            if coverage_elem is not None:
                line_rate = coverage_elem.get("line-rate")
                if line_rate:
                    return float(line_rate) * 100
            
            # Alternative: look for line-rate in root
            line_rate = root.get("line-rate")
            if line_rate:
                return float(line_rate) * 100
                
        except (ET.ParseError, ValueError) as e:
            logger.error(f"Failed to parse coverage XML: {e}")
        
        return None
    
    def _parse_unittest_output(self, stdout: str, stderr: str) -> List[TestResult]:
        """Parse unittest text output."""
        tests = []
        lines = stdout.split('\n') + stderr.split('\n')
        
        # Simple regex patterns for unittest output
        test_pattern = re.compile(r'^(test_\w+)\s+\((\w+\.\w+)\)\s+\.\.\.\s+(ok|FAIL|ERROR|SKIP)')
        
        for line in lines:
            match = test_pattern.match(line.strip())
            if match:
                test_name, class_name, result = match.groups()
                
                if result == "ok":
                    status = TestStatus.PASSED
                elif result == "FAIL":
                    status = TestStatus.FAILED
                elif result == "ERROR":
                    status = TestStatus.ERROR
                else:
                    status = TestStatus.SKIPPED
                
                tests.append(TestResult(
                    name=test_name,
                    status=status,
                    duration=0.0,  # unittest doesn't provide timing by default
                    file_path="",
                    class_name=class_name
                ))
        
        return tests
    
    def _parse_jest_json(self, json_file: str) -> List[TestResult]:
        """Parse Jest JSON results."""
        json_path = self.workspace_path / json_file
        if not json_path.exists():
            return []
        
        tests = []
        try:
            with open(json_path) as f:
                data = json.load(f)
            
            for test_file in data.get("testResults", []):
                file_path = test_file.get("name", "")
                
                for test_case in test_file.get("assertionResults", []):
                    name = test_case.get("title", "unknown")
                    duration = test_case.get("duration", 0) / 1000.0  # Convert ms to seconds
                    
                    status_str = test_case.get("status", "unknown")
                    if status_str == "passed":
                        status = TestStatus.PASSED
                    elif status_str == "failed":
                        status = TestStatus.FAILED
                    elif status_str == "skipped":
                        status = TestStatus.SKIPPED
                    else:
                        status = TestStatus.ERROR
                    
                    failure_message = None
                    if test_case.get("failureMessages"):
                        failure_message = "\n".join(test_case["failureMessages"])
                    
                    tests.append(TestResult(
                        name=name,
                        status=status,
                        duration=duration,
                        file_path=file_path,
                        failure_message=failure_message
                    ))
        
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to parse Jest JSON: {e}")
        
        return tests
    
    def _parse_go_test_output(self, output: str) -> List[TestResult]:
        """Parse Go test JSON output."""
        tests = []
        
        for line in output.strip().split('\n'):
            try:
                data = json.loads(line)
                if data.get("Action") == "pass" or data.get("Action") == "fail":
                    test_name = data.get("Test", "")
                    if test_name:  # Skip package-level results
                        duration = data.get("Elapsed", 0)
                        
                        if data.get("Action") == "pass":
                            status = TestStatus.PASSED
                        else:
                            status = TestStatus.FAILED
                        
                        tests.append(TestResult(
                            name=test_name,
                            status=status,
                            duration=duration,
                            file_path=data.get("Package", "")
                        ))
            except json.JSONDecodeError:
                continue  # Skip non-JSON lines
        
        return tests
    
    def _parse_cargo_test_output(self, output: str) -> List[TestResult]:
        """Parse Cargo test output."""
        tests = []
        
        # Cargo test doesn't have JSON output by default, parse text output
        lines = output.split('\n')
        test_pattern = re.compile(r'^test\s+(\S+)\s+\.\.\.\s+(ok|FAILED|ignored)')
        
        for line in lines:
            match = test_pattern.match(line.strip())
            if match:
                test_name, result = match.groups()
                
                if result == "ok":
                    status = TestStatus.PASSED
                elif result == "FAILED":
                    status = TestStatus.FAILED
                else:
                    status = TestStatus.SKIPPED
                
                tests.append(TestResult(
                    name=test_name,
                    status=status,
                    duration=0.0,  # Cargo doesn't provide individual test timing
                    file_path=""
                ))
        
        return tests
    
    def analyze_results(self, suite_result: TestSuiteResult) -> TestAnalysis:
        """
        Perform comprehensive analysis of test results.
        
        Args:
            suite_result: Test suite results to analyze
            
        Returns:
            TestAnalysis with detailed insights and recommendations
        """
        logger.info("Analyzing test results...")
        
        # Categorize failures
        failure_categories = self._categorize_failures(suite_result.tests)
        
        # Identify patterns
        flaky_tests = self._identify_flaky_tests(suite_result.tests)
        slow_tests = self._identify_slow_tests(suite_result.tests)
        
        # Performance analysis
        test_durations = [t.duration for t in suite_result.tests if t.duration > 0]
        avg_duration = sum(test_durations) / len(test_durations) if test_durations else 0
        slowest_test = max(suite_result.tests, key=lambda t: t.duration, default=None)
        fastest_test = min((t for t in suite_result.tests if t.duration > 0), 
                          key=lambda t: t.duration, default=None)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(suite_result, failure_categories, slow_tests)
        critical_issues = self._identify_critical_issues(suite_result, failure_categories)
        
        return TestAnalysis(
            suite_results=suite_result,
            github_context=self.github_context,
            analysis_timestamp=datetime.now(),
            failure_categories=failure_categories,
            flaky_tests=flaky_tests,
            slow_tests=slow_tests,
            avg_test_duration=avg_duration,
            slowest_test=slowest_test,
            fastest_test=fastest_test,
            duration_trend=None,  # Would need historical data
            success_rate_trend=None,  # Would need historical data
            recommendations=recommendations,
            critical_issues=critical_issues
        )
    
    def _categorize_failures(self, tests: List[TestResult]) -> Dict[str, List[TestResult]]:
        """Categorize test failures by type."""
        categories = {
            "Import Errors": [],
            "Assertion Failures": [],
            "Type Errors": [],
            "Timeout/Performance": [],
            "Configuration Issues": [],
            "Unknown Errors": []
        }
        
        failed_tests = [t for t in tests if t.status in [TestStatus.FAILED, TestStatus.ERROR]]
        
        for test in failed_tests:
            error_text = test.error_message or test.failure_message or ""
            error_text = error_text.lower()
            
            if any(keyword in error_text for keyword in ["importerror", "modulenotfounderror", "import"]):
                categories["Import Errors"].append(test)
            elif any(keyword in error_text for keyword in ["assertionerror", "assert"]):
                categories["Assertion Failures"].append(test)
            elif any(keyword in error_text for keyword in ["typeerror", "type", "attribute"]):
                categories["Type Errors"].append(test)
            elif any(keyword in error_text for keyword in ["timeout", "time", "slow"]):
                categories["Timeout/Performance"].append(test)
            elif any(keyword in error_text for keyword in ["config", "setup", "fixture"]):
                categories["Configuration Issues"].append(test)
            else:
                categories["Unknown Errors"].append(test)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _identify_flaky_tests(self, tests: List[TestResult]) -> List[TestResult]:
        """Identify potentially flaky tests (requires historical data)."""
        # This is a simplified implementation
        # In practice, you'd need historical test results to identify flaky tests
        return []
    
    def _identify_slow_tests(self, tests: List[TestResult]) -> List[TestResult]:
        """Identify slow tests based on performance threshold."""
        return [t for t in tests if t.duration > self.performance_threshold]
    
    def _generate_recommendations(self, 
                                suite_result: TestSuiteResult,
                                failure_categories: Dict[str, List[TestResult]],
                                slow_tests: List[TestResult]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Coverage recommendations
        if suite_result.coverage_percent is not None:
            if suite_result.coverage_percent < 80:
                recommendations.append(
                    f"Test coverage is {suite_result.coverage_percent:.1f}%. "
                    "Consider adding tests to reach 80%+ coverage."
                )
        
        # Performance recommendations
        if slow_tests:
            recommendations.append(
                f"Found {len(slow_tests)} slow tests (>{self.performance_threshold}s). "
                "Consider optimizing or mocking external dependencies."
            )
        
        # Failure-specific recommendations
        if "Import Errors" in failure_categories:
            recommendations.append(
                "Import errors detected. Check dependencies and installation requirements."
            )
        
        if "Assertion Failures" in failure_categories:
            count = len(failure_categories["Assertion Failures"])
            recommendations.append(
                f"{count} assertion failures found. Review test expectations and implementation."
            )
        
        # Success rate recommendations
        if suite_result.success_rate < 95:
            recommendations.append(
                f"Test success rate is {suite_result.success_rate:.1f}%. "
                "Aim for 95%+ success rate for healthy test suite."
            )
        
        return recommendations
    
    def _identify_critical_issues(self,
                                suite_result: TestSuiteResult, 
                                failure_categories: Dict[str, List[TestResult]]) -> List[str]:
        """Identify critical issues that need immediate attention."""
        critical_issues = []
        
        # High failure rate
        if suite_result.success_rate < 80:
            critical_issues.append(
                f"CRITICAL: Very low success rate ({suite_result.success_rate:.1f}%). "
                "Test suite needs immediate attention."
            )
        
        # Import errors
        if "Import Errors" in failure_categories and len(failure_categories["Import Errors"]) > 5:
            critical_issues.append(
                "CRITICAL: Multiple import errors detected. "
                "Check environment setup and dependencies."
            )
        
        # Many slow tests
        if len(self._identify_slow_tests(suite_result.tests)) > suite_result.total_tests * 0.1:
            critical_issues.append(
                "CRITICAL: More than 10% of tests are slow. "
                "Performance issues may indicate architectural problems."
            )
        
        return critical_issues
    
    def generate_github_summary(self, analysis: TestAnalysis) -> str:
        """Generate GitHub Actions job summary."""
        suite = analysis.suite_results
        
        summary = f"""## 🧪 Test Results Summary

### 📊 Overall Results
- **Total Tests**: {suite.total_tests}
- **✅ Passed**: {suite.passed}
- **❌ Failed**: {suite.failed}
- **⏭️ Skipped**: {suite.skipped}
- **🚨 Errors**: {suite.errors}
- **⏱️ Duration**: {suite.duration:.2f}s
- **📈 Success Rate**: {suite.success_rate:.1f}%

"""
        
        if suite.coverage_percent:
            summary += f"- **📊 Coverage**: {suite.coverage_percent:.1f}%\n\n"
        
        # Health status
        if suite.is_healthy:
            summary += "### ✅ Status: HEALTHY\nAll tests are passing!\n\n"
        else:
            summary += f"### ❌ Status: NEEDS ATTENTION\n{suite.failed} failed, {suite.errors} errors\n\n"
        
        # Failure breakdown
        if analysis.failure_categories:
            summary += "### 🔍 Failure Analysis\n"
            for category, failures in analysis.failure_categories.items():
                summary += f"- **{category}**: {len(failures)} tests\n"
            summary += "\n"
        
        # Performance insights
        if analysis.slow_tests:
            summary += f"### ⚡ Performance\n"
            summary += f"- **Slow Tests**: {len(analysis.slow_tests)} (>{self.performance_threshold}s)\n"
            summary += f"- **Average Duration**: {analysis.avg_test_duration:.2f}s\n"
            if analysis.slowest_test:
                summary += f"- **Slowest Test**: {analysis.slowest_test.name} ({analysis.slowest_test.duration:.2f}s)\n"
            summary += "\n"
        
        # Critical issues
        if analysis.critical_issues:
            summary += "### 🚨 Critical Issues\n"
            for issue in analysis.critical_issues:
                summary += f"- {issue}\n"
            summary += "\n"
        
        # Recommendations
        if analysis.recommendations:
            summary += "### 💡 Recommendations\n"
            for rec in analysis.recommendations:
                summary += f"- {rec}\n"
            summary += "\n"
        
        summary += f"### 📋 Test Framework: {suite.framework.value}\n"
        summary += f"### 🔗 Workflow: {analysis.github_context.workflow}\n"
        summary += f"### 📅 Analysis Time: {analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        
        return summary
    
    def save_results(self, analysis: TestAnalysis, output_path: str = "test-analysis.json"):
        """Save analysis results to JSON file."""
        output_file = self.workspace_path / output_path
        
        with open(output_file, 'w') as f:
            json.dump(analysis.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Test analysis saved to {output_file}")
    
    def post_pr_comment(self, analysis: TestAnalysis, pr_number: int) -> bool:
        """Post test analysis as PR comment (requires GitHub token)."""
        if not self.github_token:
            logger.warning("GitHub token not provided, skipping PR comment")
            return False
        
        repo_parts = self.github_context.repository.split('/')
        if len(repo_parts) != 2:
            logger.error(f"Invalid repository format: {self.github_context.repository}")
            return False
        
        owner, repo = repo_parts
        
        # Generate comment body
        comment_body = self.generate_github_summary(analysis)
        
        # Post to GitHub API
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        data = {"body": comment_body}
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            logger.info(f"Posted test analysis comment to PR #{pr_number}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to post PR comment: {e}")
            return False


def run_github_tests(test_path: str = "tests/",
                    workspace_path: str = ".",
                    github_token: Optional[str] = None,
                    enable_coverage: bool = True,
                    performance_threshold: float = 5.0,
                    save_results: bool = True,
                    post_summary: bool = True) -> TestAnalysis:
    """
    Main function to run tests and analyze results in GitHub Actions.
    
    Args:
        test_path: Path to test directory
        workspace_path: Path to workspace root
        github_token: GitHub API token
        enable_coverage: Whether to collect coverage data
        performance_threshold: Threshold for slow test detection
        save_results: Whether to save analysis to JSON file
        post_summary: Whether to post GitHub Actions summary
        
    Returns:
        TestAnalysis with comprehensive results
    """
    runner = GitHubTestRunner(
        workspace_path=workspace_path,
        github_token=github_token,
        enable_coverage=enable_coverage,
        performance_threshold=performance_threshold
    )
    
    # Run tests
    logger.info("Starting test execution...")
    suite_result = runner.run_tests(test_path)
    
    # Analyze results
    logger.info("Analyzing test results...")
    analysis = runner.analyze_results(suite_result)
    
    # Save results if requested
    if save_results:
        runner.save_results(analysis)
    
    # Post GitHub Actions summary if requested
    if post_summary:
        summary = runner.generate_github_summary(analysis)
        
        # Write to GitHub Actions summary file
        import os
        summary_file = os.getenv("GITHUB_STEP_SUMMARY")
        if summary_file:
            with open(summary_file, "w") as f:
                f.write(summary)
            logger.info("Posted GitHub Actions job summary")
        else:
            # Fallback: print summary
            print("\n" + summary)
    
    return analysis


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests and analyze results in GitHub Actions")
    parser.add_argument("--test-path", default="tests/", help="Path to test directory")
    parser.add_argument("--workspace", default=".", help="Workspace root path")
    parser.add_argument("--github-token", help="GitHub API token")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage collection")
    parser.add_argument("--performance-threshold", type=float, default=5.0, 
                       help="Slow test threshold in seconds")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--no-summary", action="store_true", help="Don't post GitHub summary")
    parser.add_argument("--pr-number", type=int, help="PR number to post comment to")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run analysis
    analysis = run_github_tests(
        test_path=args.test_path,
        workspace_path=args.workspace,
        github_token=args.github_token,
        enable_coverage=not args.no_coverage,
        performance_threshold=args.performance_threshold,
        save_results=not args.no_save,
        post_summary=not args.no_summary
    )
    
    # Post PR comment if requested
    if args.pr_number and args.github_token:
        runner = GitHubTestRunner(
            workspace_path=args.workspace,
            github_token=args.github_token
        )
        runner.post_pr_comment(analysis, args.pr_number)
    
    # Exit with appropriate code
    exit_code = 0 if analysis.suite_results.is_healthy else 1
    exit(exit_code)