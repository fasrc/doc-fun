#!/usr/bin/env python3
"""
Example usage of the GitHub Test Runner and Analysis function.

This script demonstrates various ways to use the GitHubTestRunner
for different testing scenarios.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from doc_generator.github_test_runner import (
    run_github_tests,
    GitHubTestRunner,
    TestFramework,
    TestStatus
)


def example_basic_usage():
    """Basic usage example - run tests and get analysis."""
    print("🔍 Running basic test analysis...")
    
    # Simple one-function call
    analysis = run_github_tests(
        test_path="tests/",
        enable_coverage=True,
        performance_threshold=3.0,
        save_results=True,
        post_summary=True
    )
    
    # Access results
    suite = analysis.suite_results
    print(f"✅ Passed: {suite.passed}")
    print(f"❌ Failed: {suite.failed}")
    print(f"📊 Success Rate: {suite.success_rate:.1f}%")
    
    if suite.coverage_percent:
        print(f"📈 Coverage: {suite.coverage_percent:.1f}%")
    
    # Show recommendations
    if analysis.recommendations:
        print("\n💡 Recommendations:")
        for rec in analysis.recommendations:
            print(f"  - {rec}")


def example_custom_analysis():
    """Example with custom runner configuration."""
    print("\n🔧 Running custom test analysis...")
    
    # Create custom runner
    runner = GitHubTestRunner(
        workspace_path=".",
        github_token=os.getenv("GITHUB_TOKEN"),  # Optional
        enable_coverage=True,
        performance_threshold=2.0  # Stricter performance threshold
    )
    
    # Detect framework
    framework = runner.detect_test_framework()
    print(f"📋 Detected framework: {framework.value}")
    
    # Run tests with custom args
    suite_result = runner.run_tests(
        test_path="tests/",
        extra_args=["--maxfail=5", "--tb=short"]  # Stop after 5 failures
    )
    
    # Perform analysis
    analysis = runner.analyze_results(suite_result)
    
    # Custom reporting
    print(f"\n📊 Test Results:")
    print(f"  Framework: {suite_result.framework.value}")
    print(f"  Total: {suite_result.total_tests}")
    print(f"  Duration: {suite_result.duration:.2f}s")
    print(f"  Avg per test: {analysis.avg_test_duration:.3f}s")
    
    # Show slow tests
    if analysis.slow_tests:
        print(f"\n⏳ Slow tests ({len(analysis.slow_tests)}):")
        for test in analysis.slow_tests[:3]:  # Show top 3
            print(f"  - {test.name}: {test.duration:.2f}s")
    
    # Show failure categories
    if analysis.failure_categories:
        print(f"\n🔍 Failure analysis:")
        for category, failures in analysis.failure_categories.items():
            print(f"  - {category}: {len(failures)} failures")


def example_pr_integration():
    """Example for PR integration with comments."""
    print("\n📝 Running PR integration example...")
    
    # Check if we're in a PR context
    pr_number = os.getenv("PR_NUMBER")  # You'd set this in your CI
    github_token = os.getenv("GITHUB_TOKEN")
    
    if not pr_number or not github_token:
        print("ℹ️  Skipping PR integration (no PR number or GitHub token)")
        return
    
    # Run analysis
    runner = GitHubTestRunner(
        workspace_path=".",
        github_token=github_token,
        enable_coverage=True
    )
    
    suite_result = runner.run_tests("tests/")
    analysis = runner.analyze_results(suite_result)
    
    # Post comment to PR
    success = runner.post_pr_comment(analysis, int(pr_number))
    
    if success:
        print(f"✅ Posted analysis comment to PR #{pr_number}")
    else:
        print(f"❌ Failed to post comment to PR #{pr_number}")


def example_multi_framework():
    """Example of handling multiple test frameworks."""
    print("\n🎯 Multi-framework test analysis...")
    
    frameworks_to_test = [
        ("tests/unit/", "Unit Tests"),
        ("tests/integration/", "Integration Tests"),  
        ("frontend/tests/", "Frontend Tests"),  # Might be Jest
    ]
    
    all_results = []
    
    for test_path, description in frameworks_to_test:
        if not Path(test_path).exists():
            print(f"⏭️  Skipping {description} - path {test_path} not found")
            continue
        
        print(f"\n🧪 Running {description}...")
        
        try:
            analysis = run_github_tests(
                test_path=test_path,
                save_results=False,  # Don't overwrite results
                post_summary=False   # We'll create custom summary
            )
            
            all_results.append((description, analysis))
            
            suite = analysis.suite_results
            print(f"  ✅ {suite.passed} passed, ❌ {suite.failed} failed")
            print(f"  ⏱️  Duration: {suite.duration:.2f}s")
            
        except Exception as e:
            print(f"  ❌ Error running {description}: {e}")
    
    # Create combined summary
    if all_results:
        print(f"\n📋 Combined Summary:")
        total_tests = sum(analysis.suite_results.total_tests for _, analysis in all_results)
        total_passed = sum(analysis.suite_results.passed for _, analysis in all_results)
        total_failed = sum(analysis.suite_results.failed for _, analysis in all_results)
        
        print(f"  📊 Total: {total_tests} tests")
        print(f"  ✅ Passed: {total_passed}")
        print(f"  ❌ Failed: {total_failed}")
        print(f"  📈 Overall Success: {(total_passed/total_tests*100):.1f}%")


def example_performance_analysis():
    """Example focused on performance analysis."""
    print("\n⚡ Performance-focused test analysis...")
    
    # Run with strict performance threshold
    analysis = run_github_tests(
        test_path="tests/",
        performance_threshold=1.0,  # 1 second threshold
        enable_coverage=False,      # Skip coverage for performance focus
        save_results=True
    )
    
    print(f"📈 Performance Analysis:")
    print(f"  Average test duration: {analysis.avg_test_duration:.3f}s")
    
    if analysis.slowest_test:
        print(f"  Slowest test: {analysis.slowest_test.name} ({analysis.slowest_test.duration:.2f}s)")
    
    if analysis.fastest_test:
        print(f"  Fastest test: {analysis.fastest_test.name} ({analysis.fastest_test.duration:.3f}s)")
    
    # Performance recommendations
    slow_count = len(analysis.slow_tests)
    if slow_count > 0:
        print(f"\n⚠️  Found {slow_count} slow tests:")
        for test in analysis.slow_tests[:5]:  # Show top 5
            print(f"    - {test.name}: {test.duration:.2f}s")
        
        if slow_count > 5:
            print(f"    ... and {slow_count - 5} more")
    
    # Performance grade
    if analysis.avg_test_duration < 0.1:
        grade = "A+"
    elif analysis.avg_test_duration < 0.5:
        grade = "A"
    elif analysis.avg_test_duration < 1.0:
        grade = "B"
    elif analysis.avg_test_duration < 2.0:
        grade = "C"
    else:
        grade = "D"
    
    print(f"\n🎯 Performance Grade: {grade}")


def example_quality_gate():
    """Example of implementing quality gates."""
    print("\n🚦 Quality gate example...")
    
    # Define quality gate criteria
    quality_gates = {
        "min_success_rate": 95.0,      # 95% tests must pass
        "max_failed_tests": 0,         # No failed tests allowed
        "min_coverage": 80.0,          # 80% minimum coverage
        "max_slow_tests": 5,           # Max 5 slow tests allowed
        "max_avg_duration": 2.0        # Average test should be < 2s
    }
    
    # Run analysis
    analysis = run_github_tests(
        test_path="tests/",
        performance_threshold=3.0,
        save_results=True
    )
    
    suite = analysis.suite_results
    
    # Check each quality gate
    gates_passed = []
    gates_failed = []
    
    # Success rate gate
    if suite.success_rate >= quality_gates["min_success_rate"]:
        gates_passed.append(f"✅ Success rate: {suite.success_rate:.1f}% >= {quality_gates['min_success_rate']}%")
    else:
        gates_failed.append(f"❌ Success rate: {suite.success_rate:.1f}% < {quality_gates['min_success_rate']}%")
    
    # Failed tests gate
    if suite.failed <= quality_gates["max_failed_tests"]:
        gates_passed.append(f"✅ Failed tests: {suite.failed} <= {quality_gates['max_failed_tests']}")
    else:
        gates_failed.append(f"❌ Failed tests: {suite.failed} > {quality_gates['max_failed_tests']}")
    
    # Coverage gate
    if suite.coverage_percent and suite.coverage_percent >= quality_gates["min_coverage"]:
        gates_passed.append(f"✅ Coverage: {suite.coverage_percent:.1f}% >= {quality_gates['min_coverage']}%")
    elif suite.coverage_percent:
        gates_failed.append(f"❌ Coverage: {suite.coverage_percent:.1f}% < {quality_gates['min_coverage']}%")
    else:
        gates_failed.append("⚠️  Coverage: Not measured")
    
    # Performance gates
    slow_test_count = len(analysis.slow_tests)
    if slow_test_count <= quality_gates["max_slow_tests"]:
        gates_passed.append(f"✅ Slow tests: {slow_test_count} <= {quality_gates['max_slow_tests']}")
    else:
        gates_failed.append(f"❌ Slow tests: {slow_test_count} > {quality_gates['max_slow_tests']}")
    
    if analysis.avg_test_duration <= quality_gates["max_avg_duration"]:
        gates_passed.append(f"✅ Avg duration: {analysis.avg_test_duration:.2f}s <= {quality_gates['max_avg_duration']}s")
    else:
        gates_failed.append(f"❌ Avg duration: {analysis.avg_test_duration:.2f}s > {quality_gates['max_avg_duration']}s")
    
    # Report results
    print(f"\n📊 Quality Gates Report:")
    
    if gates_passed:
        print(f"\n✅ Passed Gates ({len(gates_passed)}):")
        for gate in gates_passed:
            print(f"  {gate}")
    
    if gates_failed:
        print(f"\n❌ Failed Gates ({len(gates_failed)}):")
        for gate in gates_failed:
            print(f"  {gate}")
    
    # Overall result
    all_passed = len(gates_failed) == 0
    print(f"\n🎯 Overall Result: {'✅ ALL GATES PASSED' if all_passed else '❌ QUALITY GATES FAILED'}")
    
    return all_passed


def main():
    """Run all examples."""
    print("🧪 GitHub Test Runner Examples")
    print("=" * 50)
    
    # Check if we're in a test environment
    if not Path("tests").exists():
        print("⚠️  No tests directory found. Creating minimal example...")
        Path("tests").mkdir(exist_ok=True)
        
        # Create a simple test file
        test_content = '''
import unittest

class TestExample(unittest.TestCase):
    def test_passing(self):
        self.assertEqual(1 + 1, 2)
    
    def test_slow(self):
        import time
        time.sleep(2)  # Simulate slow test
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
'''
        with open("tests/test_example.py", "w") as f:
            f.write(test_content)
    
    try:
        # Run examples
        example_basic_usage()
        example_custom_analysis()
        example_pr_integration()
        example_multi_framework()
        example_performance_analysis()
        
        # Quality gate example (returns boolean)
        quality_passed = example_quality_gate()
        
        print(f"\n🎉 Examples completed!")
        print(f"Final Quality Status: {'✅ PASSED' if quality_passed else '❌ FAILED'}")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())