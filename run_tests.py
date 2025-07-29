#!/usr/bin/env python3
"""
Simple test runner script for the documentation generator.
"""

import subprocess
import sys

def main():
    """Run the test suite."""
    print("🧪 Running unit tests for doc_generator.py...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/test_doc_generator.py', 
            '-v'
        ], check=True)
        
        print("\n✅ All tests passed!")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("❌ pytest not found. Install with: pip install pytest pytest-mock")
        return 1

if __name__ == "__main__":
    sys.exit(main())