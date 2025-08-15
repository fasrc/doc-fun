#!/usr/bin/env python3
"""Test the perfect README prompt with a sample topic."""

import yaml
from pathlib import Path

# Load the new prompt
with open('prompts/generator/perfect_readme_with_shot.yaml', 'r') as f:
    prompt_config = yaml.safe_load(f)

# Create a test context
test_context = """
Directory: Machine_Learning/
├── Neural_Networks/
│   ├── PyTorch/
│   │   ├── CNN_Classification/
│   │   ├── RNN_Text_Generation/
│   │   └── Transfer_Learning/
│   ├── TensorFlow/
│   │   ├── Image_Recognition/
│   │   ├── Time_Series/
│   │   └── Model_Deployment/
│   └── JAX/
│       ├── Optimization/
│       └── Neural_ODE/
├── Classical_ML/
│   ├── Scikit_Learn/
│   │   ├── Classification/
│   │   ├── Regression/
│   │   └── Clustering/
│   ├── XGBoost/
│   │   └── Gradient_Boosting/
│   └── LightGBM/
│       └── Fast_Training/
├── Data_Processing/
│   ├── Pandas_Large_Data/
│   ├── Dask_Distributed/
│   └── Ray_Parallel/
└── MLOps/
    ├── Model_Versioning/
    ├── Experiment_Tracking/
    └── Pipeline_Orchestration/
"""

# Fill in the placeholders
filled_prompt = prompt_config['user_prompt'].format(
    topic="Machine Learning",
    directory_path="/home/user/Machine_Learning",
    directory_context=test_context,
    organization="Faculty Arts and Sciences Research Computing (FASRC)",
    docs_url="https://docs.rc.fas.harvard.edu/"
)

print("=" * 80)
print("SYSTEM PROMPT:")
print("=" * 80)
print(prompt_config['system_prompt'])
print("\n" + "=" * 80)
print("USER PROMPT (with test data):")
print("=" * 80)
print(filled_prompt[:1500] + "...")  # Show first part
print("\n" + "=" * 80)
print("This prompt should generate a README similar in quality to parallel_computing.md")
print("Key features to look for:")
print("- Concise opening sentence about what users can DO")
print("- Logical grouping by paradigm (Deep Learning, Classical ML, Infrastructure)")
print("- Brief, meaningful descriptions (not 'examples of X')")
print("- Practical Quick Reference table")
print("- ~50-60 lines total")
