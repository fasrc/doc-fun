#!/usr/bin/env python3
"""Test the relative path formatting guidelines."""

print("🔗 RELATIVE PATH FORMATTING TEST")
print("=" * 50)

# Example directory structure for ../User_Codes/Languages/
directory_structure = """
User_Codes/Languages/
├── Julia/
│   ├── Example1/
│   │   └── basic.jl
│   ├── Example2/
│   │   ├── figure.py
│   │   └── advanced.jl
│   └── README.md
├── Python/
│   ├── Example1/
│   │   └── hello.py
│   └── Example2/
│       └── data_analysis.py
└── README.md (this is where we are)
"""

print("Directory Structure:")
print(directory_structure)

print("\n📝 CORRECT PATH FORMATTING:")
print("=" * 50)

examples = [
    {
        "description": "Directory reference",
        "old": "- **[Julia](Julia/)**: High-level expressive numerics...",
        "new": "- **[Julia](./Julia/)**: High-level expressive numerics..."
    },
    {
        "description": "File reference in description", 
        "old": "plotting examples (see src/User_Codes/Languages/Julia/Example2/figure.py)",
        "new": "plotting examples (see [figure.py](./Julia/Example2/figure.py))"
    },
    {
        "description": "Multiple file references",
        "old": "examples in Julia/Example1/basic.jl and Julia/Example2/advanced.jl",
        "new": "examples in [basic.jl](./Julia/Example1/basic.jl) and [advanced.jl](./Julia/Example2/advanced.jl)"
    },
    {
        "description": "Python file reference",
        "old": "data analysis tutorial (Python/Example2/data_analysis.py)",
        "new": "data analysis tutorial ([data_analysis.py](./Python/Example2/data_analysis.py))"
    }
]

for i, example in enumerate(examples, 1):
    print(f"\n{i}. {example['description']}:")
    print(f"   ❌ OLD: {example['old']}")
    print(f"   ✅ NEW: {example['new']}")

print(f"\n🎯 KEY PRINCIPLES:")
print("=" * 50)
print("1. All paths are relative to the README location")
print("2. Directory links: **[DirName](./DirName/)**")
print("3. File links: [filename.ext](./path/to/filename.ext)")
print("4. NO absolute paths like 'src/User_Codes/...'")
print("5. Always use markdown link format for files")
