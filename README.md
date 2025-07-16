# doc-fun

## Overview
`doc-fun` is a Python script designed to automate the generation of HTML documentation pages from natural language prompts. It leverages few-shot prompting and structured inputs to create consistent and appropriately named documentation files, suitable for technical documentation related to Faculty Arts and Sciences Research Computing (FASRC).

## Features
- Converts natural language documentation prompts to consistent, lowercase HTML filenames.
- Enforces a standard documentation structure:
  - Description
  - Installation
  - Usage
  - Examples
  - References
- Suitable for generating reference pages for software and libraries relevant to FASRC.

## Installation
Ensure you have Python installed. You can clone and install dependencies as follows:

```bash
git clone https://github.com/<your-username>/doc-fun.git
cd doc-fun
pip install -r requirements.txt
```

## Usage
Run the script with:

```bash
python run.py
```

The script expects a YAML prompt file (`prompt.yaml`) for guidance on naming conventions and documentation structure.

## Example
Given the input prompt:

```
Create a reference page for MPI IO documentation in HTML format.
```

The script outputs the filename:

```
mpi-io.html
```

and generates a corresponding structured HTML documentation page.

## Customization
You can modify `prompt.yaml` to adjust naming conventions or documentation structures.

## License
This project is open-sourced under the MIT License.

