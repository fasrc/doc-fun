# doc-fun

Jupyter Notebook that Generates User Docs style documentation on your topic of choise, for fun.
* For generating user documentation with LLM-based workflows
* Few-Shot workflow imports existing documentation for reference
* Creates a number of new versions of topical documentation
* Weighted comparison of each section ( Intro, Usage, Examples, References, etc ) against each other
* Chooses most highly scored version of each section for our goal ( the Few-Shot examples, arbitrary scale, and other weights )
* Assembles final document with references to original generated versions and resulting weights
