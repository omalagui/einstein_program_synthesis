# einstein_program_synthesis
This repository contains code for reconstructing Einstein's conceptual path to the relativity of time using program synthesis

## Setup

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Code Structure

### Notebooks

Interactive demos and experiments: `notebooks/`.

Runs all search methods and aggregates results: `all_search_methods_demo.ipynb`.

Enumeration search: `enumeration_search_demo.ipynb`.

Pure bayesian search: `pure_bayes_search_demo.ipynb`.

Bayes-neural search: `bayes_neural_search_demo.ipynb`.

Runs assumption-conclusion switch discovery mechanism step by step: `assumption_conclusion_switch_discovery_mechanism.ipynb`.

Runs all Bayesian + neural conceptual paths to the relativity of time: `bayes_neural_search_paths.ipynb`.


### Scripts

Core library and runnable modules used by the notebooks: `scripts/`.

Search methods: `enumeration_search.py`, `pure_bayes_search.py`, `bayes_neural_search.py`.

Formal system & reasoning: `typing_rules.py`, `einstein_types.py`, `judgments.py`, `context.py`, `simplify.py`.

Synthesis state and primitives: `synthesis_state.py`, `synthesis_primitives.py`.

### Output

Results produced by notebooks (auto-created): `output/`.

Success index CSVs: `success_indices_enum.csv`, `success_indices_pure_bayes.csv`, `success_indices_bayes_neural.csv`.

Summary statistics: `enum_stats.csv`, `pure_bayes_stats.csv`, `bayes_neural_stats.csv` 

## Run Experiments

You can run the project via **notebooks** or **scripts**.

### Notebooks 
Open in VS Code/Jupyter/Google Drive and run the following:
```bash
- notebooks/all_search_methods_demo.ipynb
- enumeration_search_demo.ipynb
- pure_bayes_search_demo.ipynb
- bayes_neural_search_demo.ipynb
```

Results are stored in `output/` (CSV files listed above)

### Scripts 
Run a script directly:
```bash
- python scripts/enumeration_search.py
- python scripts/pure_bayes_search.py
- python scripts/bayes_neural_search.py
```