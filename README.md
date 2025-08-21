# einstein_program_synthesis
This repository contains code for reconstructing Einstein's conceptual path to the relativity of time using program synthesis

## Setup

```markdown
Install the dependencies using:
```

```bash
pip install -r requirements.txt
```

## Code Structure

### Notebooks

`notebooks/` — Interactive experiments and demos.

`all_search_methods_demo.ipynb` — runs all methods and aggregates results.

`enumeration_search_demo.ipynb` — pure enumeration baseline.

`pure_bayes_search_demo.ipynb` — Bayesian search baseline.

`bayes_neural_search_demo.ipynb` — Bayesian + neural hybrid.

### Other exploratory notebooks: 

`assumption_conclusion_switch_discovery_mechanism.ipynb`

`distinction_discovery_mechanism.ipynb`

`bayes_neural_search_paths.ipynb`

scripts/ — Core library and runnable modules used by the notebooks.

Search methods: `enumeration_search.py`, `pure_bayes_search.py`, `bayes_neural_search.py`.

Formal system & reasoning: `typing_rules.py`, `einstein_types.py`, `judgments.py`, `context.py`.

Synthesis engine & helpers: `synthesis_primitives.py`, `synthesis_state.py`, `simplify.py`.

output/ — Results produced by notebooks (auto-created).

Success index CSVs: `success_indices_enum.csv`, `success_indices_pure_bayes.csv`, `success_indices_bayes_neural.csv` (IDs/problems solved by each method).

Summary stats: `enum_stats.csv`, `pure_bayes_stats.csv`, `bayes_neural_stats.csv` 

## Run Experiments

You can run the project via **notebooks** or **scripts**.

### Notebooks 
Open in VS Code/Jupyter/Google Drive and **Run All**:
- `notebooks/all_search_methods_demo.ipynb` 
Or run individual search methods:
- `enumeration_search_demo.ipynb`
- `pure_bayes_search_demo.ipynb`
- `bayes_neural_search_demo.ipynb`

Artifacts are written to `output/` (CSV files listed above; figures if plotting cells are enabled).

### Scripts 
Run a method directly:
```bash
- python scripts/enumeration_search.py
- python scripts/pure_bayes_search.py
- python scripts/bayes_neural_search.py