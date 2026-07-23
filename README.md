# CEXO

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Paper](https://img.shields.io/badge/Paper-Preprint-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18426345.svg)](https://doi.org/10.5281/zenodo.18426345)

**Construction Site eXploration and Optimisation (CEXO)** is a quality-diversity framework for construction site layout planning. It combines MAP-Elites archiving, NSGA-II-style Pareto selection, genetic variation operators, and optional autoencoder-learned behavioral descriptors to produce diverse, high-performing construction site layouts.

## Key Features

- Multi-objective evaluation for safety, operational efficiency, and adaptability
- Quality-diversity archive with a bounded Pareto front in each behavioral cell
- Genetic variation through mutation, crossover, targeted generation, and repair
- Optional autoencoder-learned behavioral descriptors for latent archive organization
- Baseline runners for pure MAP-Elites and pure NSGA-II comparison
- JSON and figure exports for downstream analysis and visualization

## Repository Structure

```text
CEXO/
├── core/
│   ├── config.py
│   ├── objectives.py
│   ├── behavioral_descriptors.py
│   ├── layout_generation.py
│   ├── layout_autoencoder.py
│   ├── cexo_algorithm.py
│   ├── mapelites_with_autoencoder.py
│   ├── mapelites_algorithm.py
│   ├── nsga2_algorithm.py
│   └── visualization.py
├── assets/
├── run_cexo.py
├── run_cslpelites.py
├── run_mapelites.py
├── run_nsga2.py
├── environment.yml
├── INFO.md
└── README.md
```

`run_cexo.py` is the official CEXO entry point. `run_cslpelites.py` remains as a compatibility wrapper for older commands and scripts.

## Installation

```powershell
git clone https://github.com/ClanPing/CEXO.git
cd CEXO
conda env create -f environment.yml
conda activate cexo
```

If your Windows terminal uses a legacy code page, run scripts with Python UTF-8 mode:

```powershell
python -X utf8 run_cexo.py --test
```

## Quick Start

Run a small CEXO smoke test:

```powershell
python -X utf8 run_cexo.py --test --output output/cexo_test
```

Run a standard learned-descriptor CEXO experiment:

```powershell
python -X utf8 run_cexo.py --facilities 6 --iterations 15000 --initial-pop 500 --output output/cexo
```

Run CEXO with hand-crafted behavioral descriptors:

```powershell
python -X utf8 run_cexo.py --facilities 6 --iterations 15000 --no-learned --output output/cexo_handcrafted
```

Run baseline comparisons:

```powershell
python -X utf8 run_mapelites.py --test --output-dir output/mapelites_test
python -X utf8 run_nsga2.py --test --output-dir output/nsga2_test
```

## Methodology

CEXO follows this workflow:

1. Define the construction site, facilities, entrances, and constraints.
2. Generate an initial pool of candidate layouts.
3. Evaluate each layout using safety, efficiency, and adaptability objectives.
4. If learned descriptors are enabled, train an autoencoder on generated layouts and use its two-dimensional latent representation as the behavioral space.
5. Archive layouts in a MAP-Elites grid, keeping a bounded Pareto front in each occupied cell.
6. Continue evolution with mutation, crossover, targeted generation, and constraint repair.
7. Export summary metrics, layout JSON files, and visualization figures.

The baseline scripts preserve the comparison structure:

- `run_mapelites.py`: scalar-fitness MAP-Elites
- `run_nsga2.py`: pure NSGA-II multi-objective optimization

## Outputs

By default, generated outputs are written under `output/` and ignored by Git. CEXO exports include:

- `results.json`: run configuration and summary statistics
- `cexo_summary.json`: CEXO archive and objective summary
- `cexo_layout_*.json`: exported layout candidates
- compatibility copies for older local analysis tools
- `archive_heatmap.png`, `best_layout.png`, `diverse_layouts.png`, and optional training figures

## Project Information

For detailed problem formulation, constraints, objective functions, and behavioral descriptors, see [INFO.md](INFO.md).

## License

MIT License.
