# Bulleen Case Study

This folder documents the practical Bulleen case-study pathway included in CEXO v2. It is intentionally lightweight: the case-study geometry and facility assumptions live in `core/config.py`, while generated results remain under `output/` and are ignored by Git.

## Purpose

The Bulleen case study tests CEXO on a more constrained construction site layout problem than the standard synthetic benchmark. It adds:

- A practical fixed facility mix with 23 temporary facilities
- An approximate irregular site boundary
- Three fixed entrance/access points
- Thin road/access corridor exclusion zones
- Optional sampled facility counts for scenario variation

The case is designed to validate whether CEXO can preserve quality-diversity behavior when the layout search space is shaped by practical site constraints.

## Encoded Assumptions

The current practical case uses normalized site coordinates in `[0, 1]`. The geometry is approximate and intended for computational experimentation rather than survey-grade construction documentation.

| Component | Current representation |
|-----------|------------------------|
| Facility mix | 2 cores, 2 cranes, 10 storage areas, 5 offices, 4 rest areas |
| Boundary | Approximate irregular Bulleen polygon |
| Entrances | 3 fixed access points |
| Roads/access | 32 thin exclusion corridor polygons |
| Objectives | Safety, operational efficiency, layout adaptability |
| Archive | 20 x 20 behavioral grid with bounded per-cell Pareto fronts |

The implementation source is:

- `core/config.py`: Bulleen facility mix, boundary, entrances, and road polygons
- `run_cexo.py`: command-line flags and experiment runner
- `core/cexo_algorithm.py`: CEXO archive, Pareto, genetic search, and learned descriptor workflow

## Quick Run

From the repository root:

```powershell
python -X utf8 case_studies/bulleen/run_bulleen.py --test
```

This is equivalent to:

```powershell
python -X utf8 run_cexo.py --test --practical-bulleen --bulleen-boundary --bulleen-entrances --bulleen-roads --output output/bulleen
```

The smoke test still uses the full fixed Bulleen facility mix, so it is slower than the standard CEXO smoke test.

## Standard Experiment

```powershell
python -X utf8 case_studies/bulleen/run_bulleen.py --iterations 15000 --initial-pop 500 --output output/bulleen
```

To sample a practical facility mix from configured ranges:

```powershell
python -X utf8 case_studies/bulleen/run_bulleen.py --sample-bulleen --seed 7 --output output/bulleen_sample_seed7
```

## Outputs

Typical outputs are written to the selected output directory:

- `results.json`
- `cexo_summary.json`
- `cexo_layout_*.json`
- `archive_heatmap.png`
- `best_layout.png`
- `diverse_layouts.png`
- `training_history.png`

Compatibility JSON copies may also be written for older local analysis or dashboard readers.

## Reporting Notes

When reporting Bulleen results, state the exact flags used. The most important distinction is whether the run used:

- fixed practical facility mix: `--practical-bulleen`
- sampled practical facility mix: `--sample-bulleen`
- road/access exclusion corridors: `--bulleen-roads`
- fixed entrances: `--bulleen-entrances`
- irregular site polygon: `--bulleen-boundary`

This keeps standard CEXO experiments and Bulleen practical-case experiments clearly separated while sharing the same core methodology.
