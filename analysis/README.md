# Analysis Scripts

This folder groups the manuscript analysis code while keeping all generated
evidence in `../results/`, where the manuscript and older commands already
expect it.

Run commands from the repository root so the scripts can read and write the
shared `results/` folder.

## Reproducibility

- `reproducibility/reproducibility_analysis.py`
  Runs the 30-seed reproducibility study and writes `results/reproducibility_analysis/`.
- `reproducibility/process_reproducibility_results.py`
  Rebuilds `statistics.json` and `table_6.md` from saved CSV results.
- `reproducibility/visualize_reproducibility.py`
  Rebuilds `reproducibility_boxplots.png` and `.pdf`.

Manuscript evidence:
- `results/reproducibility_analysis/all_trials.csv`
- `results/reproducibility_analysis/intermediate_results.csv`
- `results/reproducibility_analysis/statistics.json`
- `results/reproducibility_analysis/table_6.md`
- `results/reproducibility_analysis/reproducibility_boxplots.png`
- `results/reproducibility_analysis/reproducibility_boxplots.pdf`

## Scalability

- `scalability/run_scalability_analysis.py`
  Runs the facility-count scalability sweep and writes `results/scalability_analysis/`.
- `scalability/extract_scalability_results.py`
  Reconstructs the table data from recorded terminal values.
- `scalability/visualize_scalability_results.py`
  Rebuilds `scalability_performance.png` and `.pdf`.

Manuscript evidence:
- `results/scalability_analysis/all_trials.csv`
- `results/scalability_analysis/summary_statistics.csv`
- `results/scalability_analysis/table_7.md`
- `results/scalability_analysis/scalability_performance.png`
- `results/scalability_analysis/scalability_performance.pdf`

## Sensitivity

- `sensitivity/run_sensitivity_analysis.py`
  Runs the manuscript sensitivity sweep and writes `results/sensitivity_analysis/`.
- `sensitivity/visualize_sensitivity_results.py`
  Rebuilds the tornado plot and boundary-margin tradeoff figures.
- `sensitivity/visualize_all_tradeoff_plots.py`
  Rebuilds the per-parameter tradeoff plots.
- `sensitivity/generate_sensitivity_table.py`
  Rebuilds `table_sensitivity.md`.
- `sensitivity/update_sensitivity_section.py` and
  `sensitivity/update_sensitivity_with_all_plots.py`
  Manuscript update helpers.

Manuscript evidence:
- `results/sensitivity_analysis/all_sensitivity_results.json`
- `results/sensitivity_analysis/table_sensitivity.md`
- `results/sensitivity_analysis/figure_7_tornado_plot.png`
- `results/sensitivity_analysis/figure_7_tornado_plot.pdf`
- `results/sensitivity_analysis/figure_8_tradeoff_plot.png`
- `results/sensitivity_analysis/figure_8_tradeoff_plot.pdf`
- `results/sensitivity_analysis/individual_tradeoff_plots/`

## Comparison And Supporting Figures

- `comparison/algorithm_comparison.py`
  Compares CEXO, MAP-Elites, and NSGA-II.
- `comparison/visualize_cexo_archive.py`
  Builds archive comparison figures.
- `comparison/statistical_analysis.py`
  Runs broader statistical comparisons.
- `latent_space/visualize_latent_space.py`
  Builds latent-space figures.

Manuscript evidence:
- `results/algorithm_comparison/`
- `results/archive_visualization/`
- `results/latent_space_visualization/`
- `results/autoencoder_learning/`

## Legacy

- `legacy/scalability_analysis.py`
- `legacy/sensitivity_analysis.py`

These are older generic CLIs that write to `results/scalability` and
`results/sensitivity`, not the current manuscript result folders. Preserve them
for provenance, but prefer the study-specific scripts above.
