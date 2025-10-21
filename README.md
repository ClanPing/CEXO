# CSLP-Elites
**Hybrid Quality-Diversity Framework for Construction Site Layout Planning (CSLP)**
This project presents CSLP-Elites, a hybrid optimization framework that integrates MAP-Elites an NSGA-II to generate diverse and high-performing construction site layouts.

## 🎯 Key features
- **Three optimization algorithms** with comparative modes
- **Multi-objective optimization** (3 objectives: Safety, Efficiency, Adaptability)
- **Behavioral diversity exploration** (2D behavioral space: Compactness-Spread × Worker-Operational Separation)
- **Comprehensive visualization** and analysis tools
- **Modular architecture** for easy extension and experimentation

## 📁 Project structure
```
CSLP/
[Core Modules]
│   ├── config.py                    # Configurations and data structures
│   ├── objectives.py                # Objective functions for NSGA-II
│   ├── behavioral_descriptors.py    # Behavioral descriptors for MAP-Elites
│   └── layout_generation.py         # Layout generation and genetic operators
│   └── visualization.py             # Visualization and export functions

[Algorithm Implementations]
│   ├── cslpelite_algorithm.py       # CSLP Elite (MAP-Elites + NSGA-II)
│   ├── mapelites_algorithm.py       # Pure MAP-Elites with scalar fitness
│   └── nsga2_algorithm.py           # Pure NSGA-II multi-objective

[Runnable Scripts]
│   ├── run_cslpelite.py             # Run CSLP Elite hybrid algorithm
│   ├── run_mapelites.py             # Run pure MAP-Elites
│   └── run_nsga2.py                 # Run pure NSGA-II

[Output Directories] (generated)
│   ├── cslpelite_output/            # CSLP Elite results
│   ├── mapelites_output/            # MAP-Elites results
│   └── nsga2_output/                # NSGA-II results

[Others]
├── environment.yml                  # Conda environment specification
└── readme_file.md                   # This file
```

## 📋 Project information
For the detailed problem formulation of CSLP-Elites, please refer to this [documentation](project_information.md) which include how layouts are generated, evaluated, and categorised.

This section includes:
- Layout Configurations: Rules for facility selection, placement, and spatial combinations.
- Feasibility Constraints: Boundary and overlap checks ensuring valid site layouts.
- Objective Functions: Multi-objective formulations for safety, efficiency, and adaptability.
- Behavioural Descriptors: Metrics defining diversity dimensions such as compactness–spread and worker–operational separation.

## Installation
[Details]
