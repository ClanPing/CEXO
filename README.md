# CSLP-Elites
**Hybrid Quality-Diversity Framework for Construction Site Layout Planning (CSLP)**
This project presents CSLP-Elites, a hybrid optimization framework that integrates MAP-Elites an NSGA-II to generate diverse and high-performing construction site layouts.

## 🎯 Key features
- **Three optimization algorithms** with comparative modes
- **Multi-objective optimization** (3 objectives: Safety, Efficiency, Adaptability)
- **Behavioral diversity exploration** (2D behavioral space: Compactness-Spread × Worker-Operational Separation)
- **Comprehensive visualization** and analysis tools
- **Modular architecture** for easy extension and experimentation

## Project information
### 📁 Project structure
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

### 🏗️ Layout configurations

<p align="center">
<img src="assets/config.png" alt="Facility types">
</p>

```
Facility selection range:

Minimum Facilities: 3 (minimal operational site)
- Always includes: `core`, `crane`, `storage`

Maximum Facilities: 8 (complex multi-function site)
- Includes: All 5 types + additional operational facilities

Default Configuration**: 6 facilities
- Typical mix: `core`, `crane`, `storage`, `office`, `rest_area`, + 1 operational facility
```

```
Facility combination:

If count ≥ 3:  Add [core, crane, storage]           # Operational facilities
If count ≥ 5:  Add [office, rest_area]              # Worker facilities
If count > 5:  Fill remaining with [core, storage, crane]  # Additional operational
Finally:       Shuffle order randomly (seed-controlled)
```

**Example Generations** (default seed=42):

| Count | Facility Mix | Breakdown |
|-------|--------------|-----------|
| 3 | `['storage', 'core', 'crane']` | 3 operational only |
| 4 | `['storage', 'core', 'crane', 'crane']` | 3 operational + 1 extra operational |
| 5 | `['storage', 'core', 'crane', 'rest_area', 'office']` | 3 operational + 2 worker |
| 6 | `['storage', 'core', 'crane', 'rest_area', 'office', 'storage']` | Balanced + 1 extra operational |
| 7 | `['storage', 'core', 'crane', 'rest_area', 'office', 'storage', 'crane']` | Balanced + 2 extra operational |
| 8 | `['storage', 'core', 'crane', 'rest_area', 'office', 'storage', 'crane', 'storage', 'core', 'crane']` | Full site |

### Objective functions
