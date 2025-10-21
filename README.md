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

### 🚧 Constraints
These are feasibility requirements that all valid layouts must meet:

#### 1) Boundary compliance
All facilities musst remain within site boundaries with margin clearance

$$C_1: \quad \forall i \in facilities, \quad \text{margin} \leq x_i, y_i \leq 1 - \text{margin}$$

where:
- $(x_i, y_i)$ = center position of facility $i$
- $\text{margin}$ = boundary clearance (default: 0.08)

**Violation measure:**

A layout violates this constraint if any facility extends beyond the boundary margins.

$$V_{boundary} = \sum_{i=1}^{n} \max\left(0, \text{margin} - x_i, x_i + \frac{w_i}{2} - 1 + \text{margin}, \text{margin} - y_i, y_i + \frac{h_i}{2} - 1 + \text{margin}\right)$$

#### 2) No overlapping facilities
Facilities cannot physically overlap each other.

$$C_2: \quad \forall i \neq j, \quad A_{\mathrm{overlap}}(f_i, f_j) = 0$$

where:
- $f_i, f_j$ = facility rectangles $i$ and $j$
- $A_{\mathrm{overlap}}(\cdot, \cdot)$ = 2D rectangular intersection area function

**Violation measure:**

A layout violates this constraint if any two facilities have overlapping areas.

$$V_{\mathrm{overlap}} = \sum_{i < j} A_{\mathrm{overlap}}(f_i, f_j)$$

### 🎯 Objective functions
#### 1) Safety compliance
Measures hazard prevention and worker protection.

$$O_1 = 1 - \min\left(1, \frac{\sum_{j \in \text{workers}} P_{danger}(j)}{n_{workers}}\right)$$

**Crane danger penalty** for worker facility $j$:

$$P_{danger}(j) = \begin{cases}
0 & \text{if } d_j \geq r_{danger} \\
\left(\frac{r_{danger} - d_j}{r_{danger}}\right) \times 0.3 & \text{if } d_j < r_{danger}
\end{cases}$$

where:
- $d_j$ = distance from nearest crane to worker facility $j$
- $r_{danger} = 0.25$ (crane danger radius)
- $\text{workers} = \{\text{office}, \text{rest\_area}\}$
