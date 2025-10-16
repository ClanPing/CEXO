# Construction Site Layout Optimization - Modular Implementation

This project implements both MAP-Elites and NSGA-II algorithms for construction site layout optimization, organized into efficient modular components.

## Project Structure

````markdown
# Construction Site Layout Optimization - CSLP Elite

This project implements three optimization algorithms for construction site layout optimization: **CSLP Elite** (MAP-Elites + NSGA-II hybrid), **Pure MAP-Elites**, and **Pure NSGA-II**. The system optimizes facility placement considering safety, efficiency, and adaptability while exploring diverse behavioral characteristics.

## 🎯 Key Features

- **Three optimization algorithms** with distinct advantages
- **Multi-objective optimization** (3 objectives: Safety, Efficiency, Adaptability)
- **Behavioral diversity exploration** (2D behavioral space: Compactness-Spread × Worker-Operational Separation)
- **Comprehensive visualization** and analysis tools
- **Modular architecture** for easy extension and experimentation

---

## 📁 Project Structure

```
CSLP v2 (CHECKPOINT)/
├── Core Modules
│   ├── config.py                    # Configurations and data structures
│   ├── objectives.py                # Three objective functions (shared)
│   ├── behavioral_descriptors.py    # Behavioral descriptors for MAP-Elites
│   └── layout_generation.py         # Layout generation and genetic operators
├── Algorithm Implementations
│   ├── cslpelite_algorithm.py       # CSLP Elite (MAP-Elites + NSGA-II)
│   ├── mapelites_algorithm.py       # Pure MAP-Elites with scalar fitness
│   └── nsga2_algorithm.py           # Pure NSGA-II multi-objective
├── Runnable Scripts
│   ├── run_cslpelite.py            # Run CSLP Elite hybrid algorithm
│   ├── run_mapelites.py            # Run pure MAP-Elites
│   └── run_nsga2.py                # Run pure NSGA-II
├── Utilities
│   └── visualization.py             # Visualization and export functions
├── Output Directories (generated)
│   ├── cslpelite_output/           # CSLP Elite results
│   ├── mapelites_output/           # MAP-Elites results
│   └── nsga2_output/               # NSGA-II results
├── environment.yml                  # Conda environment specification
└── readme_file.md                  # This file
```

---

## 📦 Module Descriptions

### **Core Modules**

#### 1. **`config.py`** - Configuration and Data Structures
**Purpose**: Centralized configuration hub and shared data structures

**Key Components**:
- **`FACILITY_SPECS`**: Dictionary defining facility properties (dimensions, categories, noise levels, danger zones)
  - `core`: Construction work area (0.20×0.20, operational, noise: 0.8)
  - `crane`: Heavy lifting equipment (0.10×0.10, equipment, noise: 0.95, danger radius: 0.25)
  - `storage`: Material storage (0.20×0.15, operational, noise: 0.6)
  - `office`: Worker office (0.15×0.12, worker, noise: 0.1)
  - `rest_area`: Rest facility (0.18×0.10, worker, noise: 0.1)

- **`SiteConfig`** dataclass: Site-level parameters
  - `facility_count`: Number of facilities (3-8)
  - `boundary_margin`: Safety margin from edges (default: 0.08)
  - `min_entrances`/`max_entrances`: Entrance count range (1-6)
  - `entrance_clearance`: Minimum clearance around entrances (0.15)
  - `crane_safety_distance`: Safety distance from crane to workers (0.30)
  - `pareto_size`: Maximum Pareto front size per cell (12)

- **`Individual`** dataclass: Solution representation
  - `solution`: List of facility placements (type + center coordinates)
  - `entrances`: List of entrance positions
  - `objectives`: Tuple of 3 objectives (safety, efficiency, adaptability)
  - `behaviors`: Tuple of 2 behavioral descriptors (for MAP-Elites)
  - `feasible`: Boolean constraint satisfaction flag
  - `violations`: List of constraint violation descriptions

- **Utility Functions**:
  - `generate_facility_mix(count, seed)`: Smart facility type selection
  - `rectangles_overlap(f1, f2)`: Geometric overlap detection
  - `calculate_overlap_area(f1, f2)`: Overlap area computation

**Role**: Provides the foundation that all other modules depend on

---

#### 2. **`objectives.py`** - Three Objective Functions
**Purpose**: Multi-objective evaluation functions shared across all algorithms

**Objective 1: Safety & Constraint Compliance** (Higher is better)

Combines three safety components with graduated penalties:

$$O_1 = 0.4 \times S_{boundary} + 0.3 \times S_{overlap} + 0.3 \times S_{safety}$$

**Boundary Compliance** (40%):
$$S_{boundary} = 1 - \min\left(1, \frac{n_{violations}}{n_{facilities}} \times 0.7 + \sum_{i} v_i \times 2.0\right)$$

where $v_i$ = distance violation for facility $i$ from boundary

**Overlap Prevention** (30%):
$$S_{overlap} = 1 - \min\left(1, n_{overlaps} \times 0.15 + A_{total\_overlap} \times 3.0\right)$$

where $A_{total\_overlap}$ = sum of all overlap areas

**Critical Safety** (30%):
$$S_{safety} = 1 - \min\left(1, n_{safety\_violations} \times 0.1 + \sum_{j} \frac{r_{danger} - d_j}{r_{danger}} \times 0.3\right)$$

where:
- $r_{danger}$ = crane danger radius (0.25)
- $d_j$ = distance from crane to worker facility $j$
- Only counts when $d_j < r_{danger}$

**Function**: `calculate_safety_compliance(facilities, entrances, config)`
- Returns: (score ∈ [0,1], feasible flag, violation list)

---

**Objective 2: Operational Efficiency** (Higher is better)

Optimizes material flows and equipment accessibility:

$$O_2 = 0.4 \times E_{flow} + 0.4 \times E_{access} + 0.2 \times E_{sequence}$$

**Material Flow Efficiency** (40%):
$$E_{flow} = 1 - \frac{\bar{d}_{flow}}{d_{diagonal}}$$

where:
- $\bar{d}_{flow}$ = average distance for critical flows (storage→core, crane→core, storage→crane)
- $d_{diagonal} = \sqrt{2} \times 0.8$ = site diagonal

**Equipment Accessibility** (40%):
$$E_{access} = \frac{1}{W} \sum_{w \in workAreas} C_w \times weight_w$$

where crane coverage $C_w$ for work area $w$:

$$C_w = \begin{cases} 
1.0 & \text{if } d_w \leq r_{optimal} = 0.25 \\
1.0 - \frac{(d_w - r_{optimal})}{(r_{max} - r_{optimal})} \times 0.4 & \text{if } r_{optimal} < d_w \leq r_{max} = 0.40 \\
0.0 & \text{if } d_w > r_{max}
\end{cases}$$

With redundancy bonus for multiple crane coverage:
$$C_w' = \min\left(1.0, C_w + 0.15 \times (n_{effective\_cranes} - 1)\right)$$

**Work Sequence Support** (20%):
$$E_{sequence} = \frac{1}{n_{offices}} \sum_{o \in offices} \max\left(0, 1 - \frac{d_{o,nearest\_entrance}}{0.4\sqrt{2}}\right)$$

**Function**: `calculate_operational_efficiency(facilities, entrances)`
- Returns: score ∈ [0,1]

---

**Objective 3: Layout Adaptability** (Higher is better)

Measures future flexibility and reconfiguration potential:

$$O_3 = 0.4 \times A_{expansion} + 0.35 \times A_{redundancy} + 0.25 \times A_{reconfig}$$

**Expansion Potential** (40%):
$$A_{expansion} = \frac{n_{available\_cells}}{n_{usable\_cells}}$$

Uses 10×10 grid overlay to measure unused space

**Route Redundancy** (35%):
$$A_{redundancy} = \frac{1}{n_{pairs}} \sum_{p \in keyPairs} \frac{1}{1 + 10 \times Var(d_p)}$$

where:
- $keyPairs$ = {(office→core), (storage→core), (crane→storage)}
- $Var(d_p)$ = variance of distances for pair $p$

**Reconfiguration Ease** (25%):
$$A_{reconfig} = \frac{1}{n_{facilities}} \sum_{f \in facilities} \frac{n_{valid\_positions}}{n_{tested\_positions}}$$

Tests 20 random alternative positions per facility within 0.5 distance

**Function**: `calculate_layout_adaptability(facilities, entrances, config)`
- Returns: score ∈ [0,1]

---

**Combined Evaluation**:
**Function**: `evaluate_individual(solution, entrances, config, calculate_behaviors=False)`
- Evaluates all three objectives
- Optionally computes behavioral descriptors for MAP-Elites
- Returns: Dict with objectives, feasibility, violations, and behaviors

---

#### 3. **`behavioral_descriptors.py`** - MAP-Elites Behavioral Space
**Purpose**: Defines 2D behavioral space for diversity exploration

**Behavioral Descriptor 1: Compactness vs. Spread**

Measures spatial organization from centralized to distributed:

$$BD_1 = \frac{\bar{d}_{centroid}}{B}$$

where:
- $\bar{d}_{centroid} = \frac{1}{n} \sum_{i=1}^{n} \|\mathbf{p}_i - \mathbf{c}\|$
- $\mathbf{c} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{p}_i$ (global centroid)
- $B = 0.50$ (normalization bound - maximum mean distance)

**Range**: [0, 1]
- 0.0 = Very compact (all facilities clustered near center)
- 0.5 = Moderate distribution
- 1.0 = Very spread out (facilities dispersed across site)

**Function**: `calculate_compactness_vs_spread(facilities)` or `calculate_spatial_organization(facilities)`

---

**Behavioral Descriptor 2: Worker-Operational Separation**

Measures functional integration from embedded to segregated:

$$BD_2 = \frac{\bar{d}_{separation}}{S}$$

where:
- $\bar{d}_{separation} = \frac{1}{n_{workers}} \sum_{w \in workers} \min_{o \in operational} \|\mathbf{p}_w - \mathbf{p}_o\|$
- $S = 0.30$ (normalization bound - maximum reasonable separation)
- $workers$ = {office, rest_area}
- $operational$ = {core, storage, crane}

**Range**: [0, 1]
- 0.0 = Workers embedded within operational zones
- 0.5 = Moderate separation
- 1.0 = Strong segregation (workers isolated from operations)

**Function**: `calculate_worker_operational_separation(facilities)`

**Inverted Version** (Functional Integration):
$$BD_2' = 1 - BD_2$$

**Function**: `calculate_functional_integration(facilities)`

---

**Behavioral Quadrants**:
The 2D space creates four behavioral regions:

| BD1 \ BD2 | Segregated (BD2 < 0.5) | Integrated (BD2 ≥ 0.5) |
|-----------|------------------------|------------------------|
| **Centralized (BD1 < 0.5)** | Clustered with clear separation | Clustered with mixed zones |
| **Distributed (BD1 ≥ 0.5)** | Spread with clear separation | Spread with mixed zones |

**Utility Functions**:
- `get_behavioral_description(bd1, bd2)`: Human-readable interpretation
- `get_behavioral_quadrant(bd1, bd2)`: Quadrant name
- `analyze_behavioral_regions(individuals)`: Regional statistics and analysis

---

#### 4. **`layout_generation.py`** - Genetic Operators and Solution Construction
**Purpose**: Provides all solution generation and manipulation operations

**Generation Functions**:
- **`generate_random_entrances(config, seed)`**: 
  - Creates 1-3 entrances on site boundaries
  - Ensures minimum separation (0.2) between entrances
  - Distributes across edges (bottom, top, left, right)

- **`create_random_layout(facility_types, boundary_margin)`**:
  - Sequential facility placement with overlap avoidance
  - Priority order: office → rest_area → core → storage → crane
  - Uses iterative position sampling (100 attempts per facility)

- **`create_targeted_layout(facility_types, boundary_margin, target_spatial_org, target_functional_int)`**:
  - Behaviorally-targeted layout generation
  - Positions facilities to achieve specific BD values
  - Used for initial population diversity

**Variation Operators**:
- **`mutate_layout(layout, boundary_margin, p_mut=0.4, sigma=0.04)`**:
  - Gaussian mutation with constraint enforcement
  - Each facility mutated with probability $p_{mut} = 0.4$
  - Displacement: $\Delta x, \Delta y \sim \mathcal{N}(0, \sigma^2)$ where $\sigma = 0.04$
  - Evaluates 8 candidate positions, selects best (minimum violations)

- **`mutate_toward_behavioral_diversity(layout, boundary_margin, current_spatial_org, current_functional_int)`**:
  - **Unique to CSLP Elite!** Behavioral-directed mutation
  - Pushes solutions toward opposite behavioral regions
  - Target: $BD_{target} = 1 - BD_{current} + \mathcal{U}(-0.2, 0.2)$
  - Spatial adjustment toward/away from centroid by 0.15 units
  - 40% mutation probability per facility

- **`crossover_layouts(layout1, layout2, p_swap=0.5)`**:
  - Position-level crossover preserving facility types
  - Swaps coordinates between matching facility types
  - Each pair crossed with probability $p_{swap} = 0.5$
  - Returns two offspring

**Constraint Repair**:
- **`repair_layout_constraints(layout, boundary_margin, entrances, config)`**:
  - Fixes infeasible solutions with minimal behavioral change
  - Tests 25 positions in expanding circles around original position
  - Prioritizes: entrance clearance < overlaps < boundaries
  - Preserves behavioral characteristics by searching locally first

---

### **Algorithm Implementations**

#### 5. **`cslpelite_algorithm.py`** - CSLP Elite (Hybrid MAP-Elites + NSGA-II)
**Purpose**: Combined behavioral diversity exploration with multi-objective optimization

**Key Components**:

**`ParetoFront` Class**:
- Maintains non-dominated set of solutions per behavioral cell
- Maximum size: 12 solutions (configurable)
- Uses NSGA-II dominance and crowding distance for trimming
- **Strict safety threshold**: Only accepts solutions with $O_1 \geq 0.95$

**`MapElitesArchive` Class**:
- 2D grid structure: default 20×20 = 400 cells
- Each cell contains a Pareto front (not single solution!)
- Grid mapping: $(i, j) = (\lfloor BD_1 \times 20 \rfloor, \lfloor BD_2 \times 20 \rfloor)$
- Tracks total evaluations and coverage metrics

**`MapElitesNSGA2Optimizer` Class**:
- **Main loop** (15,000 iterations):
  1. Generate/select parent(s)
  2. Apply variation operators (crossover + mutation)
  3. Evaluate offspring
  4. Add to appropriate behavioral cell if non-dominated
  5. Track statistics

- **Variation strategy**:
  - 60% behavioral diversity mutation
  - 40% standard mutation + crossover

- **Parent selection**:
  - Random cell selection
  - Random individual from cell's Pareto front

**Performance Evaluation**:
**Function**: `evaluate_mapelites_performance(archive, config)`
- Coverage metrics: cells filled, percentage, individuals per cell
- Diversity metrics: BD1 range, BD2 range, behavioral uniformity
- Quality metrics: average objectives, best per cell
- Returns comprehensive evaluation dictionary

**Best for**: Discovering diverse high-quality solutions across behavioral space

---

#### 6. **`mapelites_algorithm.py`** - Pure MAP-Elites with Scalar Fitness
**Purpose**: Behavioral diversity exploration with scalar fitness ranking

**Key Differences from CSLP Elite**:
- **Single solution per cell** (not Pareto front)
- **Scalar fitness function**:

$$F_{scalar} = 0.5 \times O_1 + 0.3 \times O_2 + 0.2 \times O_3 + bonus$$

where:
$$bonus = \begin{cases} +0.1 & \text{if feasible} \\ \times 0.7 & \text{if infeasible} \end{cases}$$

- Cell replacement: new solution replaces old if $F_{new} > F_{old}$

**`PureMapElitesArchive` Class**:
- Simpler structure than CSLP Elite
- Direct fitness comparison
- Faster iteration (no Pareto operations)

**`PureMapElitesOptimizer` Class**:
- Similar main loop to CSLP Elite
- Scalar fitness-based selection
- Same behavioral descriptors

**Best for**: Fast exploration, baseline comparisons, single-objective focus with diversity

---

#### 7. **`nsga2_algorithm.py`** - Pure NSGA-II Multi-Objective Optimization
**Purpose**: Standard NSGA-II without behavioral constraints

**Core Operations**:

**Pareto Dominance**:
**Function**: `dominates(ind1, ind2)`
- $ind_1$ dominates $ind_2$ iff:
  - $\forall i: O_i^{(1)} \geq O_i^{(2)}$ (better or equal in all objectives)
  - $\exists j: O_j^{(1)} > O_j^{(2)}$ (strictly better in at least one)

**Non-Dominated Sorting**:
**Function**: `non_dominated_sort(population)`
- Returns fronts: $F_1, F_2, ..., F_k$
- $F_1$ = non-dominated individuals (rank 0)
- $F_{i+1}$ = individuals dominated only by $F_1, ..., F_i$

**Crowding Distance**:
**Function**: `calculate_crowding_distance(front)`

For individual $i$ in front:
$$CD_i = \sum_{m=1}^{M} \frac{f_m^{(i+1)} - f_m^{(i-1)}}{f_m^{max} - f_m^{min}}$$

where $M = 3$ objectives, individuals sorted by objective $m$

Boundary individuals: $CD = \infty$

**`PureNSGA2Optimizer` Class**:
- **Population**: 200 individuals (default)
- **Generations**: 300 (default)
- **Selection**: Tournament selection (size = 3)
- **Crossover rate**: 0.8
- **Mutation rate**: 0.4

**Evolution Loop**:
1. Initialize population
2. Evaluate and sort
3. For each generation:
   - Select parents via tournament
   - Apply crossover and mutation
   - Combine parents + offspring
   - Non-dominated sorting
   - Select next generation using rank + crowding distance

**Performance Metrics**:
**Function**: `calculate_nsga2_metrics(final_population)`
- Pareto front size
- Hypervolume (if available)
- Spacing metric
- Objective ranges
- Convergence statistics

**Best for**: Finding optimal trade-offs, classical MOO benchmarking

---

#### 8. **`visualization.py`** - Visualization and Export Functions
**Purpose**: Comprehensive results visualization and data export

**Layout Visualization**:
- **`visualize_layout(ax, individual, config, title)`**:
  - Renders facility rectangles with color coding
  - Shows entrances, danger zones, clearance areas
  - Displays objectives and behavioral descriptors
  - Feasibility status indicator

**Algorithm-Specific Visualizations**:

**CSLP Elite / MAP-Elites**:
- **`create_mapelites_visualizations(archive, config, output_dir)`**:
  - Archive grid heatmap (coverage + quality)
  - Best layouts per behavioral region
  - 3D objective space plot
  - Behavioral distribution analysis

**NSGA-II**:
- **`create_nsga2_visualizations(results, config, output_dir)`**:
  - Pareto front 3D scatter
  - Parallel coordinates plot
  - Trade-off analysis
  - Convergence curves

**Export Functions**:
- **`export_cslpelite_results(archive, config, output_dir, max_layouts)`**
- **`export_mapelites_results(archive, config, output_dir, max_layouts)`**
- **`export_nsga2_results(results, config, output_dir, max_layouts)`**

Each exports:
- Individual layout JSON files with coordinates and metrics
- Summary JSON with statistics
- Evaluation metrics JSON
- PNG visualizations

**Gallery Generation**:
- **`show_quality_layouts(individuals, config, output_dir, title, max_display)`**:
  - Multi-panel figure with best solutions
  - Sorted by objective quality
  - Annotated with metrics

---

## 🚀 Runnable Scripts and Execution Commands

### **1. CSLP Elite (Hybrid)**: `run_cslpelite.py`
**Best Algorithm**: Combines behavioral diversity exploration with multi-objective optimization

**Basic Usage**:
```powershell
# Default run (6 facilities, 15000 iterations, 20×20 grid)
python run_cslpelite.py --visualize

# Quick test (reduced parameters for validation)
python run_cslpelite.py --test --visualize

# Production run with custom parameters
python run_cslpelite.py --facilities 7 --iterations 25000 --grid-size 25 --pareto-size 15 --visualize
```

**Command-Line Arguments**:

**Core Parameters**:
- `--facilities N`: Number of facilities (3-8, default: 6)
- `--iterations N`: Evolution iterations (default: 15000)
- `--init-pop N`: Initial population size (default: 500)

**Archive Configuration**:
- `--grid-size N`: Grid size per dimension (10-30, default: 20)
  - Total cells = N²  (e.g., 20×20 = 400 cells)
- `--pareto-size N`: Max Pareto front size per cell (5-25, default: 12)

**Site Configuration**:
- `--min-entrances N`: Minimum entrances (1-6, default: 1)
- `--max-entrances N`: Maximum entrances (1-6, default: 3)
- `--margin F`: Boundary margin (default: 0.08)
- `--entrance-clearance F`: Entrance clearance distance (default: 0.15)
- `--crane-safety F`: Crane safety distance (default: 0.30)

**Output Options**:
- `--output-dir DIR`: Output directory (default: `cslpelite_output`)
- `--export-count N`: Number of layouts to export (default: 25)
- `--visualize`: Create visualizations (flag)
- `--seed N`: Random seed (default: 42)
- `--test`: Run quick test mode (flag)

**Examples**:
```powershell
# Large complex site
python run_cslpelite.py --facilities 8 --iterations 30000 --grid-size 30 --visualize

# Strict safety requirements
python run_cslpelite.py --margin 0.1 --entrance-clearance 0.20 --crane-safety 0.40 --visualize

# High diversity exploration
python run_cslpelite.py --grid-size 25 --pareto-size 20 --init-pop 800 --visualize

# Reproducible run with specific seed
python run_cslpelite.py --seed 123 --facilities 6 --iterations 15000 --visualize
```

**Output Files**:
- `cslpelite_output/`
  - `cslpelite_layout_000.json` to `cslpelite_layout_024.json`: Individual layouts
  - `cslpelite_summary.json`: Overall statistics
  - `cslpelite_evaluation.json`: Archive performance metrics
  - `behavioral_analysis.json`: Behavioral region analysis
  - `cslpelite_analysis.png`: Multi-panel visualization
  - `cslpelite_layouts.png`: Layout gallery

---

### **2. Pure MAP-Elites**: `run_mapelites.py`
**Focus**: Behavioral diversity with scalar fitness

**Basic Usage**:
```powershell
# Default run
python run_mapelites.py --visualize

# Quick test
python run_mapelites.py --test --visualize

# Custom fitness weights
python run_mapelites.py --safety-weight 0.6 --efficiency-weight 0.25 --adaptability-weight 0.15 --visualize
```

**Command-Line Arguments**:
*(Similar to CSLP Elite, plus:)*

**Scalar Fitness Weighting**:
- `--safety-weight F`: Weight for safety (default: 0.5)
- `--efficiency-weight F`: Weight for efficiency (default: 0.3)
- `--adaptability-weight F`: Weight for adaptability (default: 0.2)
  - Note: Weights must sum to 1.0 (auto-normalized if not)

**Examples**:
```powershell
# Prioritize safety heavily
python run_mapelites.py --safety-weight 0.7 --efficiency-weight 0.2 --adaptability-weight 0.1 --visualize

# Fast exploration (smaller grid)
python run_mapelites.py --grid-size 15 --iterations 10000 --visualize

# Balanced objectives
python run_mapelites.py --safety-weight 0.33 --efficiency-weight 0.33 --adaptability-weight 0.34 --visualize
```

**Output Files**:
- `mapelites_output/` (or `pure_mapelites_output/`)
  - `mapelites_layout_000.json` to `mapelites_layout_024.json`
  - `mapelites_summary.json`
  - `mapelites_evaluation.json`
  - `behavioral_analysis.json`
  - `mapelites_analysis.png`
  - `mapelites_layouts.png`

---

### **3. Pure NSGA-II**: `run_nsga2.py`
**Focus**: Multi-objective Pareto optimization without behavioral diversity

**Basic Usage**:
```powershell
# Default run (200 pop, 300 generations)
python run_nsga2.py --visualize

# Quick test
python run_nsga2.py --test --visualize

# High-quality convergence
python run_nsga2.py --population 400 --generations 600 --visualize
```

**Command-Line Arguments**:

**Core Parameters**:
- `--facilities N`: Number of facilities (3-8, default: 6)
- `--population N`: Population size (default: 200)
- `--generations N`: Number of generations (default: 300)

**NSGA-II Specific**:
- `--tournament-size N`: Tournament selection size (default: 3)
- `--crossover-rate F`: Crossover probability (default: 0.8)
- `--mutation-rate F`: Mutation probability (default: 0.4)

**Site Configuration**: (Same as CSLP Elite)
- `--min-entrances`, `--max-entrances`, `--margin`, `--entrance-clearance`, `--crane-safety`

**Output Options**:
- `--output-dir DIR`: Output directory (default: `nsga2_output`)
- `--export-count N`: Number of layouts to export (default: 25)
- `--visualize`, `--seed`, `--test`

**Examples**:
```powershell
# Large population for better convergence
python run_nsga2.py --population 400 --generations 500 --visualize

# High selection pressure
python run_nsga2.py --tournament-size 5 --crossover-rate 0.9 --mutation-rate 0.3 --visualize

# Extended optimization
python run_nsga2.py --population 300 --generations 800 --visualize

# Complex site
python run_nsga2.py --facilities 8 --population 400 --generations 600 --visualize
```

**Output Files**:
- `nsga2_output/`
  - `nsga2_layout_000.json` to `nsga2_layout_024.json`
  - `nsga2_summary.json`
  - `nsga2_detailed_metrics.json`
  - `nsga2_analysis.png`
  - `nsga2_pareto_layouts.png`

---

## 🔬 Algorithm Comparison and Selection Guide

| Aspect | **CSLP Elite** | **Pure MAP-Elites** | **Pure NSGA-II** |
|--------|----------------|---------------------|------------------|
| **Optimization Approach** | Multi-objective + Diversity | Scalar + Diversity | Multi-objective Only |
| **Output Structure** | 2D grid of Pareto fronts | 2D grid of single solutions | Single Pareto front |
| **Solution Count** | Up to 400×12 = 4,800 | Up to 400 solutions | ~100-200 solutions |
| **Behavioral Space** | ✅ Explored systematically | ✅ Explored systematically | ❌ Not considered |
| **Computational Cost** | High (most expensive) | Medium | Low (fastest) |
| **Best For** | Diverse alternatives, design exploration | Fast diversity exploration | Optimal trade-offs |
| **Use When** | Need solution database with diversity | Need quick behavioral coverage | Need best Pareto front |
| **Typical Runtime** | 5-10 minutes | 3-5 minutes | 2-4 minutes |

**Decision Tree**:
```
Need diverse solutions? 
├─ Yes → Need multi-objective optimization?
│        ├─ Yes → Use CSLP Elite ★★★
│        └─ No → Use Pure MAP-Elites ★★
└─ No → Use Pure NSGA-II ★
```

---

## 📊 Performance Metrics and Evaluation

### **CSLP Elite / MAP-Elites Metrics**:
- **Coverage**: Percentage of behavioral cells filled
- **Density**: Average individuals per occupied cell
- **Behavioral Diversity**: Range of BD1 and BD2 values
- **Quality**: Average and best objectives per cell
- **MAP-Elites Effectiveness**: Combined coverage × quality score

### **NSGA-II Metrics**:
- **Pareto Front Size**: Number of non-dominated solutions
- **Hypervolume**: Volume dominated in objective space
- **Spacing**: Uniformity of solution distribution
- **Convergence**: Progress toward optimal front
- **Objective Ranges**: Spread in each objective

---

## 📦 Dependencies and Installation

### **Required Packages**:
```powershell
# Using pip
pip install numpy matplotlib

# Or using conda (recommended)
conda env create -f environment.yml
conda activate cslp
```

### **Python Version**:
- Python 3.8 or higher recommended
- Tested on Python 3.10, 3.11, 3.12

### **Optional Packages** (for advanced analysis):
```powershell
pip install scipy pandas seaborn
```

---

## 💡 Usage Examples and Workflows

### **Quick Start: Testing All Algorithms**
```powershell
# Test CSLP Elite (1-2 minutes)
python run_cslpelite.py --test --visualize

# Test Pure MAP-Elites (1 minute)
python run_mapelites.py --test --visualize

# Test Pure NSGA-II (30-60 seconds)
python run_nsga2.py --test --visualize
```

### **Typical Research Workflow**:

**1. Initial Exploration** (CSLP Elite):
```powershell
# Generate diverse solution database
python run_cslpelite.py --facilities 6 --iterations 15000 --visualize
```

**2. Behavioral Analysis**:
- Examine `behavioral_analysis.json` for region distribution
- Review `cslpelite_layouts.png` for visual diversity
- Identify interesting behavioral patterns

**3. Refinement** (Targeted runs):
```powershell
# Focus on specific facility count
python run_cslpelite.py --facilities 7 --iterations 25000 --grid-size 25 --visualize

# Increase solution quality per cell
python run_cslpelite.py --pareto-size 20 --iterations 30000 --visualize
```

**4. Baseline Comparison**:
```powershell
# Pure NSGA-II for comparison
python run_nsga2.py --facilities 6 --population 300 --generations 500 --visualize

# Pure MAP-Elites for speed comparison
python run_mapelites.py --facilities 6 --iterations 15000 --visualize
```

### **Scenario-Specific Configurations**:

**Small Construction Site (3-4 facilities)**:
```powershell
python run_cslpelite.py --facilities 4 --min-entrances 1 --max-entrances 2 --iterations 10000 --visualize
```

**Medium Site (5-6 facilities)** [Default]:
```powershell
python run_cslpelite.py --facilities 6 --iterations 15000 --visualize
```

**Large Complex Site (7-8 facilities)**:
```powershell
python run_cslpelite.py --facilities 8 --iterations 30000 --init-pop 800 --grid-size 25 --visualize
```

**High Safety Requirements**:
```powershell
python run_cslpelite.py --margin 0.12 --entrance-clearance 0.20 --crane-safety 0.40 --visualize
```

**Urban Constrained Site** (tight margins):
```powershell
python run_cslpelite.py --margin 0.06 --entrance-clearance 0.12 --facilities 5 --visualize
```

**Multi-Entrance Site**:
```powershell
python run_cslpelite.py --min-entrances 2 --max-entrances 4 --visualize
```

### **Batch Processing for Comparative Studies**:
```powershell
# Run multiple facility counts
foreach ($n in 4..7) {
    python run_cslpelite.py --facilities $n --seed 42 --output-dir "results_$n" --visualize
}

# Run with different seeds for statistical analysis
foreach ($s in 42, 123, 456, 789, 1024) {
    python run_cslpelite.py --seed $s --output-dir "seed_$s" --visualize
}

# Compare all three algorithms
python run_cslpelite.py --facilities 6 --seed 42 --output-dir "compare/cslp" --visualize
python run_mapelites.py --facilities 6 --seed 42 --output-dir "compare/mapelites" --visualize
python run_nsga2.py --facilities 6 --seed 42 --output-dir "compare/nsga2" --visualize
```

---

## 📁 Output Files Structure

### **CSLP Elite Output** (`cslpelite_output/`):
```
cslpelite_output/
├── cslpelite_layout_000.json      # Best layout (highest objectives)
├── cslpelite_layout_001.json      # Second best layout
├── ...
├── cslpelite_layout_024.json      # 25th best layout
├── cslpelite_summary.json         # Overall statistics
│   └── Contains: coverage, total individuals, avg objectives, runtime
├── cslpelite_evaluation.json      # Archive performance metrics
│   └── Contains: coverage metrics, diversity metrics, effectiveness score
├── behavioral_analysis.json       # Behavioral region statistics
│   └── Contains: counts per quadrant, avg objectives per region
├── cslpelite_analysis.png         # Multi-panel visualization
│   └── Shows: archive grid, 3D objectives, behavioral distribution
└── cslpelite_layouts.png          # Gallery of top 12 layouts
```

### **Layout JSON Format**:
```json
{
  "metadata": {
    "algorithm": "CSLP Elite",
    "facility_count": 6,
    "grid_size": [20, 20],
    "timestamp": "2025-10-16T14:30:00"
  },
  "objectives": {
    "safety": 0.982,
    "efficiency": 0.876,
    "adaptability": 0.823
  },
  "behaviors": {
    "compactness_spread": 0.342,
    "worker_operational_separation": 0.678
  },
  "facilities": [
    {
      "type": "office",
      "center": [0.245, 0.678],
      "width": 0.15,
      "depth": 0.12
    },
    ...
  ],
  "entrances": [
    [0.5, 0.08],
    [0.8, 0.92]
  ],
  "feasible": true,
  "violations": []
}
```

### **Pure MAP-Elites Output** (`mapelites_output/` or `pure_mapelites_output/`):
Similar structure to CSLP Elite but:
- Only one solution per cell (not Pareto front)
- Includes scalar fitness values
- `mapelites_evaluation.json` has simplified metrics

### **Pure NSGA-II Output** (`nsga2_output/`):
```
nsga2_output/
├── nsga2_layout_000.json          # Pareto front solutions
├── ...
├── nsga2_layout_024.json
├── nsga2_summary.json             # Population statistics
├── nsga2_detailed_metrics.json    # Pareto metrics
│   └── Contains: hypervolume, spacing, convergence, front size
├── nsga2_analysis.png             # 3D Pareto front + convergence
└── nsga2_pareto_layouts.png       # Top Pareto solutions gallery
```

---

## 🎨 Key Design Principles

1. **Modularity**: Each module has single, well-defined responsibility
   - `config.py`: Configuration only
   - `objectives.py`: Evaluation only
   - `layout_generation.py`: Genetic operators only

2. **Reusability**: Core functions shared across algorithms
   - All three algorithms use same objective functions
   - Same layout generation and mutation operators
   - Common visualization functions

3. **Configurability**: Extensive command-line customization
   - 20+ configurable parameters per algorithm
   - Seed control for reproducibility
   - Test mode for quick validation

4. **Efficiency**: Optimized data structures
   - NumPy arrays for geometric calculations
   - Dictionary-based archive for O(1) lookup
   - Graduated penalty functions avoid binary constraints

5. **Extensibility**: Easy to extend
   - Add new objectives to `objectives.py`
   - Add new behavioral descriptors to `behavioral_descriptors.py`
   - Add new facility types to `FACILITY_SPECS`
   - Add new mutation operators to `layout_generation.py`

---

## ⚡ Performance Tips and Best Practices

### **Optimization Speed**:
- **Use `--test` flag** for quick validation (3x faster)
- **Start small**: Begin with 4-5 facilities before scaling up
- **Monitor feasibility**: If <30% feasible, relax constraints
- **Grid size trade-off**: Larger grids = better diversity but slower

### **Quality Improvement**:

**For CSLP Elite / MAP-Elites**:
```powershell
# Better coverage
python run_cslpelite.py --init-pop 800 --iterations 20000 --visualize

# Higher quality per cell
python run_cslpelite.py --pareto-size 20 --iterations 25000 --visualize

# Fine-grained behavioral space
python run_cslpelite.py --grid-size 30 --iterations 30000 --visualize
```

**For Pure NSGA-II**:
```powershell
# Better convergence
python run_nsga2.py --population 400 --generations 600 --visualize

# Higher selection pressure
python run_nsga2.py --tournament-size 5 --visualize

# More genetic diversity
python run_nsga2.py --mutation-rate 0.5 --crossover-rate 0.9 --visualize
```

### **Constraint Satisfaction**:
If getting "No safety-feasible layouts found":
1. **Relax margins**: `--margin 0.10` (from 0.08)
2. **Relax clearances**: `--entrance-clearance 0.12` (from 0.15)
3. **Fewer facilities**: `--facilities 5` (from 6)
4. **Longer optimization**: `--iterations 25000` or `--generations 500`

### **Reproducibility**:
```powershell
# Always use same seed for reproducible results
python run_cslpelite.py --seed 42 --facilities 6 --iterations 15000 --visualize

# For statistical significance, run multiple seeds
foreach ($s in 42, 123, 456, 789, 1024) {
    python run_cslpelite.py --seed $s --output-dir "trial_$s" --visualize
}
```

### **Memory Management**:
- Large grids (30×30) with large Pareto fronts (25) = ~20MB memory
- For very large runs, consider reducing `--export-count`
- Close visualization windows to free memory during batch runs

---

## 🔧 Troubleshooting Guide

### **Issue 1: No Feasible Solutions Found**
**Symptom**: "✗ No safety-feasible layouts found!"

**Causes**:
- Too tight boundary margins
- Too many facilities for site size
- Too strict clearance requirements

**Solutions**:
```powershell
# Solution A: Relax constraints
python run_cslpelite.py --margin 0.10 --entrance-clearance 0.12 --visualize

# Solution B: Reduce complexity
python run_cslpelite.py --facilities 5 --max-entrances 2 --visualize

# Solution C: Longer optimization
python run_cslpelite.py --iterations 25000 --init-pop 800 --visualize

# Solution D: Combination
python run_cslpelite.py --margin 0.10 --facilities 5 --iterations 20000 --visualize
```

---

### **Issue 2: Low Archive Coverage (<20%)**
**Symptom**: Coverage percentage very low in MAP-Elites output

**Causes**:
- Grid too large for solution space
- Insufficient iterations
- Difficult constraint landscape

**Solutions**:
```powershell
# Solution A: Reduce grid size
python run_cslpelite.py --grid-size 15 --visualize

# Solution B: More iterations + larger initial population
python run_cslpelite.py --iterations 25000 --init-pop 800 --visualize

# Solution C: Increase behavioral diversity mutation rate
# (requires code modification: increase p_behavioral_mutation in algorithm)

# Solution D: Check if feasibility is the issue
python run_cslpelite.py --margin 0.10 --grid-size 15 --visualize
```

---

### **Issue 3: Poor NSGA-II Convergence**
**Symptom**: Pareto front not improving after many generations

**Causes**:
- Insufficient population diversity
- Low selection pressure
- Premature convergence

**Solutions**:
```powershell
# Solution A: Increase population + generations
python run_nsga2.py --population 400 --generations 600 --visualize

# Solution B: Higher tournament size (more selection pressure)
python run_nsga2.py --tournament-size 5 --generations 400 --visualize

# Solution C: Adjust genetic operators
python run_nsga2.py --mutation-rate 0.5 --crossover-rate 0.9 --visualize

# Solution D: Restart with different seed
python run_nsga2.py --seed 123 --population 300 --generations 500 --visualize
```

---

### **Issue 4: All Solutions in One Behavioral Region**
**Symptom**: Behavioral analysis shows 80%+ solutions in single quadrant

**Causes**:
- Constraints favor certain layouts
- Insufficient behavioral diversity pressure
- Grid size mismatch with solution distribution

**Solutions**:
```powershell
# Solution A: Larger grid for finer resolution
python run_cslpelite.py --grid-size 25 --visualize

# Solution B: Relax constraints to allow more diversity
python run_cslpelite.py --margin 0.10 --entrance-clearance 0.12 --visualize

# Solution C: Longer run for better exploration
python run_cslpelite.py --iterations 30000 --init-pop 800 --visualize

# Check: Is the objective landscape biased toward certain behaviors?
# Review objectives.py to see if certain behavioral patterns score better
```

---

### **Issue 5: Visualization Not Showing / Errors**
**Symptom**: `--visualize` flag doesn't create PNG files or shows errors

**Solutions**:
```powershell
# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"

# Try different backend (if on remote server)
$env:MPLBACKEND="Agg"
python run_cslpelite.py --visualize

# Check if output directory is writable
python run_cslpelite.py --output-dir "test_output" --test --visualize

# Update matplotlib if outdated
pip install --upgrade matplotlib
```

---

### **Issue 6: Out of Memory Errors**
**Symptom**: Program crashes with memory error during large runs

**Solutions**:
```powershell
# Solution A: Reduce grid size and Pareto size
python run_cslpelite.py --grid-size 15 --pareto-size 8 --visualize

# Solution B: Reduce export count
python run_cslpelite.py --export-count 10 --visualize

# Solution C: For NSGA-II, reduce population
python run_nsga2.py --population 150 --visualize

# Solution D: Process in batches (batch script)
foreach ($n in 4..6) {  # Instead of 4..8
    python run_cslpelite.py --facilities $n --output-dir "batch_$n"
}
```

---

### **Issue 7: Runtime Too Long**
**Symptom**: Algorithm takes >30 minutes to complete

**Analysis**:
Expected runtimes on modern PC (i5/i7, 8GB RAM):
- CSLP Elite: 5-10 minutes (default params)
- Pure MAP-Elites: 3-5 minutes
- Pure NSGA-II: 2-4 minutes

**Solutions if much slower**:
```powershell
# Use test mode first
python run_cslpelite.py --test --visualize  # Should be ~1-2 min

# Reduce parameters incrementally
python run_cslpelite.py --iterations 10000 --grid-size 15 --visualize

# Check system resources
# Task Manager → Performance tab (Windows)
# Ensure no other CPU-intensive tasks running

# For batch runs, process sequentially not in parallel
```

---

## 🎯 Advanced Configuration Strategies

### **Site Complexity Scaling**:

| Facility Count | Iterations (CSLP/ME) | Generations (NSGA-II) | Grid Size | Notes |
|----------------|---------------------|----------------------|-----------|-------|
| 3-4 (Simple) | 10,000 | 200 | 15×15 | Quick convergence |
| 5-6 (Medium) | 15,000 | 300 | 20×20 | **Default** |
| 7-8 (Complex) | 25,000-30,000 | 500-600 | 25×25 | Double compute time |

### **Parameter Tuning Guidelines**:

**For Maximum Diversity** (CSLP Elite):
```powershell
python run_cslpelite.py --grid-size 30 --pareto-size 15 --init-pop 1000 --iterations 40000 --visualize
```

**For Maximum Quality** (CSLP Elite):
```powershell
python run_cslpelite.py --pareto-size 25 --iterations 30000 --grid-size 20 --visualize
```

**For Speed** (all algorithms):
```powershell
# CSLP Elite
python run_cslpelite.py --grid-size 12 --pareto-size 6 --iterations 8000 --visualize

# MAP-Elites
python run_mapelites.py --grid-size 12 --iterations 8000 --visualize

# NSGA-II
python run_nsga2.py --population 100 --generations 150 --visualize
```

**For Strict Safety** (all algorithms):
```powershell
# Tighten constraints + more computation
python run_cslpelite.py --margin 0.12 --entrance-clearance 0.20 --crane-safety 0.40 --iterations 30000 --visualize
```

---

## 🔌 Extension Points

The modular architecture makes extensions straightforward:

### **1. Adding New Objectives** (`objectives.py`):
```python
def calculate_custom_objective(facilities, entrances, config):
    """Your new objective (0=worst, 1=best)"""
    score = 0.0
    # Your calculation here
    return float(np.clip(score, 0.0, 1.0))

# Then modify evaluate_individual() to include 4th objective
```

### **2. Adding New Behavioral Descriptors** (`behavioral_descriptors.py`):
```python
def calculate_custom_behavior(facilities: List[Dict]) -> float:
    """Your new behavioral descriptor (0-1 normalized)"""
    value = 0.0
    # Your calculation here
    return float(np.clip(value, 0.0, 1.0))

# Update archive to use 3D grid instead of 2D
# Modify MapElitesArchive.grid_size to (20, 20, 20)
```

### **3. Adding New Facility Types** (`config.py`):
```python
FACILITY_SPECS = {
    # Existing facilities...
    "workshop": {
        "w": 0.18, 
        "d": 0.14, 
        "category": "operational", 
        "noise_level": 0.7
    },
    "security": {
        "w": 0.10, 
        "d": 0.10, 
        "category": "worker", 
        "noise_level": 0.2
    }
}

# Add colors for visualization
FACILITY_COLORS["workshop"] = "#f28e2b"  # Orange
FACILITY_COLORS["security"] = "#76b7b2"  # Teal
```

### **4. Adding New Mutation Operators** (`layout_generation.py`):
```python
def mutate_smart_swap(layout: List[Dict], boundary_margin: float) -> List[Dict]:
    """Swap positions of two facilities"""
    result = list(layout)
    if len(layout) >= 2:
        i, j = random.sample(range(len(layout)), 2)
        result[i]["center"], result[j]["center"] = result[j]["center"], result[i]["center"]
    return result

# Use in algorithm's mutation step
```

### **5. Custom Visualization** (`visualization.py`):
```python
def create_custom_plot(archive, config, output_dir):
    """Your custom analysis plot"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # Your plotting code
    plt.savefig(os.path.join(output_dir, "custom_analysis.png"), dpi=300)
    plt.close()
```

### **6. Hybrid Algorithm Variants**:
- **MAP-Elites with 3D behavioral space**: Add third descriptor
- **NSGA-III integration**: Replace NSGA-II with NSGA-III for 4+ objectives
- **Adaptive grid**: Dynamic grid refinement in promising regions
- **Island model**: Multiple parallel archives with migration

---

## 📐 Technical Details

### **Coordinate System**:
- **Unit square**: All positions normalized to [0, 1] × [0, 1]
- **Origin**: Bottom-left corner (0, 0)
- **Facility representation**: Center point (x, y) + dimensions (width, depth)
- **Boundaries**: Facilities must stay within margin boundaries

### **Constraint Handling**:
- **Graduated penalties**: Violations have severity levels, not binary
- **Soft constraints**: Small violations get small penalties
- **Constraint repair**: `repair_layout_constraints()` fixes violations post-generation
- **Feasibility tracking**: Boolean flag + violation list for transparency

### **Behavioral Space Discretization**:
- **Grid mapping**: $cell(i,j) = (\lfloor BD_1 \times N \rfloor, \lfloor BD_2 \times N \rfloor)$
- **Grid size**: N × N cells (default: 20×20 = 400 cells)
- **Resolution**: Each cell represents $\frac{1}{N} \times \frac{1}{N}$ behavioral region
- **Edge handling**: Values exactly at 1.0 mapped to last cell (N-1)

### **Objective Normalization**:
- All objectives normalized to [0, 1] range
- Higher values = better performance
- Clipping prevents out-of-range values
- Float64 precision for calculations

### **Random Number Generation**:
- Seeded RNG for reproducibility
- Separate RNG instances avoid state conflicts
- Default seed: 42
- Numpy + Python random both seeded

### **Data Structures**:
- **Archive**: Dictionary with (i,j) tuple keys for O(1) access
- **Pareto Front**: List with O(n) dominance checks
- **Individual**: Dataclass for type safety and clarity
- **Facilities**: List of dictionaries (JSON-compatible)

### **Performance Optimizations**:
- NumPy vectorization for distance calculations
- Pre-computed facility specifications
- Lazy evaluation of behaviors (only for MAP-Elites)
- Early termination in constraint checks

---

## 📚 References and Related Work

### **Algorithms**:
- **MAP-Elites**: Mouret & Clune (2015) - "Illuminating the Space of Possibilities"
- **NSGA-II**: Deb et al. (2002) - "A Fast and Elitist Multi-Objective Genetic Algorithm"
- **Quality Diversity**: Pugh et al. (2016) - "Quality Diversity: A New Frontier"

### **Construction Site Layout Optimization**:
- Site layout planning with genetic algorithms
- Multi-objective construction site optimization
- Behavioral diversity in engineering design

### **Hybrid Approaches**:
- Combining QD and MOO algorithms
- Multi-objective MAP-Elites variants
- Archive-based evolutionary algorithms

---

## 📄 Citation

If you use this implementation in your research, please cite:

```bibtex
@software{cslp_elite_2024,
  title = {CSLP Elite: Construction Site Layout Optimization with Behavioral Diversity},
  author = {[Your Name]},
  year = {2024},
  version = {2.0},
  note = {MAP-Elites + NSGA-II hybrid algorithm for multi-objective optimization with behavioral diversity},
  url = {[Your Repository URL]}
}
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional objective functions (noise, cost, time)
- New behavioral descriptors (3D space)
- Algorithm variants (NSGA-III, CMA-ES, etc.)
- Visualization enhancements
- Performance benchmarks
- Real-world case studies

---

## 📞 Support

For issues, questions, or discussions:
1. Check this README first
2. Review troubleshooting section
3. Examine output JSON files for diagnostics
4. Check console output for error messages
5. Verify all dependencies installed correctly

---

## 📝 License

[Specify your license here - MIT, Apache 2.0, etc.]

---

## 🏆 Acknowledgments

- MAP-Elites algorithm: Jean-Baptiste Mouret and Jeff Clune
- NSGA-II algorithm: Kalyanmoy Deb and colleagues
- Construction site layout optimization: Domain experts and practitioners

---

**Version**: 2.0  
**Last Updated**: October 2025  
**Status**: Production-ready for research and experimentation

---

## 📊 Quick Reference Card

| Task | Algorithm | Command |
|------|-----------|---------|
| **Diverse solutions** | CSLP Elite | `python run_cslpelite.py --visualize` |
| **Fast diversity** | MAP-Elites | `python run_mapelites.py --visualize` |
| **Optimal trade-offs** | NSGA-II | `python run_nsga2.py --visualize` |
| **Quick test** | Any | Add `--test` flag |
| **More quality** | CSLP/ME | `--iterations 25000` |
| **Better convergence** | NSGA-II | `--population 400 --generations 600` |
| **Relax constraints** | Any | `--margin 0.10 --entrance-clearance 0.12` |
| **Complex site** | Any | `--facilities 8` |
| **Reproducible** | Any | `--seed 42` |

---

**End of README** 🎉