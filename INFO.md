# Project information
In CSLP-Elites, our goal is to **generate diverse and high-performing construction site layouts** that balance safety, efficiency, and adaptability. To achieve this, the framework combines objective functions and behavioural descriptors:

- **Objective Functions** measure how well a layout performs against practical goals, such as maintaining safety distances, reducing material handling time, and improving site adaptability. They guide the optimisation process toward high-quality, feasible configurations.

- **Behavioural Descriptors (BDs)** describe how layouts differ in their spatial and functional organisation (e.g., compact vs spread, integrated vs segregated). They encourage exploration of the design space by preserving variation across behavioural dimensions.

<p align="center">
<img src="assets/model.png" alt="Workflow Overview" width="700"/>
</p>

<p align="center">
  <em>CSLP-Elites model</em>
</p>

This section separates into 4 main sections:
- [Layout Configurations](#layout-configurations)
- [Constraints](#-constraints)
- [Objective Functions](#-objective-functions)
- [Behavioral Descriptors](#-behavioral-descriptors)
  
---
<h2 id="layout-configurations">🏗️ Layout Configurations</h2>

<p align="center">
<img src="assets/config.png" alt="Facility types">
</p>

### Facility selection range:

- **Minimum Facilities**: 3 (minimal operational site)
  - Always includes: `core`, `crane`, `storage`

- **Maximum Facilities**: 8 (complex multi-function site)
  - Includes: All 5 types + additional operational facilities

- **Default Configuration**: 6 facilities
  - Typical mix: `core`, `crane`, `storage`, `office`, `rest_area`, + 1 operational facility

```
Facility combination:

- If count ≥ 3:  Add [core, crane, storage]                     # Operational facilities
- If count ≥ 5:  Add [office, rest_area]                        # Worker facilities
- If count > 5:  Fill remaining with [core, storage, crane]     # Additional operational
- Finally: Shuffle order randomly (seed-controlled)
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

---
## 🚧 Constraints
These are feasibility requirements that all valid layouts must meet:

### 1) Boundary compliance
All facilities must remain within site boundaries with margin clearance.

<p align="center">
<img src="assets/constraint1.png" alt="Boundary compliance constraint" width="550">
</p>

$$C_1: \quad \forall i \in facilities, \quad \text{margin} \leq x_i, y_i \leq 1 - \text{margin}$$

where:
- $(x_i, y_i)$ = center position of facility $i$
- $\text{margin}$ = boundary clearance (default: 0.08)

**Violation measure:**

A layout violates this constraint if any facility extends beyond the boundary margins.

$$V_{boundary} = \sum_{i=1}^{n} \max\left(0, \text{margin} - x_i, x_i + \frac{w_i}{2} - 1 + \text{margin}, \text{margin} - y_i, y_i + \frac{h_i}{2} - 1 + \text{margin}\right)$$

### 2) No overlapping facilities
Facilities cannot physically overlap each other.

<p align="center">
<img src="assets/constraint2.png" alt="No overlap constraint" width="550">
</p>

$$C_2: \quad \forall i \neq j, \quad A_{\mathrm{overlap}}(f_i, f_j) = 0$$

where:
- $f_i, f_j$ = facility rectangles $i$ and $j$
- $A_{\mathrm{overlap}}(\cdot, \cdot)$ = 2D rectangular intersection area function

**Violation measure:**

A layout violates this constraint if any two facilities have overlapping areas.

$$V_{\mathrm{overlap}} = \sum_{i < j} A_{\mathrm{overlap}}(f_i, f_j)$$

---
## 🎯 Objective Functions
### 1) Safety compliance
Measures hazard prevention and worker protection.

<p align="center">
<img src="assets/objective1.png" alt="Objective function 1" width="550">
</p>

$$O_1 = 1 - \min\left(1, \frac{\sum_{j \in \text{workers}} P_{danger}(j)}{n_{workers}}\right)$$

**Crane danger penalty** for worker facility $j$:

$$P_{danger}(j) = \begin{cases}
0 & \text{if } d_j \geq r_{danger} \\
\left(\frac{r_{danger} - d_j}{r_{danger}}\right) \times 0.3 & \text{if } d_j < r_{danger}
\end{cases}$$

where:
- $d_j$ = distance from nearest crane to worker facility $j$
- $r_{danger} = 0.25$ (crane danger radius)
- $\text{workers} = \{\text{office}, \text{ rest area}\}$

**Interpretation**:
- $O_1 = 1.0$ → All workers outside danger zones (safest)
- $O_1 = 0.5$ → Moderate safety compliance
- $O_1 = 0.0$ → Workers directly in crane operation zones (unsafe)

**Function**: `calculate_safety_compliance(facilities, entrances, config)`
- Returns: `(safety_score, feasible_flag, violation_list)`

### 2) Operational efficiency
Optimizes material flows, equipment accessibility, and workflow support.

<p align="center">
<img src="assets/objective2.png" alt="Objective function 2" width="650">
</p>

$$O_2 = 0.4 \times E_{flow} + 0.4 \times E_{access} + 0.2 \times E_{sequence}$$

where:
- $$E_{flow}$$ = Material flow efficiency
- $$E_{access}$$ = Equipment accessibility
- $$E_{sequence}$$ = Work sequence support

**Overall Efficiency Interpretation**:
- $O_2 = 1.0$ → Optimal material flows, full crane coverage, convenient access
- $O_2 = 0.7$ → Good efficiency with minor inefficiencies
- $O_2 < 0.5$ → Poor logistics, inadequate coverage

**Function**: `calculate_operational_efficiency(facilities, entrances)`
- Returns: `efficiency_score ∈ [0, 1]`

#### a) Material flow efficiency
Minimizes transport distances for critical material flows:Minimizes transport distances for critical material flows:

$$E_{flow} = 1 - \frac{\bar{d}_{flow}}{d_{max}}$$

where:
- $\bar{d}_{flow}$ = average distance for critical material flows: (storage→core, crane→core, storage→crane)
- $d_{max} = \sqrt{2} \times 0.8$ (site diagonal)

#### b) Equipment accessibility
Measures crane coverage over work areas with range-based quality:

$$E_{access} = \frac{1}{n_{work}} \sum_{w=1}^{n_{work}} C_w$$

where:
- $n_{work}$ = number of work areas (operational facilities: core, storage)
- $C_w$ = coverage quality for work area $w$
- $d_w$ = distance from work area $w$ to nearest crane

Crane coverage quality for work area $w$:

$$C_w = \begin{cases} 
1.0 & \text{if } d_w \leq 0.25 \quad \text{(optimal range)} \\
1.0 - \frac{d_w - 0.25}{0.15} \times 0.4 & \text{if } 0.25 < d_w \leq 0.40 \quad \text{(acceptable range)} \\
0.0 & \text{if } d_w > 0.40 \quad \text{(out of range)}
\end{cases}$$

Multiple crane bonus (encourages redundancy):

$$C_w' = \min\left(1.0, C_w + 0.15 \times (n_{cranes} - 1)\right)$$

where $n_{cranes}$ = number of cranes covering work area $w$ (within acceptable range)

#### c) Work sequence support
Ensures worker facilities have convenient entrance access:

$$E_{sequence} = \frac{1}{n_{offices}} \sum_{o \in \text{offices}} \max\left(0, 1 - \frac{d_{o,entrance}}{0.4\sqrt{2}}\right)$$

where $d_{o,entrance}$ = distance from office $o$ to nearest entrance

### 3) Layout adaptability
Measures future flexibility, expansion capacity, and reconfiguration potential.

<p align="center">
<img src="assets/objective3.png" alt="Objective function 3" width="650">
</p>

$$O_3 = 0.4 \times A_{expansion} + 0.35 \times A_{redundancy} + 0.25 \times A_{reconfig}$$

where:
- $$A_{expansion}$$ = Expansion potential
- $$A_{redundancy}$$ = Route redundancy
- $$A_{reconfig}$$ = Reconfiguration ease

**Overall Adaptability Interpretation**:
- $O_3 = 1.0$ → Maximum flexibility, ample expansion space, easy reconfiguration
- $O_3 = 0.7$ → Good adaptability with some constraints
- $O_3 < 0.5$ → Rigid layout, limited future options

**Function**: `calculate_layout_adaptability(facilities, entrances, config)`
- Returns: `adaptability_score ∈ [0, 1]`

#### a) Expansion potential
Measures available free space for future facilities:

$$A_{expansion} = \frac{n_{available\_cells}}{n_{usable\_cells}}$$

where:
- Uses 10×10 grid overlay on site
- $n_{usable\_cells}$ = cells within boundary margins
- $n_{available\_cells}$ = usable cells not occupied by facilities

**Interpretation**: Higher score = more room for expansion

#### b) Route redundancy
Evaluates alternative path availability for critical facility pairs:

$$A_{redundancy} = \frac{1}{n_{pairs}} \sum_{p \in \text{keyPairs}} \frac{1}{1 + 10 \times \text{Var}(d_p)}$$

where:
- $\text{keyPairs} = \{(\text{office} \to \text{core}), (\text{storage} \to \text{core}), (\text{crane} \to \text{storage})\}$
- $\text{Var}(d_p)$ = variance of distances for multiple instances of pair type $p$

**Interpretation**: Low variance = multiple similar-length paths = good redundancy

#### c) Reconfiguration ease
Tests how easily facilities can be relocated to alternative positions:

$$A_{reconfig} = \frac{1}{n_{facilities}} \sum_{f \in \text{facilities}} \frac{n_{valid\_positions}}{20}$$

where:
- For each facility, 20 random alternative positions tested (within distance 0.5)
- $n_{valid\_positions}$ = positions that satisfy constraints

**Interpretation**: Higher score = more flexibility for future layout changes

---
## 🎨 Behavioral Descriptors
### 1) Spatial organization
Measures how facilities are distributed across the site:

<p align="center">
<img src="assets/bd1.png" alt="Behavioral descriptor 1" width="550">
</p>

$$BD_1 = \frac{\bar{d}_{centroid}}{B}$$

where:

$$\bar{d}_{centroid} = \frac{1}{n} \sum_{i=1}^{n} \|\mathbf{p}_i - \mathbf{c}\|$$

$$\mathbf{c} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{p}_i \quad \text{(global centroid)}$$

- $\mathbf{p}_i$ = center position of facility $i$
- $B = 0.50$ = normalization bound (maximum expected mean distance)

#### **Range**: [0, 1]
| Value | Interpretation | Layout Pattern |
|-------|---------------|----------------|
| **0.0 - 0.3** | Very compact | All facilities clustered near site center |
| **0.3 - 0.5** | Moderately compact | Facilities grouped with some spread |
| **0.5 - 0.7** | Moderately distributed | Facilities spread across regions |
| **0.7 - 1.0** | Highly spread | Facilities dispersed to site edges |

#### **Function**: 
- `calculate_compactness_vs_spread(facilities)` 
- Alias: `calculate_spatial_organization(facilities)`

### 2) Functional Integration
Measures the spatial relationship between worker facilities and operational zones:

<p align="center">
<img src="assets/bd2.png" alt="Behavioral descriptor 2" width="550">
</p>

$$BD_2 = \frac{\bar{d}_{separation}}{S}$$

where:

$$\bar{d}_{separation} = \frac{1}{n_{workers}} \sum_{w \in \text{workers}} \min_{o \in \text{operational}} \|\mathbf{p}_w - \mathbf{p}_o\|$$

- $\text{workers} = \{\text{office}, \text{rest area}\}$
- $\text{operational} = \{\text{core}, \text{storage}, \text{crane}\}$
- $S = 0.30$ = normalization bound (maximum reasonable separation)

#### **Range**: [0, 1]
| Value | Interpretation | Layout Pattern |
|-------|---------------|----------------|
| **0.0 - 0.3** | Highly integrated | Workers embedded within operational zones |
| **0.3 - 0.5** | Moderately integrated | Workers adjacent to operational areas |
| **0.5 - 0.7** | Moderately separated | Clear buffer between workers and operations |
| **0.7 - 1.0** | Strongly segregated | Workers isolated far from operations |

#### **Function**: 
- `calculate_worker_operational_separation(facilities)`

#### **Inverted Version** (for alternative interpretation):
$$BD_2' = 1 - BD_2$$

- Function: `calculate_functional_integration(facilities)`
- Higher $BD_2'$ = more integration (workers closer to operations)
