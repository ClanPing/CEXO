# CEXO Project Information

CEXO aims to **generate diverse and high-performing construction site layouts** that balance safety, operational efficiency, and adaptability. The framework combines objective functions, behavioral descriptors, quality-diversity archiving, Pareto selection, and genetic search operators.

This page documents the implementation-level formulation used in the repository. It does not introduce or modify literature references; it describes how layouts are generated, evaluated, and categorized in CEXO.

- **Objective Functions** measure how well a layout performs against practical goals such as safety clearance, material flow, equipment accessibility, and future adaptability.
- **Behavioral Descriptors (BDs)** describe how layouts differ spatially and functionally, allowing the archive to preserve multiple layout patterns instead of only one best solution.
- **Learned Descriptors** are available in CEXO v2 through an autoencoder. The hand-crafted descriptors documented below remain useful for interpretation, baselines, and `--no-learned` runs.

<p align="center">
<img src="assets/model.png" alt="Workflow Overview" width="700"/>
</p>

<p align="center">
  <em>CEXO model</em>
</p>

This page is organised into four sections:

- [Layout Configurations](#layout-configurations)
- [Constraints](#constraints)
- [Objective Functions](#objective-functions)
- [Behavioral Descriptors](#behavioral-descriptors)

---

<h2 id="layout-configurations">🏗️ Layout Configurations</h2>

<p align="center">
<img src="assets/config.png" alt="Facility types">
</p>

## Facility Selection Range

- **Minimum Facilities**: 3
  - Always includes: `core`, `crane`, `storage`

- **Maximum Facilities**: 8 for the standard synthetic benchmark
  - Includes all five facility types plus additional operational facilities

- **Default Standard Configuration**: 6 facilities
  - Typical mix: `core`, `crane`, `storage`, `office`, `rest_area`, plus one additional operational facility

The standard facility generator follows this pattern:

```text
If count >= 3:  add [core, crane, storage]
If count >= 5:  add [office, rest_area]
If count > 5:   fill remaining slots with [core, storage, crane]
Finally:        shuffle order randomly using the selected seed
```

**Example generations** using the default seed:

| Count | Facility Mix | Breakdown |
|-------|--------------|-----------|
| 3 | `['storage', 'core', 'crane']` | 3 operational only |
| 4 | `['storage', 'core', 'crane', 'crane']` | 3 operational + 1 extra operational |
| 5 | `['storage', 'core', 'crane', 'rest_area', 'office']` | 3 operational + 2 worker |
| 6 | `['storage', 'core', 'crane', 'rest_area', 'office', 'storage']` | Balanced + 1 extra operational |
| 7 | `['storage', 'core', 'crane', 'rest_area', 'office', 'storage', 'crane']` | Balanced + 2 extra operational |
| 8 | `['storage', 'core', 'crane', 'rest_area', 'office', 'storage', 'crane', 'storage']` | Full standard site |

## Bulleen Practical Case

The Bulleen case study uses the same CEXO methodology with a more constrained site setup:

- An approximate irregular boundary polygon
- Three fixed entrance/access points
- Thin road/access corridor exclusion zones
- A fixed or sampled practical facility mix

The detailed Bulleen setup is documented in [case_studies/bulleen](case_studies/bulleen/README.md).

---

<h2 id="constraints">🚧 Constraints</h2>

These feasibility requirements are checked during layout evaluation and repair.

## 1. Boundary Compliance

All facilities must remain within the site boundary. For the standard rectangular site, the default boundary margin is:

$$m = 0.03$$

The safety objective applies a small additional checking margin:

$$m_{\mathrm{eff}} = m + 0.01$$

For a rectangular site, each facility must remain inside:

$$m_{\mathrm{eff}} \leq x_i - \frac{w_i}{2}, \quad x_i + \frac{w_i}{2} \leq 1 - m_{\mathrm{eff}}$$

$$m_{\mathrm{eff}} \leq y_i - \frac{d_i}{2}, \quad y_i + \frac{d_i}{2} \leq 1 - m_{\mathrm{eff}}$$

where:

- $(x_i, y_i)$ = centre position of facility $i$
- $w_i, d_i$ = facility width and depth
- $m_{\mathrm{eff}}$ = effective boundary checking margin

For polygonal sites such as Bulleen, the facility footprint must remain inside the polygonal boundary and outside configured exclusion zones.

<p align="center">
<img src="assets/constraint1.png" alt="Boundary compliance constraint" width="550">
</p>

## 2. No Overlapping Facilities

Facilities cannot physically overlap each other.

$$C_2: \quad \forall i \neq j, \quad A_{\mathrm{overlap}}(f_i, f_j) = 0$$

where:

- $f_i, f_j$ = facility rectangles
- $A_{\mathrm{overlap}}(\cdot, \cdot)$ = 2D rectangular intersection area

The implementation scores overlap using both the number of overlapping pairs and the total overlap area, so small violations receive a smaller penalty than severe clashes.

<p align="center">
<img src="assets/constraint2.png" alt="No overlap constraint" width="550">
</p>

## 3. Safety Clearances

CEXO checks three practical safety relationships:

- Crane-to-crane clearance to reduce collision risk
- Crane danger zones around worker facilities
- Entrance/access clearance around all facilities

The current default values are:

| Parameter | Default |
|-----------|---------|
| Crane danger radius | `0.14` |
| Minimum crane-to-crane clearance | `2 x danger_radius = 0.28` |
| Entrance clearance | `0.08` |
| Facility clearance buffer | `0.006` |

---

<h2 id="objective-functions">🎯 Objective Functions</h2>

CEXO evaluates each layout using three objective scores:

$$\mathbf{O} = (O_1, O_2, O_3)$$

where:

- $O_1$ = safety and constraint compliance
- $O_2$ = operational efficiency
- $O_3$ = layout adaptability

Each score is clipped to $[0, 1]$, where higher is better.

## 1. Safety and Constraint Compliance

Safety combines boundary compliance, overlap compliance, and critical safety clearance:

$$O_1 = 0.4 C_{\mathrm{boundary}} + 0.3 C_{\mathrm{overlap}} + 0.3 C_{\mathrm{safety}}$$

The safety component includes crane clearance, crane danger-zone checks, and entrance/access clearance. For worker facility $j$ near a crane:

$$P_{\mathrm{danger}}(j) =
\begin{cases}
0 & \text{if } d_j \geq r_{\mathrm{danger}} \\
\frac{r_{\mathrm{danger}} - d_j}{r_{\mathrm{danger}}} & \text{if } d_j < r_{\mathrm{danger}}
\end{cases}$$

where:

- $d_j$ = distance from worker facility $j$ to the nearest crane
- $r_{\mathrm{danger}} = 0.14$
- worker facilities are `office` and `rest_area`

<p align="center">
<img src="assets/objective1.png" alt="Objective function 1" width="550">
</p>

**Function**: `calculate_safety_compliance(facilities, entrances, config)`

Returns: `(safety_score, feasible_flag, violation_list)`

## 2. Operational Efficiency

Operational efficiency measures material movement, crane coverage, access, and worker-support clustering:

$$O_2 =
0.25 E_{\mathrm{flow}}
+ 0.25 E_{\mathrm{access}}
+ 0.20 E_{\mathrm{crane-core}}
+ 0.15 E_{\mathrm{entrance}}
+ 0.15 E_{\mathrm{worker-cluster}}$$

where:

- $E_{\mathrm{flow}}$ = material flow efficiency
- $E_{\mathrm{access}}$ = crane/equipment accessibility over work areas
- $E_{\mathrm{crane-core}}$ = crane operating coverage of core work zones
- $E_{\mathrm{entrance}}$ = office access to entrances
- $E_{\mathrm{worker-cluster}}$ = clustering quality of office and rest modules

<p align="center">
<img src="assets/objective2.png" alt="Objective function 2" width="650">
</p>

### Material Flow Efficiency

Critical material flows are:

- `storage -> core`
- `crane -> core`
- `storage -> crane`

$$E_{\mathrm{flow}} = 1 - \frac{\bar{d}_{\mathrm{flow}}}{d_{\max}}$$

where:

- $\bar{d}_{\mathrm{flow}}$ = average critical-flow distance
- $d_{\max} = \sqrt{2} \times 0.8$

### Equipment Accessibility

Crane accessibility uses the current crane specification:

| Parameter | Default |
|-----------|---------|
| Optimal reach | `0.20` |
| Operating radius | `0.26` |

For a work area $w$:

$$C_w =
\begin{cases}
1.0 & \text{if } d_w \leq r_{\mathrm{optimal}} \\
1.0 - 0.4\frac{d_w-r_{\mathrm{optimal}}}{r_{\mathrm{operating}}-r_{\mathrm{optimal}}} & \text{if } r_{\mathrm{optimal}} < d_w \leq r_{\mathrm{operating}} \\
0.0 & \text{if } d_w > r_{\mathrm{operating}}
\end{cases}$$

Multiple cranes can add a redundancy bonus when more than one crane provides meaningful coverage.

### Work Sequence and Worker Support

Entrance access rewards offices that are close to entrances:

$$E_{\mathrm{entrance}} =
\frac{1}{n_{\mathrm{offices}}}
\sum_{o \in \mathrm{offices}}
\max\left(0, 1 - \frac{d_{o,\mathrm{entrance}}}{0.4\sqrt{2}}\right)$$

Worker-support clustering rewards offices and rest areas that form a coherent support zone rather than being scattered without functional relationship.

**Function**: `calculate_operational_efficiency(facilities, entrances)`

Returns: `efficiency_score ∈ [0, 1]`

## 3. Layout Adaptability

Layout adaptability measures future flexibility, expansion capacity, and reconfiguration potential:

$$O_3 = 0.4 A_{\mathrm{expansion}} + 0.35 A_{\mathrm{redundancy}} + 0.25 A_{\mathrm{reconfig}}$$

where:

- $A_{\mathrm{expansion}}$ = expansion potential
- $A_{\mathrm{redundancy}}$ = route redundancy
- $A_{\mathrm{reconfig}}$ = reconfiguration ease

<p align="center">
<img src="assets/objective3.png" alt="Objective function 3" width="650">
</p>

### Expansion Potential

Expansion potential measures available space on a 25 x 25 grid:

$$A_{\mathrm{expansion}} = \frac{n_{\mathrm{available\_cells}}}{n_{\mathrm{usable\_cells}}}$$

where:

- $n_{\mathrm{usable\_cells}}$ = grid cells inside the boundary margin
- $n_{\mathrm{available\_cells}}$ = usable cells not occupied by facilities

### Route Redundancy

Route redundancy evaluates alternative path availability for key facility pairs:

$$A_{\mathrm{redundancy}} =
\frac{1}{n_{\mathrm{pairs}}}
\sum_{p \in \mathrm{keyPairs}}
\frac{1}{1 + 10 \times \mathrm{Var}(d_p)}$$

where:

$$\mathrm{keyPairs} =
\{(\mathrm{office}, \mathrm{core}),(\mathrm{storage}, \mathrm{core}),(\mathrm{crane}, \mathrm{storage})\}$$

Low distance variance means multiple routes or facility relationships have similar lengths, which supports redundancy.

### Reconfiguration Ease

Reconfiguration ease tests random alternative positions for each facility:

$$A_{\mathrm{reconfig}} =
\frac{1}{n_{\mathrm{facilities}}}
\sum_{f \in \mathrm{facilities}}
\frac{n_{\mathrm{valid\_positions}}}{20}$$

where:

- 20 candidate relocation positions are tested per facility
- A relocation is valid if it remains inside the site boundary, avoids overlap, and stays within a reasonable relocation distance

**Function**: `calculate_layout_adaptability(facilities, entrances, config)`

Returns: `adaptability_score ∈ [0, 1]`

---

<h2 id="behavioral-descriptors">🎨 Behavioral Descriptors</h2>

CEXO supports two descriptor modes:

- **Learned descriptors**: the default v2 mode, using an autoencoder latent representation of generated layouts
- **Hand-crafted descriptors**: used for `--no-learned`, baseline comparison, compatibility, and interpretation

The hand-crafted descriptors are:

## 1. Same-Type Module Clustering vs Dispersion

BD1 measures whether repeated modules of the same type are clustered together or dispersed across the site:

$$BD_1 = \frac{\bar{d}_{\mathrm{nearest\_same\_type}}}{0.16}$$

where:

- $\bar{d}_{\mathrm{nearest\_same\_type}}$ = mean nearest-neighbour distance among modules of the same type
- $BD_1 = 0$ indicates same-type modules are clustered
- $BD_1 = 1$ indicates same-type modules are dispersed

When no repeated facility types exist, CEXO falls back to the older centroid-spread descriptor:

$$BD_1 = \frac{\bar{d}_{\mathrm{centroid}}}{0.50}$$

<p align="center">
<img src="assets/bd1.png" alt="Behavioral descriptor 1" width="550">
</p>

| Value | Interpretation | Layout Pattern |
|-------|----------------|----------------|
| 0.0 - 0.3 | Strongly clustered | Same-type modules placed near each other |
| 0.3 - 0.5 | Moderately clustered | Same-type modules remain near local groups |
| 0.5 - 0.7 | Moderately dispersed | Same-type modules spread across several regions |
| 0.7 - 1.0 | Highly dispersed | Same-type modules distributed widely |

**Function**: `calculate_compactness_vs_spread(facilities)`

Compatibility alias: `calculate_spatial_organization(facilities)`

## 2. Worker-Operational Separation

BD2 measures the spatial relationship between worker facilities and operational zones:

$$BD_2 =
\frac{
0.6 \bar{d}_{\mathrm{nearest\_operational}}
+ 0.4 d_{\mathrm{centroid\_separation}}
}{0.28}$$

where:

- worker facilities are `office` and `rest_area`
- operational facilities are `core`, `storage`, and `crane`
- $\bar{d}_{\mathrm{nearest\_operational}}$ = average nearest operational distance for worker modules
- $d_{\mathrm{centroid\_separation}}$ = distance between worker and operational centroids

<p align="center">
<img src="assets/bd2.png" alt="Behavioral descriptor 2" width="550">
</p>

| Value | Interpretation | Layout Pattern |
|-------|----------------|----------------|
| 0.0 - 0.3 | Highly embedded | Worker modules close to operational zones |
| 0.3 - 0.5 | Moderately embedded | Worker modules adjacent to operations |
| 0.5 - 0.7 | Moderately separated | Clear buffer between workers and operations |
| 0.7 - 1.0 | Strongly segregated | Worker modules isolated from operational zones |

**Function**: `calculate_worker_operational_separation(facilities)`

Compatibility alias: `calculate_functional_integration(facilities)`

## Learned Descriptor Mode

When learned descriptors are enabled, CEXO trains an autoencoder on generated layouts and uses the two-dimensional latent representation as the behavioral space. This allows the archive to adapt its categories to the layouts actually being generated, rather than relying only on manually selected descriptor formulas.

The learned descriptor workflow is:

1. Generate an unbiased training pool of layouts.
2. Train the autoencoder on encoded facility and entrance geometry.
3. Convert latent coordinates into two normalized descriptors in `[0, 1]`.
4. Build and update the MAP-Elites archive using the learned descriptor coordinates.

This preserves the same objective functions and genetic search process while changing how behavioral diversity is represented.
