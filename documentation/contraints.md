# 🚧 Constraints
These are feasibility requirements that all valid layouts must meet:

## 1) Boundary compliance
All facilities must remain within site boundaries with margin clearance

$$C_1: \quad \forall i \in facilities, \quad \text{margin} \leq x_i, y_i \leq 1 - \text{margin}$$

where:
- $(x_i, y_i)$ = center position of facility $i$
- $\text{margin}$ = boundary clearance (default: 0.08)

**Violation measure:**

A layout violates this constraint if any facility extends beyond the boundary margins.

$$V_{boundary} = \sum_{i=1}^{n} \max\left(0, \text{margin} - x_i, x_i + \frac{w_i}{2} - 1 + \text{margin}, \text{margin} - y_i, y_i + \frac{h_i}{2} - 1 + \text{margin}\right)$$

## 2) No overlapping facilities
Facilities cannot physically overlap each other.

$$C_2: \quad \forall i \neq j, \quad A_{\mathrm{overlap}}(f_i, f_j) = 0$$

where:
- $f_i, f_j$ = facility rectangles $i$ and $j$
- $A_{\mathrm{overlap}}(\cdot, \cdot)$ = 2D rectangular intersection area function

**Violation measure:**

A layout violates this constraint if any two facilities have overlapping areas.

$$V_{\mathrm{overlap}} = \sum_{i < j} A_{\mathrm{overlap}}(f_i, f_j)$$
