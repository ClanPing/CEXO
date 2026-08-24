"""
Update sensitivity analysis section to use all individual trade-off plots
"""

# Read the file
with open('experiment.md', 'r', encoding='utf-8') as f:
    content = f.read()

# New sensitivity section with all individual plots
new_sensitivity_text = """3.  *Sensitivity analysis*

The sensitivity analysis evaluated the effect of 8 key hyperparameters
on the overall performance of CEXO with autoencoder-learned behavioral
descriptors. Each parameter was perturbed across its operational range
while all others remained fixed at baseline values (6 facilities, 20 × 20
grid, 15K iterations, baseline coverage 99.75%). The tested parameters
included site-specific constraints (boundary margin, crane safety distance,
entrance clearance), autoencoder hyperparameters (pretrain iterations,
training frequency, latent dimensions), and algorithm parameters (initial
population size, Pareto front size). Relative changes in performance
metrics were computed with respect to the baseline configuration.

Figures 7-14 present the multidimensional trade-off analysis for each
parameter, showing how coverage, safety, efficiency, and adaptability
change relative to baseline. All 8 parameters exhibited low sensitivity
for coverage (&lt;10% maximum absolute change), with changes ranging from
0.00% to 5.76%. This demonstrates that CEXO with learned behavioral
descriptors remains remarkably robust to hyperparameter variations once
near-complete coverage (98-99%) is achieved.

Site-specific constraint parameters show varying effects: boundary margin
(Figure 7) exhibits the most dramatic impact on adaptability (29.7%
decrease at extreme values) while maintaining coverage within 5.5%,
reflecting the trade-off between spatial flexibility and constraint
tightness. Crane safety distance (Figure 8) demonstrates minimal effects
across all objectives (&lt;4% variation), confirming robust performance
regardless of safety buffer sizing. Entrance clearance (Figure 9) displays
a non-monotonic coverage pattern with a 5.8% drop at intermediate values,
suggesting complex interactions between entrance constraints and
autoencoder descriptor learning.

<img src="media/image7.png" style="width:6.5in;height:4in" />

**Figure 7.** Parameter sensitivity: Boundary margin.

<img src="media/image8.png" style="width:6.5in;height:4in" />

**Figure 8.** Parameter sensitivity: Crane safety distance.

<img src="media/image9.png" style="width:6.5in;height:4in" />

**Figure 9.** Parameter sensitivity: Entrance clearance.

Autoencoder hyperparameters reveal the importance of proper descriptor
learning configuration: pretrain iterations (Figure 10) shows 5.8% coverage
degradation at extended pretraining (10,000 iterations), indicating that
excessive pretraining before switching to learned descriptors may hinder
adaptation. Training frequency (Figure 11) exhibits minimal coverage
sensitivity (&lt;0.5%) but notable adaptability variation (1.3%),
suggesting that periodic retraining intervals have limited impact on
behavioral space exploration. Latent dimensions (Figure 12) demonstrates
optimal performance at the baseline 2D configuration, with 3D-6D spaces
showing 2-3% coverage reduction, confirming that 2D latent space
sufficiently captures layout diversity for this problem scale.

<img src="media/image10.png" style="width:6.5in;height:4in" />

**Figure 10.** Parameter sensitivity: Pretrain iterations.

<img src="media/image11.png" style="width:6.5in;height:4in" />

**Figure 11.** Parameter sensitivity: Training frequency.

<img src="media/image12.png" style="width:6.5in;height:4in" />

**Figure 12.** Parameter sensitivity: Latent dimensions.

Algorithm parameters show stable performance: initial population size
(Figure 13) maintains coverage within 1.8% across tested values (100-1000),
indicating that the baseline 500 population provides sufficient diversity
without requiring larger computational investment. Pareto front size
(Figure 14) exhibits zero sensitivity—all tested values (4, 8, 12, 16, 20)
produce identical 99.75% coverage—demonstrating that once learned
descriptors achieve near-complete coverage, the number of solutions stored
per cell does not affect behavioral space exploration.

<img src="media/image13.png" style="width:6.5in;height:4in" />

**Figure 13.** Parameter sensitivity: Initial population size.

<img src="media/image14.png" style="width:6.5in;height:4in" />

**Figure 14.** Parameter sensitivity: Pareto front size.

Overall, the consistently low coverage sensitivity across all parameters
(&lt;6% maximum change) indicates that CEXO maintains stable performance
once autoencoder-learned descriptors achieve near-complete behavioral space
coverage. Unlike traditional hand-crafted descriptors where parameter tuning
could increase coverage from lower baselines (e.g., 58% → 75%), the 99.75%
baseline with learned descriptors means sensitivity analysis primarily
reveals coverage degradation patterns and objective trade-offs rather than
optimization opportunities. Safety remains robustly high (&lt;2% variation)
across all parameters, while adaptability shows the most dramatic variations
(up to 29.7%), reflecting the inherent trade-off between spatial flexibility
and constraint satisfaction.
"""

# Find the start and end of sensitivity section
start_marker = "3.  *Sensitivity analysis*"
end_marker = "4.  *Comparative performance*"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print("Error: Could not find sensitivity section markers")
    exit(1)

# Replace the section
new_content = (
    content[:start_idx] + 
    new_sensitivity_text + "\n\n" +
    content[end_idx:]
)

# Write back
with open('experiment.md', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Successfully updated sensitivity analysis section with all individual trade-off plots")
print(f"Old section length: {end_idx - start_idx} characters")
print(f"New section length: {len(new_sensitivity_text)} characters")
print("\nNew structure:")
print("- Figure 7: Boundary margin")
print("- Figure 8: Crane safety distance")
print("- Figure 9: Entrance clearance")
print("- Figure 10: Pretrain iterations")
print("- Figure 11: Training frequency")
print("- Figure 12: Latent dimensions")
print("- Figure 13: Initial population size")
print("- Figure 14: Pareto front size")
