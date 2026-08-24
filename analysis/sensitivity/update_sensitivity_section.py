"""
Update sensitivity analysis section in experiment.md with new results
"""

# Read the file
with open('experiment.md', 'r', encoding='utf-8') as f:
    content = f.read()

# New sensitivity section
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

Figure 7 ranks the parameters according to their maximum absolute
coverage change from baseline. All 8 parameters exhibited low sensitivity
(&lt;10%), with maximum changes ranging from 0.25% to 5.75%. This
demonstrates that CEXO with learned behavioral descriptors remains
remarkably robust to hyperparameter variations. Entrance clearance showed
the highest sensitivity (5.75% maximum change), followed by pretrain
iterations and boundary margin (5.50-5.75%), while training frequency and
Pareto front size showed minimal effects (&lt;0.50%). The consistently low
sensitivity across all parameters indicates that once autoencoder-learned
descriptors achieve near-complete coverage (98-99%), the algorithm
maintains stable performance regardless of moderate parameter adjustments.

<img src="media/image8.png" style="width:5.8005in;height:3.54197in" alt="A graph with green and orange bars AI-generated content may be incorrect." />

**Figure 7.** Parameter sensitivity ranking via tornado plot.

Figure 8 examines the multidimensional trade-offs for the most sensitive
parameter (entrance clearance). When entrance clearance varies from 0.10
to 0.20, coverage exhibits a non-monotonic pattern with a notable 5.75%
drop at intermediate values (0.125), recovering to near-baseline levels
at both extremes. This U-shaped response suggests that entrance clearance
interacts with autoencoder descriptor learning—moderate clearances may
fragment the behavioral space differently than extreme values. Safety
remains stable across all tested values (&lt;2% variation), confirming
robust constraint satisfaction. Efficiency shows modest improvements
(2-3%) at larger clearances due to reduced spatial conflicts. Adaptability
demonstrates the most dramatic variation, with decreases up to 28% at
larger margins reflecting reduced spatial flexibility as entrance
constraints tighten. Unlike traditional hand-crafted descriptors where
parameter changes produced coverage increases from lower baselines, the
99.75% baseline with learned descriptors means sensitivity analysis
primarily reveals coverage degradation patterns and objective trade-offs
rather than optimization opportunities.

<img src="media/image9.png" style="width:6.26597in;height:3.47075in" />

**Figure 8.** Relative change based on varying entrance clearance.
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

print("Successfully updated sensitivity analysis section in experiment.md")
print(f"Old section length: {end_idx - start_idx} characters")
print(f"New section length: {len(new_sensitivity_text)} characters")
