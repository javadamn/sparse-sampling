# Reproduction check for the revised manuscript

This report records the validated outputs used in the revised manuscript.

| Revised statement | Rerun result |
|---|---:|
| Joined HMP2 source cohort | 106 subjects, 546 samples |
| Primary cohort | 4 subjects |
| Samples and transitions per primary subject | median 8 and 7 |
| Retained observation gaps | 14–124 days, pooled median 41 |
| Negative apparent return coefficients | 3/4 |
| Sign-stable paired-bootstrap ranges | 0/4 |
| Robust empirical multistability | 0/4 subjects |
| Selected-seed toy minima | {('double_well', 'dense'): 2, ('double_well', 'sparse'): 1, ('ou', 'dense'): 1, ('ou', 'sparse'): 1} |
| Double-well detection at sigma=0.6, stride=225 | 0.005 across 200 runs |
| OU spurious second minima across the full grid | 0/5000 |
| ARX candidates screened | 8 |

## Provenance checks

- Raw-to-analysis reconstruction: 132/132 compared columns match.
- Maximum numeric reconstruction difference: 0.
- Primary analysis IDs: C3032, C3037, H4001, H4004; the code requires exact agreement with the seven-transition rule.
- Bootstrap replicates requested per subject: 1000.
- Toy sensitivity trajectories per cell: 200.

## Interpretation

The validated package supports claims about detection limits and study design. It does not support biological resilience, subject ranking, or metabolite-modulator claims in the primary empirical cohort.
