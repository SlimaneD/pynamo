# PyGambit comparison

Experiment: 17 built-in examples and six additional degenerate/boundary games.
Installed PyGambit 16.7.0 in `/tmp/pynamo-gambit-env` (Python 3.12.1,
macOS ARM64). No production code or project dependencies were changed.

## Reproduce

From the repository root:

```bash
python3.12 -m venv /tmp/pynamo-gambit-env
/tmp/pynamo-gambit-env/bin/pip install -r benchmarks/gambit_requirements.txt
/tmp/pynamo-gambit-env/bin/python benchmarks/compare_gambit.py > benchmarks/gambit_results.json
```

Pass example names after the script path to run a subset. The JSON records
versions, profiles, validations, warnings, unmatched profiles and timings.
Timings are single-run observations, not controlled performance benchmarks:
pyNamo also computes dynamical stability, whereas Gambit only computes NE.

## Method

- Two-player games: `enummixed_solve(rational=True)` with decimal-string payoff
  conversion. This preserves the catalogue's supplied finite decimal values;
  it does not reconstruct a user's intended exact fractions from floats.
- Three-player games: `enumpoly_solve(use_strategic=True, maxregret=1e-10)`.
- Symmetric game A is converted to the bimatrix game (A, A.T); only profiles
  with equal player strategies are compared to pyNamo population states.
- pyNamo's asymmetric two-player matrices store the player's own action on
  rows, so its second matrix is transposed for Gambit.
- Current three-player dynamics use probability of tensor action 1, despite
  the public docstrings saying first strategy. The comparison follows the
  implementation, converting each coordinate p to (1-p, p).
- Independent verification explicitly enumerates opponents' action profiles,
  calculates every pure-action payoff, and checks probability feasibility and
  maximum unilateral gain <= 1e-7. It does not call pyNamo's Nash classifier.
- Duplicate Gambit profiles are removed with absolute tolerance 1e-7 before
  set comparison. Both raw counts and deduplicated profiles are retained.

## Results

All 23 cases completed. All 111 deduplicated Gambit profiles passed independent
Nash verification; maximum observed regret was 4.44e-16. All returned pyNamo
Nash profiles also passed. Its strict-Nash labels agreed with independent
pure-action payoff comparisons.

All 12 catalogue games for which pyNamo reported `degenerate=False` matched.
Coordination Cube also matched, despite being flagged degenerate because its
replicator rest-point set contains continua (not necessarily Nash continua).
The other four catalogue games differed:

| Game | pyNamo NE | Gambit symmetric NE profiles |
|---|---:|---:|
| Hawk-Dove-Retaliator | 2 | 3 |
| Repeated PD: TFT, ALLC, ALLD | 3 | 4 |
| RPS with a Twin | 0 | 2 |
| Ownership Game | 2 | 4 |

Every pyNamo NE appeared in the applicable Gambit output in all 23 cases.
For Ownership Game, Gambit returned 17 raw profiles, including 7 symmetric
entries; deduplication leaves 14 profiles, including 4 symmetric profiles.

These counts are **returned points, not counts of the entire equilibrium set**.
Seven explicitly checked interior points of equilibrium continua were absent
from both lists: zero symmetric, zero bimatrix, zero three-player, duplicate
coordination, RPS with a Twin, Hawk-Dove-Retaliator and repeated PD.
For example, RPS with a Twin has symmetric NE
(1/3, 1/3, t, 1/3-t), 0 <= t <= 1/3. Gambit's filtered list contains the two
endpoints, while pyNamo contains none; neither list contains t=1/6.
This does not invalidate endpoint enumeration, but a wrapper must represent
families or clearly state that it returns representatives/extreme points.

All three zero-payoff cases incorrectly have `degenerate=False` in pyNamo,
even though every state is stationary and Nash. SymPy's empty result for an
identically zero equation system is not recognized as a continuum.

## ESS control

For A = [[0,1,0],[0,0,0],[-1,0,0]], both solvers find p=(1,0,0) as Nash;
pyNamo incorrectly reports ESS=False. Any mutant with positive third component
loses against p. For a tied mutant q=(1-t,t,0), t>0,
p.T A q - q.T A q = t^2 > 0, establishing ESS analytically.
Changing A[0,1] to -1 yields a non-ESS control: that expression becomes -t^2.
pyNamo reports False for this control, correctly. Gambit NE enumeration does
not perform the ESS classification and therefore does not fix this defect.

## Historical recommendation (superseded)

PyGambit is a promising NE backend: it recovered additional valid boundary
points and reproduced the existing nondegenerate catalogue results. Adopt only
with an adapter that handles payoff conventions, deduplication, symmetric
population equilibria, and explicit continuum semantics. Keep replicator rest
points separate, independently verify solver outputs, and retain/fix an ESS
classifier. Pin versions and solver options for reproducible research.

This finite comparison is evidence for these inputs, not a completeness proof
for arbitrary games, a validation of Gambit's family reconstruction, or a
cross-platform reproducibility study. No family reconstruction was implemented.

## Current project scope

The application annotates only detected replicator rest points. The runtime
backend proposal above was withdrawn: PyGambit remains an optional independent
comparison tool, not a production dependency. No Nash family reconstruction or
extra Nash points are added to analysis. The ESS and zero-game fixes remain.
Saved JSON records the pre-fix experiment; rerunning the script against current
code will reflect those fixes, so pyNamo ESS and degeneracy fields can differ.
