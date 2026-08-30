# pyNamo

Tools for plotting and analyzing replicator dynamics of evolutionary games. The
project draws phase portraits on simplices, highlights equilibria, and ships a
small catalogue of example games to try immediately.

## Features

- Replicator dynamics for asymmetric 2-player/2-strategy, symmetric 2-player/3-strategy, symmetric 2-player/4-strategy, and asymmetric 3-player/2-strategy games.
- Preloaded game catalogue with attribute lookup and string lookup.
- Matplotlib plots of trajectories, equilibria, speed fields, and vector fields on the appropriate simplex.
- Equilibrium tables with stability status, Nash, strict Nash, ESS where applicable, eigenvalues, and eigenvectors.
- Optional Jupyter widgets to explore trajectories interactively without touching code.

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `matplotlib`, `sympy`, `pandas`
- Optional for notebooks: `jupyter`/`ipykernel`, `ipywidgets`

Install the dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib sympy pandas ipywidgets
```

For development tests:

```bash
pip install pytest
python -m pytest -q
```

## Interactive notebook widget

If you prefer sliders and dropdowns inside a notebook:

```python
from interactive import launch_replicator_widget
launch_replicator_widget()
```

Pick a game family and example, set the time horizon and number of trajectories,
toggle speed/vector fields, and display payoff data and equilibrium analysis.

## Plotting a Game

The main plotting interface is `drawer.plot_game`:

```python
import drawer
import examples

game = examples.games.matching_pennies
fig, ax = drawer.plot_game(game)
```

You can retrieve built-in games by attribute or by name:

```python
game = examples.games.battle_of_the_sexes
same_game = examples.games("battle_of_the_sexes")
examples.games.by_class("2P2S")
```

Override plot parameters directly:

```python
fig, ax = drawer.plot_game(
    game,
    random_state=1,
    tmax=60,
    trajectory_arrows=[],
    trajectory_color="black",
    show_vector_field=True,
    vector_grid=20,
    speed_levels=20,
    equilibrium_size=100,
)
```

Use one color per trajectory by passing a list:

```python
fig, ax = drawer.plot_game(
    examples.games.cyclic_mismatching_pennies,
    starts=[[0.52, 0.50, 0.48], [0.70, 0.45, 0.35]],
    trajectory_color=["tab:blue", "tab:orange"],
    trajectory_arrows=[],
    tmax=1000,
)
```

Vector fields are available for both 2D and 3D game classes. Sparse 3D vector
fields can be useful for exploration, but they are often hard to read in static
figures; trajectories are usually clearer for publication graphics.

## Equilibrium Analysis Table

For notebooks, use `analysis.equilibrium_table` to inspect isolated equilibria without plotting:

```python
import analysis
import examples

game = examples.games.good_rps
analysis.equilibrium_table(game)
```

If you do not want to depend on `pandas`, use `analysis.analyze_equilibria(game).to_rows()`.

Position vectors follow the order in which strategies are supplied when the
game is defined. For symmetric 2-player games, `Position = [0.2, 0.3, 0.5]`
means probabilities of the first, second, and third entries in
`strategy_labels`.

For asymmetric 2-strategy games, each coordinate is the probability that the
corresponding player uses their first listed strategy. For example, if
`player_strategy_labels=[["Fight", "Flee"], ["Aggressive", "Cautious"]]`, then
`Position = [0.7, 0.4]` means:

- Prey: `Pr(Fight) = 0.7`
- Predator: `Pr(Aggressive) = 0.4`

For quick access to static equilibrium concepts:

```python
analysis.find_nash(game)
analysis.find_strict_nash(game)
analysis.find_ess(game)
```

`find_ess` returns ESS only for symmetric games.

## Equilibrium Classification Caveats

Stability is classified from the linearization restricted to admissible
directions in the simplex. This matters at boundaries because outward
perturbations are not valid evolutionary deviations.

Some equilibria are non-hyperbolic or belong to degenerate equilibrium sets. In
these cases pyNamo emits warnings rather than pretending to prove more than it
has. Non-isolated equilibrium manifolds are not plotted automatically; isolated
equilibria are still shown when they can be identified.

The category `unstable` means linearization proves the equilibrium is not
stable, but does not always distinguish source from saddle. In plots, unstable
equilibria are drawn with the source color for visual compatibility.

## Useful files

- `game.py` – core `Game` class and game-class inference.
- `examples.py` – catalogue of predefined games.
- `dynamics.py` – replicator vector fields and rest-point computation.
- `analysis.py` – structured equilibrium analysis tables.
- `drawer.py` – plotting helpers and the high-level `plot_game` interface.
- `interactive.py` – Jupyter widget front-end.

Provide new payoff matrices directly with `game.Game`, or add curated examples
in `examples.py`.
