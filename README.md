# pyNamo

Tools for plotting replicator dynamics of evolutionary games. The project draws phase portraits on simplices (2D and 3D), highlights equilibria, and ships a small catalogue of example games to try immediately.

## Features

- Replicator dynamics for asymmetric 2-player/2-strategy, symmetric 2-player/3-strategy, symmetric 2-player/4-strategy, and 3-population/2-strategy games.
- Preloaded game catalogue (Matching Pennies, Rock–Paper–Scissors variants, Hawk–Dove, Skyrms 1992, and more).
- Matplotlib plots of trajectories, equilibria, and speed fields on the appropriate simplex.
- Optional Jupyter widgets to explore trajectories interactively without touching code.

## Requirements

- Python 3.9+
- `numpy`, `matplotlib`, `sympy`
- Optional for notebooks: `jupyter`/`ipykernel`, `ipywidgets`
- Optional for equilibrium tables: `pandas`

Install the dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install numpy matplotlib sympy ipywidgets pandas
```

## Interactive notebook widget

If you prefer sliders and dropdowns inside a notebook:

```python
from interactive import launch_replicator_widget
launch_replicator_widget()
```

Pick a game family and example, set the time horizon and number of trajectories, and the widget will render the corresponding simplex with trajectories and equilibria.

## Plotting a Game

The main plotting interface is `drawer.plot_game`:

```python
import drawer
import parameters as param

game = param.available_games("2P2S")[1]
fig, ax = drawer.plot_game(game)
```

Override plot parameters directly:

```python
fig, ax = drawer.plot_game(
    game,
    random_state=1,
    tmax=60,
    trajectory_arrows=[],
    show_vector_field=True,
    vector_grid=20,
    speed_levels=20,
    equilibrium_size=100,
)
```

Vector fields are available for both 2D and 3D game classes. Sparse 3D vector
fields can be useful for exploration, but they are often hard to read in static
figures; trajectories are usually clearer for publication graphics.

## Equilibrium Analysis Table

For notebooks, use `analysis.equilibrium_table` to inspect isolated equilibria without plotting:

```python
import analysis
import parameters as param

game = param.available_games("2P3S")[1]
analysis.equilibrium_table(game)
```

If you do not want to depend on `pandas`, use `analysis.analyze_equilibria(game).to_rows()`.

## Useful files

- `parameters.py` – catalogue of predefined games and plotting settings.
- `analysis.py` – structured equilibrium analysis tables.
- `drawer.py` – plotting helpers for simplices, trajectories, and equilibria.
- `interactive.py` – Jupyter widget front-end.

Additional example plots live in `images/`. Provide new payoff matrices or tweak parameters in `parameters.py` to extend the catalogue.
