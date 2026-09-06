# pyNamo-EGT

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SlimaneD/pynamo/master?filepath=tutorial.ipynb)

pyNamo-EGT is a Python package for plotting and analyzing replicator dynamics
in evolutionary games. It focuses on game classes whose state spaces can be
visualized directly, producing phase portraits on simplices with trajectories,
speed fields, vector fields, equilibria, and stability information.

The package is designed for researchers, teachers, and students who want clear,
publication-quality diagrams of theoretical phase portraits. Its high-level
interface is built for Jupyter notebooks and produces informative figures with
minimal code, while still exposing fine-grained controls for plotting details.

## Start Here

There are three main ways to try pyNamo-EGT.

**1. Full interactive tutorial in Binder**

Use Binder if you want the closest experience to a local Jupyter notebook,
including the interactive widget and rotatable 3D Matplotlib figures:

[Launch the Binder tutorial](https://mybinder.org/v2/gh/SlimaneD/pynamo/master?filepath=tutorial.ipynb)

Binder runs in the browser and does not require a local installation. First launch
can take a few minutes while Binder builds the environment.

**2. Faster static tutorial in Google Colab**

Use Colab if you want a faster browser-based preview of the tutorial:

[Open the Colab tutorial](https://colab.research.google.com/github/SlimaneD/pynamo/blob/master/tutorial_colab.ipynb)

The Colab notebook supports ordinary plotting cells, but not the interactive
widget or rotatable 3D Matplotlib figures.

**3. Local installation from GitHub**

Use a local installation if you want to use pyNamo-EGT in your own notebooks or
modify the code:

```bash
git clone https://github.com/SlimaneD/pynamo.git
cd pynamo
pip install ".[notebook]"
```

## Features

- Replicator dynamics for asymmetric 2-player / 2-strategy games (`2P2S`), symmetric 2-player / 3-strategy games (`2P3S`), symmetric 2-player / 4-strategy games (`2P4S`), and asymmetric 3-player / 2-strategy games (`3P2S`).
- A curated catalogue of built-in example games with descriptions, references, parameter notes, and explanations of what each example illustrates.
- Matplotlib phase portraits with trajectories, equilibria, speed fields, vector fields, and optional colored faces for 3D state spaces.
- Equilibrium analysis with linear stability classification, Nash equilibria, strict Nash equilibria, and ESS checks where applicable.
- A Jupyter widget for quick exploration of built-in example games.

## Requirements

- Python 3.12+
- `numpy`, `scipy`, `matplotlib`, `sympy`, `pandas`
- Optional for notebooks/widgets: `jupyter`, `ipykernel`, `ipywidgets`, `ipympl`

Install dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib sympy pandas ipywidgets ipympl
```

## Installation

pyNamo-EGT is currently distributed from GitHub. For ordinary notebook use, install with:

```bash
pip install ".[notebook]"
```

For core functionality only, without notebook/widget dependencies:

```bash
pip install .
```

For development tests:

```bash
pip install ".[dev]"
python -m pytest -q
```

## First use

```python
import matplotlib.pyplot as plt

import drawer
import examples

fig, ax = drawer.phase_portrait(examples.games.good_rps)
plt.show()
```

`drawer.phase_portrait` returns ordinary Matplotlib objects, so figures can be
modified or saved with standard Matplotlib commands:

```python
fig.savefig("good_rps.svg", bbox_inches="tight")
fig.savefig("good_rps.pdf", bbox_inches="tight")
```

## Built-In Examples

Built-in games are available through `examples.games`:

```python
g = examples.games.battle_of_the_sexes
same_game = examples.games("battle_of_the_sexes")
examples.games.by_class("2P2S")
```

Each catalogue game carries metadata:

```python
g = examples.games.chaotic_four_strategy_game
g.describe()
```

You can also use the module-level helper:

```python
examples.describe(g)
```

The metadata include the game description, reference, parameter values, and the
main mathematical point illustrated by the example.

## Game Classes

pyNamo currently supports four game classes:

- `2P2S`: 2-player / 2-strategy games, represented by one payoff matrix per player.
- `2P3S`: symmetric 2-player / 3-strategy games, represented by one `3 x 3` payoff matrix.
- `2P4S`: symmetric 2-player / 4-strategy games, represented by one `4 x 4` payoff matrix.
- `3P2S`: 3-player / 2-strategy games, represented by one `2 x 2 x 2` payoff tensor per player.

For asymmetric 2-strategy games, each coordinate in a reduced state is the
probability that the corresponding player uses their first listed strategy.

For symmetric 3-strategy games, initial conditions use two coordinates and the
third strategy frequency is inferred. For symmetric 4-strategy games, initial
conditions use three coordinates and the fourth strategy frequency is inferred.

## Defining Games

A symmetric 3-strategy game:

```python
import numpy as np
import game

my_game = game.Game(
    name="My RPS Variant",
    payoffs=np.array([
        [0, -1, 2],
        [2, 0, -1],
        [-1, 2, 0],
    ], dtype=float),
    strategy_labels=["R", "P", "S"],
)
```

An asymmetric 2-player / 2-strategy game:

```python
my_asymmetric_game = game.Game(
    name="My Asymmetric Game",
    payoffs=(
        np.array([[3, 0], [1, 2]], dtype=float),
        np.array([[2, 1], [0, 3]], dtype=float),
    ),
    player_strategy_labels=[["A", "B"], ["C", "D"]],
    player_labels=["Player 1", "Player 2"],
    symmetric=False,
)
```

## Plot Customization

Most common plotting options are parameters of `drawer.phase_portrait`:

```python
fig, ax = drawer.phase_portrait(
    examples.games.matching_pennies,
    starts=[[0.2, 0.7], [0.7, 0.5], [0.9, 0.9]],
    tmax=40,
    speed_cmap=plt.cm.cividis,
    speed_levels=20,
    show_vector_field=True,
    vector_grid=18,
    trajectory_color="black",
    trajectory_linewidth=0.8,
    trajectory_arrows=[0.001],
)
```

Use one color per trajectory by passing a list:

```python
fig, ax = drawer.phase_portrait(
    examples.games.cyclic_mismatching_pennies,
    starts=[[0.52, 0.50, 0.48], [0.70, 0.45, 0.35]],
    trajectory_color=["tab:blue", "tab:orange"],
    trajectory_arrows=[],
    tmax=1000,
)
```

Colored faces are available for 3D state spaces:

```python
fig, ax = drawer.phase_portrait(
    examples.games.ownership_game,
    show_faces=True,
    face_alpha=0.15,
)
```

Vector fields are available for both 2D and 3D game classes. Sparse 3D vector
fields can be useful for exploration, but trajectories are usually clearer in
static publication figures.

Labels are Matplotlib text labels and can include simple LaTeX-style math
notation such as `"$S_1$"` or `"$x = P(A)$"`. pyNamo-EGT does not require a
full LaTeX installation by default; users who want full LaTeX rendering can
enable Matplotlib's `text.usetex` option manually.

For the full parameter documentation:

```python
help(drawer.phase_portrait)
```

## Equilibrium Analysis

For notebooks, use `analysis.equilibrium_table`:

```python
import analysis

analysis.equilibrium_table(examples.games.good_rps)
```

For programmatic use:

```python
result = analysis.analyze_equilibria(examples.games.good_rps)
rows = result.to_rows()
```

For quick access to static equilibrium concepts:

```python
analysis.find_nash(examples.games.good_rps)
analysis.find_strict_nash(examples.games.good_rps)
analysis.find_ess(examples.games.good_rps)
```

`find_ess` returns ESS only for symmetric games.

## Stability Caveats

Stability is classified from the linearization restricted to admissible directions
in the state space. This is important at boundaries because outward perturbations
are not valid evolutionary deviations.

Some equilibria are non-hyperbolic or belong to degenerate equilibrium sets. In
these cases pyNamo emits warnings rather than forcing a classification. Non-isolated
equilibrium manifolds are not plotted automatically; isolated equilibria are still
shown when they can be identified.

The category `unstable` means that linearization proves the equilibrium is not
stable, but does not always distinguish source from saddle. In plots, unstable
equilibria are drawn with the source color for visual compatibility.

## Interactive Widget

In a notebook, use:

```python
%matplotlib widget

from interactive import launch_replicator_widget
launch_replicator_widget()
```

The widget lets users choose a game class and example, adjust trajectories, toggle
speed/vector fields, and inspect payoff data and equilibrium analysis.

If 3D rotation does not work, make sure the notebook kernel has `ipympl` installed
and that `%matplotlib widget` has been evaluated.

## Repository Structure

- `game.py`: core `Game` class and game-class inference.
- `examples.py`: curated catalogue of predefined games.
- `dynamics.py`: replicator vector fields and rest-point computation.
- `analysis.py`: equilibrium and stability analysis.
- `drawer.py`: plotting helpers and `phase_portrait`.
- `interactive.py`: Jupyter widget front-end.
- `tutorial.ipynb`: notebook tutorial.
- `tests/`: pytest test suite.
