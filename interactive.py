"""Notebook widgets for exploring pyNamo replicator dynamics."""
from __future__ import annotations

from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np

import drawer
import parameters as param

__all__ = ["launch_replicator_widget"]

try:  # pragma: no cover - optional dependency
    import ipywidgets as widgets
    from IPython.display import clear_output, display
except ImportError as exc:  # pragma: no cover - optional path
    widgets = None  # type: ignore[assignment]
    _IPYWIDGETS_IMPORT_ERROR = exc
else:
    _IPYWIDGETS_IMPORT_ERROR = None


GAME_LABELS = {
    "2P3S": "Symmetric 2-player / 3-strategy",
    "2P4S": "Symmetric 2-player / 4-strategy",
    "2P2S": "Asymmetric 2-pop / 2-strategy",
    "3P2S": "Three populations / 2-strategy",
}


def _sample_initial_conditions(payoff, count: int, rng: np.random.Generator) -> List[List[float]]:
    if isinstance(payoff, np.ndarray):
        n = payoff.shape[0]
        samples = rng.dirichlet(np.ones(n), size=count)
        return [vec[:-1].tolist() for vec in samples]

    first = payoff[0]
    if first.ndim == 2:  # 2-pop 2-strategy
        return rng.random((count, 2)).tolist()

    if first.ndim == 3:  # cube (three populations)
        return rng.random((count, 3)).tolist()

    return rng.random((count, 2)).tolist()


def _plot(
    game_key: str,
    example_id: int,
    *,
    num_traj: int,
    tmax: float,
    seed: int,
    show_speed: bool,
    dynamic_label: str,
    arrow_positions: Sequence[float],
    show_equilibria: bool,
) -> None:
    games = param.available_games(game_key)
    game = games[example_id]
    payoff = game.payoff_data

    rng = np.random.default_rng(seed)
    starts = _sample_initial_conditions(payoff, num_traj, rng)
    colors = plt.cm.tab10(np.linspace(0, 1, max(num_traj, 3)))
    _, ax = drawer.plot_game(
        game,
        starts=starts,
        tmax=tmax,
        trajectory_arrows=arrow_positions,
        trajectory_color=colors,
        show_speed=show_speed,
        speed_cmap=plt.cm.Greys,
        speed_levels=8,
        show_equilibria=show_equilibria,
        sink_color="black",
        saddle_color="gray",
        source_color="white",
        equilibrium_size=60,
    )
    ax.set_title(f"{game_key} - {game.name} ({dynamic_label})")
    plt.show()


def launch_replicator_widget() -> None:
    """Display interactive controls inside a notebook."""
    if widgets is None:  # pragma: no cover
        raise ImportError(
            "ipywidgets is required for this feature"
        ) from _IPYWIDGETS_IMPORT_ERROR

    game_dropdown = widgets.Dropdown(
        options=[(label, key) for key, label in GAME_LABELS.items() if key in param.GAME_CATALOG],
        description="Game",
        value="2P3S",
    )

    example_dropdown = widgets.Dropdown(description="Example")
    dynamic_dropdown = widgets.Dropdown(
        options=[("Replicator", "Replicator")],
        description="Dynamic",
    )

    tmax_slider = widgets.FloatSlider(
        value=30,
        min=5,
        max=100,
        step=5,
        description="tmax",
        continuous_update=False,
    )
    traj_slider = widgets.IntSlider(
        value=4,
        min=1,
        max=12,
        step=1,
        description="# trajectories",
        continuous_update=False,
    )
    seed_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=999,
        step=1,
        description="Seed",
        continuous_update=False,
    )
    speed_toggle = widgets.Checkbox(value=True, description="Show speed field")
    eq_toggle = widgets.Checkbox(value=True, description="Show equilibria")
    arrow_slider = widgets.IntSlider(
        value=4,
        min=1,
        max=12,
        step=1,
        description="# arrows",
        continuous_update=False,
    )
    arrow_text = widgets.Text(
        value="",
        description="Custom arrows",
        placeholder="e.g. 0.2,0.4,0.8",
        continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )

    def _current_arrow_positions() -> List[float]:
        raw = arrow_text.value.strip()
        if raw:
            positions: List[float] = []
            for chunk in raw.split(","):
                try:
                    val = float(chunk)
                except ValueError:
                    continue
                if 0.0 <= val <= 1.0:
                    positions.append(val)
            if positions:
                return positions
        count = arrow_slider.value
        if count == 1:
            return [0.5]
        return np.linspace(0.1, 0.9, count).tolist()

    def _update_examples(*_):
        games = param.available_games(game_dropdown.value)
        options = [(g.name, idx) for idx, g in games.items()]
        example_dropdown.options = options
        if options:
            example_dropdown.value = options[0][1]

    _update_examples()
    game_dropdown.observe(_update_examples, names="value")

    controls = widgets.VBox(
        [
            game_dropdown,
            example_dropdown,
            dynamic_dropdown,
            widgets.HBox([traj_slider, tmax_slider]),
            widgets.HBox([seed_slider, speed_toggle, eq_toggle]),
            widgets.HBox([arrow_slider]),
            arrow_text,
        ]
    )

    output = widgets.Output()

    def _render(*args):
        with output:
            clear_output(wait=True)
            _plot(
                game_dropdown.value,
                example_dropdown.value,
                num_traj=traj_slider.value,
                tmax=tmax_slider.value,
                seed=seed_slider.value,
                show_speed=speed_toggle.value,
                show_equilibria=eq_toggle.value,
                dynamic_label=dynamic_dropdown.value,
                arrow_positions=_current_arrow_positions(),
            )

    for widget in [
        game_dropdown,
        example_dropdown,
        dynamic_dropdown,
        traj_slider,
        tmax_slider,
        seed_slider,
        speed_toggle,
        eq_toggle,
        arrow_slider,
        arrow_text,
    ]:
        widget.observe(_render, names="value")

    display(controls, output)
    _render()
