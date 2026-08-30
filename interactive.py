"""Notebook widgets for exploring pyNamo replicator dynamics."""
from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np

import analysis
import drawer
import examples

__all__ = ["launch_replicator_widget"]

try:  # pragma: no cover - optional dependency
    import ipywidgets as widgets
    from IPython.display import Markdown, Math, clear_output, display
except ImportError as exc:  # pragma: no cover - optional path
    widgets = None  # type: ignore[assignment]
    _IPYWIDGETS_IMPORT_ERROR = exc
else:
    _IPYWIDGETS_IMPORT_ERROR = None


GAME_LABELS = {
    "2P3S": "Symmetric 2-player / 3-strategy",
    "2P4S": "Symmetric 2-player / 4-strategy",
    "2P2S": "Asymmetric 2-player / 2-strategy",
    "3P2S": "Asymmetric 3-player / 2-strategy",
}


def _sample_initial_conditions(game, count: int, rng: np.random.Generator) -> List[List[float]]:
    game_class = game.game_class
    if game_class in ("2P3S", "2P4S"):
        n = game.num_strategies()
        samples = rng.dirichlet(np.ones(n), size=count)
        return [vec[:-1].tolist() for vec in samples]

    if game_class == "2P2S":
        return rng.random((count, 2)).tolist()

    if game_class == "3P2S":
        return rng.random((count, 3)).tolist()

    return rng.random((count, 2)).tolist()


def _plot(
    game_key: str,
    example_name: str,
    *,
    num_traj: int,
    tmax: float,
    seed: int,
    show_speed: bool,
    show_vector_field: bool,
    num_arrows: int,
    show_equilibria: bool,
    show_payoff_data: bool,
    show_analysis_table: bool,
) -> None:
    game = examples.games(example_name)
    payoff = game.payoff_data

    if show_payoff_data:
        _display_payoff_data(game)

    if show_analysis_table:
        _display_section_title("Equilibrium Analysis")
        _display_analysis_table(game)

    rng = np.random.default_rng(seed)
    starts = _sample_initial_conditions(game, num_traj, rng)
    trajectory_color = "royalblue" if game_key in ("2P4S", "3P2S") else "black"
    _, ax = drawer.plot_game(
        game,
        starts=starts,
        tmax=tmax,
        trajectory_arrows=_arrow_positions(num_arrows),
        trajectory_color="black" if game_key in ("2P3S", "2P2S") else trajectory_color,
        show_speed=show_speed,
        speed_cmap=plt.cm.cividis,
        speed_levels=80,
        show_vector_field=show_vector_field,
        vector_color="white" if game_key in ("2P3S", "2P2S") else "black",
        show_equilibria=show_equilibria,
        sink_color="black",
        saddle_color="gray",
        source_color="white",
        equilibrium_size=60,
    )
    ax.set_title(f"{game_key} - {game.name}")
    plt.show()


def _arrow_positions(count: int) -> List[float]:
    if count <= 0:
        return []
    return [0.001]


def _display_payoff_data(game) -> None:
    _display_section_title("Payoff Data")
    payoff = game.payoff_data
    game_class = game.game_class

    if game_class in ("2P3S", "2P4S"):
        display(Math(_symmetric_payoff_latex(payoff, game.strategy_labels)))
        return

    if game_class == "2P2S":
        display(Math(_bimatrix_payoff_latex(payoff, game.player_strategy_labels)))
        return

    if game_class == "3P2S":
        display(Markdown(f"$$\n{_three_player_payoff_latex(payoff).strip()}\n$$"))
        return

    if isinstance(payoff, np.ndarray):
        display(payoff)
        return

    for idx, player_payoff in enumerate(payoff, start=1):
        display(Markdown(f"**Player {idx}**"))
        display(player_payoff)


def _display_section_title(title: str) -> None:
    display(Markdown(f"<br><h3>{title}</h3><br>"))


def _display_analysis_table(game) -> None:
    rows = analysis.analyze_equilibria(game).to_rows()
    display(Math(_analysis_table_latex(rows)))


def _analysis_table_latex(rows: List[dict]) -> str:
    if not rows:
        return r"\text{No isolated equilibria found.}"

    columns = [
        ("Position", "Position"),
        ("Stability", "Stability Status"),
        ("Nash", "Nash"),
        ("ESS", "ESS"),
        ("Strict Nash", "Strict Nash"),
        ("Eigenvalues", "Eigenvalues"),
        ("Eigenvectors", "Eigenvectors"),
    ]

    header = " & ".join(rf"\text{{{label}}}" for label, _ in columns) + r" \\"
    table_rows = [header, r"\hline"]
    for row in rows:
        entries = [_analysis_cell(row[key]) for _, key in columns]
        table_rows.append(" & ".join(entries) + r" \\")

    return _latex_array("c" * len(columns), table_rows)


def _analysis_cell(value) -> str:
    if isinstance(value, list):
        if value and all(isinstance(item, list) for item in value):
            return _latex_vector_list(value)
        return _latex_column_vector(value)
    if value is None:
        return r"\text{None}"
    if isinstance(value, bool):
        return rf"\text{{{value}}}"
    if isinstance(value, str):
        return rf"\text{{{value}}}"
    return _format_number_latex(value)


def _latex_column_vector(values) -> str:
    entries = r" \\ ".join(_format_number_latex(value) for value in values)
    return rf"\begin{{pmatrix}} {entries} \end{{pmatrix}}"


def _latex_vector_list(vectors) -> str:
    entries = ", ".join(_latex_column_vector(vector) for vector in vectors)
    return rf"\left\{{ {entries} \right\}}"


def _format_number_latex(value) -> str:
    value = complex(value)
    real = float(value.real)
    imag = float(value.imag)

    if np.isclose(imag, 0.0):
        return _format_payoff(real)
    if np.isclose(real, 0.0):
        return rf"{_format_payoff(imag)}i"

    sign = "+" if imag > 0 else "-"
    return rf"{_format_payoff(real)} {sign} {_format_payoff(abs(imag))}i"


def _symmetric_payoff_latex(payoff, labels: List[str]) -> str:
    n = payoff.shape[0]
    labels = _strategy_labels_or_default(labels, n)
    header = " & " + " & ".join(labels) + r" \\"
    rows = [header, r"\hline"]
    for i, label in enumerate(labels):
        entries = " & ".join(_format_payoff(payoff[i, j]) for j in range(n))
        rows.append(rf"{label} & {entries} \\")
    return _latex_array("c|" + "c" * n, rows)


def _bimatrix_payoff_latex(payoff, player_strategy_labels: List[List[str]]) -> str:
    p1_payoff, p2_payoff = payoff
    row_labels = _strategy_labels_or_default(player_strategy_labels[0], 2)
    column_labels = _strategy_labels_or_default(player_strategy_labels[1], 2)
    rows = [
        " & " + " & ".join(column_labels) + r" \\",
        r"\hline",
    ]
    for i, row_label in enumerate(row_labels):
        entries = []
        for j in range(2):
            # Player 2 matrices use own action as row internally, so transpose for display.
            entries.append(
                rf"({_format_payoff(p1_payoff[i, j])}, {_format_payoff(p2_payoff[j, i])})"
            )
        rows.append(rf"{row_label} & {' & '.join(entries)} \\")
    return _latex_array("c|cc", rows)


def _three_player_payoff_latex(payoff) -> str:
    matrices = []
    for k in range(2):
        rows = [
            rf" & \text{{Player 3: {k + 1}}} & \\[0.4em]",
            r"\hline",
            r" & \text{Player 2: 1} & \text{Player 2: 2} \\",
            r"\hline",
        ]
        for i in range(2):
            entries = []
            for j in range(2):
                payoff_tuple = ", ".join(_format_payoff(tensor[i, j, k]) for tensor in payoff)
                entries.append(rf"({payoff_tuple})")
            rows.append(rf"\text{{Player 1: {i + 1}}} & {entries[0]} & {entries[1]} \\")
        rows.append(r"\hline")
        matrices.append(_latex_array("ccc", rows, display_math=False))

    return rf"""
{matrices[0]}
\qquad
{matrices[1]}
"""


def _latex_array(column_spec: str, rows: List[str], display_math: bool = True) -> str:
    body = "\n".join(rows)
    array = rf"""
\begin{{array}}{{{column_spec}}}
{body}
\end{{array}}
"""
    if display_math:
        return array
    return array


def _strategy_labels_or_default(labels: List[str], n: int) -> List[str]:
    if len(labels) == n:
        return [_latex_label(label) for label in labels]
    return [str(idx + 1) for idx in range(n)]


def _latex_label(label: str) -> str:
    label = str(label).strip()
    if label.startswith("$") and label.endswith("$") and len(label) >= 2:
        return label[1:-1]
    return rf"\text{{{_escape_latex_text(label)}}}"


def _escape_latex_text(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def _format_payoff(value) -> str:
    value = float(value)
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.3g}"


def launch_replicator_widget() -> None:
    """Display the interactive pyNamo widget inside a Jupyter notebook.

    The widget exposes the built-in example catalogue, trajectory controls,
    speed-field and vector-field toggles, payoff display, and equilibrium
    analysis display.

    Raises
    ------
    ImportError
        Raised when `ipywidgets` or IPython display tools are unavailable.

    Notes
    -----
    This function is intended for notebook use. In a development notebook, reload
    `interactive`, `drawer`, `analysis`, `dynamics`, `game`, and `examples`
    before calling it if those modules have been edited during the session.
    """
    if widgets is None:  # pragma: no cover
        raise ImportError(
            "ipywidgets is required for this feature"
        ) from _IPYWIDGETS_IMPORT_ERROR

    game_dropdown = widgets.Dropdown(
        options=[
            (label, key)
            for key, label in GAME_LABELS.items()
            if examples.games.by_class(key)
        ],
        description="Game",
        value="2P3S",
    )

    example_dropdown = widgets.Dropdown(description="Example")

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
    vector_toggle = widgets.Checkbox(value=False, description="Show vector field")
    eq_toggle = widgets.Checkbox(value=True, description="Show equilibria")
    payoff_toggle = widgets.Checkbox(value=True, description="Show payoff data")
    analysis_toggle = widgets.Checkbox(value=True, description="Show analysis table")
    arrow_slider = widgets.IntSlider(
        value=1,
        min=0,
        max=1,
        step=1,
        description="# arrows",
        continuous_update=False,
    )

    def _update_examples(*_):
        games_by_class = examples.games.by_class(game_dropdown.value)
        options = [(g.name, name) for name, g in games_by_class.items()]
        example_dropdown.options = options
        if options:
            example_dropdown.value = options[0][1]

    _update_examples()
    game_dropdown.observe(_update_examples, names="value")

    controls = widgets.VBox(
        [
            game_dropdown,
            example_dropdown,
            widgets.HBox([traj_slider, tmax_slider]),
            widgets.HBox([seed_slider, arrow_slider]),
            widgets.HBox([speed_toggle, vector_toggle, eq_toggle]),
            widgets.HBox([payoff_toggle, analysis_toggle]),
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
                show_vector_field=vector_toggle.value,
                num_arrows=arrow_slider.value,
                show_equilibria=eq_toggle.value,
                show_payoff_data=payoff_toggle.value,
                show_analysis_table=analysis_toggle.value,
            )

    for widget in [
        game_dropdown,
        example_dropdown,
        traj_slider,
        tmax_slider,
        seed_slider,
        arrow_slider,
        speed_toggle,
        vector_toggle,
        eq_toggle,
        payoff_toggle,
        analysis_toggle,
    ]:
        widget.observe(_render, names="value")

    display(controls, output)
    _render()
