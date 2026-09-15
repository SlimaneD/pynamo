"""One-population phase-line rendering and sampled bifurcation diagrams.

Internal helpers are shared by drawer.phase_portrait and bifurcation_diagram.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .dynamics import _analyze_1d as _analysis, replicator_1d as _flow
from .drawer import DEFAULT_PLOT_STYLE

from matplotlib.patches import Polygon
from matplotlib.transforms import IdentityTransform


class _DisplayArrowhead(Polygon):
    """Original four-vertex arrowhead, rebuilt in display space on every draw."""

    def __init__(self, ax, start, end, arrow_size, arrow_width, *, centered=False, color="black"):
        dimensions = np.asarray([arrow_size, arrow_width], dtype=float)
        if not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
            raise ValueError("Arrow size and width must be finite and positive.")
        self._centered = centered
        self._phase_ax = ax
        self._start = np.asarray(start, dtype=float)
        self._end = np.asarray(end, dtype=float)
        # Preserve the familiar 0.04 / 0.015 defaults: 10 pt / 3.75 pt.
        self._length_pt = 250 * arrow_size
        self._halfwidth_pt = 250 * arrow_width
        super().__init__(np.zeros((4, 2)), closed=True,
                         facecolor=color, edgecolor=color, linewidth=.6,
                         transform=IdentityTransform(), zorder=5, clip_on=False)

    def draw(self, renderer):
        start, tip = self._phase_ax.transData.transform([self._start, self._end])
        delta = tip - start
        distance = np.linalg.norm(delta)
        if distance == 0:
            return
        direction = delta / distance
        normal = np.array([-direction[1], direction[0]])
        length = renderer.points_to_pixels(self._length_pt)
        width = renderer.points_to_pixels(self._halfwidth_pt)
        # Same tip, two shoulders, and inset notch as the package polygon.
        if self._centered:
            tip = tip + .5 * length * direction
        shoulder = tip - length * direction
        notch = shoulder + width * direction
        self.set_xy([tip, shoulder + width * normal,
                     notch, shoulder - width * normal])
        super().draw(renderer)


def _draw_phase_line(ax, A, position=0., vertical=False, atol=1e-12,
                     arrow_size=.04, arrow_width=.015, equilibrium_size=80,
                     sink_color="black", source_color="white", equilibrium_edgecolor="black",
                     trajectory_color="black", trajectory_linewidth=1.2,
                     trajectory_arrows=None, show_trajectories=True, show_equilibria=True,
                     continuum_color="tab:purple"):
    points, continuum = _analysis(A, atol)
    if trajectory_arrows is None:
        positions = [(p.x+q.x)/2 for p,q in zip(points[:-1], points[1:])]
    else:
        positions = np.asarray(trajectory_arrows, dtype=float)
        if positions.ndim != 1 or not np.isfinite(positions).all():
            raise ValueError("trajectory_arrows must be a finite one-dimensional list.")
        for x in positions:
            if not 0 < x < 1 or continuum or any(abs(x-p.x) <= atol for p in points):
                raise ValueError(f"Arrow at {x:g} must be inside (0,1) and away from equilibria.")
    coords = lambda x: (position, x) if vertical else (x, position)
    ax.plot(*zip(coords(0), coords(1)),
            color=trajectory_color if show_trajectories else "black",
            lw=trajectory_linewidth if show_trajectories else 1.2, zorder=3)
    if continuum:
        if show_equilibria:
            ax.plot(*zip(coords(0), coords(1)), color=continuum_color, lw=4, zorder=6)
        return
    if show_trajectories:
        for x in positions:
            direction = np.sign(_flow(x, A))
            ax.add_artist(_DisplayArrowhead(
                ax, coords(x-direction*.01), coords(x), arrow_size, arrow_width,
                centered=True, color=trajectory_color))
    if show_equilibria:
        for p in points:
            ax.scatter(*coords(p.x), marker="o", s=equilibrium_size,
                       facecolors=sink_color if p.stable else source_color,
                       edgecolors="none" if equilibrium_edgecolor is None else equilibrium_edgecolor,
                       zorder=6, clip_on=False)


def _phase_portrait_1d(game, *, fig, ax, figsize, xlabel, title_pad,
                       simplex_font_size, **style):
    if ax is not None:
        if fig is not None and ax.figure is not fig:
            raise ValueError("fig must be the figure containing ax.")
        fig = ax.figure
    else:
        if fig is None:
            fig = plt.figure(figsize=figsize, layout="constrained")
        ax = fig.add_subplot(111)
    A = getattr(game, "payoff_data", game)
    _draw_phase_line(ax, A, **style)
    labels = getattr(game, "strategy_labels", None) or ["Strategy 1", "Strategy 2"]
    ax.set(xlim=(-.04, 1.04), ylim=(-.2, .2), yticks=[], xticks=[], xlabel=xlabel or "")
    ax.set_title(getattr(game, "name", ""), pad=title_pad)
    for x, label in ((0, labels[1]), (1, labels[0])):
        ax.annotate(
            label,
            xy=(x, 0),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=simplex_font_size,
            annotation_clip=False,
        )
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    return fig, ax


def bifurcation_diagram(payoffs, *, parameter_range, phase_line_values=None, phase_line_labels=None,
                        parameter_values=None, samples=601, parameter_label="Parameter",
                        strategy_labels=("A", "B"), fig=None, ax=None, figsize=None,
                        xlabel=None, ylabel=None, title_pad=18, atol=1e-12,
                        trajectory_color="black", trajectory_linewidth=1.2,
                        trajectory_arrows=None, show_trajectories=True, show_equilibria=True,
                        stable_color="black", stable_linewidth=1.8, stable_linestyle="-",
                        unstable_color="black", unstable_linewidth=1.8, unstable_linestyle="--",
                        arrow_size=DEFAULT_PLOT_STYLE["arrow_size"],
                        arrow_width=DEFAULT_PLOT_STYLE["arrow_width"],
                     equilibrium_size=DEFAULT_PLOT_STYLE["equilibrium_size"],
                     sink_color=DEFAULT_PLOT_STYLE["sink_color"],
                     source_color=DEFAULT_PLOT_STYLE["source_color"],
                     equilibrium_edgecolor=DEFAULT_PLOT_STYLE["equilibrium_edgecolor"],
                        continuum_color="tab:purple"):
    """Plot equilibria of a one-parameter family of symmetric 2x2 games.

    payoffs is a callable taking one finite scalar and returning a finite 2x2
    matrix. parameter_range is (minimum, maximum); samples controls the grid.
    This is a sampled diagram, not an exhaustive bifurcation detector. Supply
    known critical values in parameter_values or phase_line_values.

    phase_line_values selects vertical phase lines. phase_line_labels supplies
    one string per input value (including LaTeX); None shows numeric values,
    and empty strings hide labels. Unsorted values retain their label pairing.

    trajectory_arrows=None centers heads between consecutive equilibria on
    each line; explicit frequencies override this, and [] omits heads. An
    equilibrium position is invalid. trajectory_color/linewidth style lines
    and heads. show_trajectories=False retains a basic frame. show_equilibria
    hides only phase-line equilibrium markers, not equilibrium curves.

    sink_color/source_color specify marker fills, equilibrium_size is marker
    area in points squared, equilibrium_edgecolor=None removes outlines.
    stable_color/linewidth/linestyle and unstable_color/linewidth/linestyle
    independently style curves. continuum_color styles stationary continua.

    arrow_size/width use display-space sizing: 0.04/0.015 correspond to a
    10-point head length / 3.75-point half-width. This preserves the custom
    arrow shape across axis ranges and resizing. atol is an absolute tolerance
    for zero payoff differences. Figure controls are fig, ax, figsize, xlabel,
    ylabel and title_pad. Returns (fig, ax).
    """
    lo, hi = map(float, parameter_range)
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("Use a finite increasing parameter_range.")
    if not isinstance(samples, int) or samples < 2:
        raise ValueError("samples must be an integer >= 2.")
    def checked(values):
        a = np.asarray([] if values is None else values, dtype=float)
        if a.ndim != 1 or not np.isfinite(a).all() or np.any((a < lo) | (a > hi)):
            raise ValueError("Selected parameter values must be finite and inside the range.")
        return np.unique(a)
    lines = checked(phase_line_values)
    if phase_line_labels is None:
        line_labels = {value: f"{value:g}" for value in lines}
    else:
        supplied = np.asarray([] if phase_line_values is None else phase_line_values, dtype=float)
        if isinstance(phase_line_labels, str):
            raise ValueError("phase_line_labels must be a list of strings.")
        labels = list(phase_line_labels)
        if len(labels) != len(supplied) or not all(isinstance(label, str) for label in labels):
            raise ValueError("Provide one string label per supplied phase_line_value.")
        line_labels = {}
        for value, label in zip(supplied, labels):
            if value in line_labels and line_labels[value] != label:
                raise ValueError("Duplicate phase-line values must have identical labels.")
            line_labels[value] = label
    grid = np.unique(np.concatenate([np.linspace(lo, hi, samples), lines, checked(parameter_values)]))
    if ax is not None:
        if fig is not None and ax.figure is not fig:
            raise ValueError("fig must be the figure containing ax.")
        fig = ax.figure
    else:
        if fig is None:
            fig = plt.figure(figsize=figsize or (9, 5), layout="constrained")
        ax = fig.add_subplot(111)
    if figsize is not None:
        fig.set_size_inches(figsize)
    ax.set_title(ax.get_title(), pad=title_pad)
    # Keep the two boundary branches and the interior branch separate.
    # NaNs break lines wherever a branch is absent or changes stability.
    branches = {
        (branch, stable): np.full(len(grid), np.nan)
        for branch in ("zero", "interior", "one")
        for stable in (True, False)
    }
    continua = []
    for index, s in enumerate(grid):
        points, continuum = _analysis(payoffs(float(s)), atol)
        if continuum:
            continua.append(s)
        for p in points:
            branch = "zero" if p.x == 0 else "one" if p.x == 1 else "interior"
            branches[branch, p.stable][index] = p.x
    for stable, color, width, style in [
        (True, stable_color, stable_linewidth, stable_linestyle),
        (False, unstable_color, unstable_linewidth, unstable_linestyle),
    ]:
        for branch in ("zero", "interior", "one"):
            ax.plot(grid, branches[branch, stable], color=color, linestyle=style, lw=width, zorder=2,
                    label="stable branch" if stable else "unstable branch")

    if continua:
        ax.vlines(continua, 0, 1, color=continuum_color, alpha=.4, lw=2)
    for s in lines:
        try:
            _draw_phase_line(
                ax, payoffs(float(s)), position=s, vertical=True, atol=atol,
                arrow_size=arrow_size, arrow_width=arrow_width,
                equilibrium_size=equilibrium_size, sink_color=sink_color,
                source_color=source_color, equilibrium_edgecolor=equilibrium_edgecolor,
                trajectory_color=trajectory_color, trajectory_linewidth=trajectory_linewidth,
                trajectory_arrows=trajectory_arrows, show_trajectories=show_trajectories,
                show_equilibria=show_equilibria, continuum_color=continuum_color,
            )
        except ValueError as exc:
            raise ValueError(f"Phase line at parameter {s:g}: {exc}") from exc
        if line_labels[s]:
            ax.annotate(line_labels[s], (s, 1), xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=8)
    pad = .025 * (hi-lo)
    ax.set(xlim=(lo-pad, hi+pad), ylim=(-.05, 1.12), xlabel=parameter_label if xlabel is None else xlabel,
           ylabel=f"Equilibrium frequency of {strategy_labels[0]}" if ylabel is None else ylabel)
    ax.set_yticks([0, .25, .5, .75, 1])
    handles = []
    for color, width, style, fill, label in [
        (stable_color, stable_linewidth, stable_linestyle, sink_color, "Attracting"),
        (unstable_color, unstable_linewidth, unstable_linestyle, source_color, "Repelling"),
    ]:
        handles.append(Line2D([], [], color=color, lw=width, linestyle=style,
                              marker="o" if show_equilibria and len(lines) else None,
                              markersize=np.sqrt(equilibrium_size), markerfacecolor=fill,
                              markeredgecolor="none" if equilibrium_edgecolor is None else equilibrium_edgecolor,
                              label=label))
    if continua:
        handles.append(Line2D([], [], color=continuum_color, label="Continuum of equilibria"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.22),
              ncol=2, fontsize=8, frameon=False)
    return ax.figure, ax
