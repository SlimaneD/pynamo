# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:22:02 2020

@author: Benjamin Giraudon
Status : - need to check for presence of LaTeX installation on the device
         - better management of equilibria in 3D games (2P4S)
         To add : - feature that plots higher dimension manifolds
                  - automatically draw relevant trajectories
"""

import math
import warnings
import numpy as np
from scipy.integrate import odeint

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D

import analysis
import dynamics
from game import infer_game_class

__all__ = [
    "PlottingWarning",
    "p_to_sim",
    "sim_to_p",
    "sim_to_p_2p4s",
    "p_to_sim_2p4s",
    "draw_state_space",
    "plot_trajectory",
    "plot_equilibria",
    "plot_speed_field",
    "plot_vector_field",
    "plot_game",
]


class PlottingWarning(RuntimeWarning):
    """Warning raised when plotting compresses analysis categories."""


warnings.simplefilter("always", PlottingWarning)


DEFAULT_PLOT_STYLE = {
    "figsize": (6, 6),
    "view_elev": 25,
    "view_azim": 35,
    "simplex_font_size": 13,
    "simplex_zorder": 30,
    "show_speed": True,
    "speed_grid": 60,
    "speed_cmap": plt.cm.Spectral,
    "speed_levels": 12,
    "speed_zorder": 10,
    "show_vector_field": False,
    "vector_grid": 15,
    "vector_margin": 0.02,
    "vector_color": "black",
    "vector_alpha": 0.75,
    "vector_length": 0.04,
    "vector_width": 0.003,
    "vector_zorder": 15,
    "vector_normalize": True,
    "show_trajectories": True,
    "trajectory_step": 0.02,
    "trajectory_arrows": [0.02],
    "tmax": 45,
    "trajectory_color": "black",
    "arrow_size": 0.04,
    "arrow_width": 0.015,
    "trajectory_zorder": 20,
    "show_equilibria": True,
    "sink_color": "black",
    "saddle_color": "gray",
    "source_color": "white",
    "center_color": None,
    "equilibrium_size": 80,
    "equilibrium_zorder": 40,
}


def p_to_sim(x, y):
    """Convert simplex coordinates to 2P3S plotting plane."""
    return [-0.5 * x - y + 1, (np.sqrt(3) / 2) * x]


def sim_to_p(x, y):
    """Convert 2P3S plotting plane coordinates back to simplex coordinates."""
    return [2 / 3 * np.sqrt(3) * y, -1 / 3 * np.sqrt(3) * y - x + 1]


def sim_to_p_2p4s(x, y, z):
    """Convert simplex coordinates to 2P4S 3D plotting space."""
    return [0.5 * (-y + z + 1), np.sqrt(3) / 4 * (x - y - z + 1), -np.sqrt(13) / 4 * (x + y + z - 1)]


def p_to_sim_2p4s(x, y, z):
    """Convert 2P4S plotting space coordinates back to simplex coordinates."""
    return [2 * (np.sqrt(3) / 3 * y - np.sqrt(13) / 13 * z), -x + np.sqrt(3) / 3 * y - np.sqrt(13) / 13 * z + 1, x + np.sqrt(3) / 3 * y - np.sqrt(13) / 13 * z]


def _arrow_tip_candidates(x_A, y_A, s, a):
    """Coordinates of the arrow tip offset from (x_A, y_A) along slope s with length a."""
    return [
        [((s**2 + 1) * x_A - np.sqrt(s**2 + 1) * a) / (s**2 + 1), -((np.sqrt(s**2 + 1) * a * s - (s**2 + 1) * y_A) / (s**2 + 1))],
        [((s**2 + 1) * x_A + np.sqrt(s**2 + 1) * a) / (s**2 + 1), ((np.sqrt(s**2 + 1) * a * s + (s**2 + 1) * y_A) / (s**2 + 1))],
    ]


def _arrow_base_candidates(x_F, y_F, s, c):
    """Coordinates of the arrow base points around (x_F, y_F) with width c along slope s."""
    return [
        [((s**2 + 1) * x_F - np.sqrt(s**2 + 1) * c) / (s**2 + 1), -((np.sqrt(s**2 + 1) * c * s - (s**2 + 1) * y_F) / (s**2 + 1))],
        [((s**2 + 1) * x_F + np.sqrt(s**2 + 1) * c) / (s**2 + 1), ((np.sqrt(s**2 + 1) * c * s + (s**2 + 1) * y_F) / (s**2 + 1))],
    ]


def _arrow_side_candidates(x_F, y_F, i_p, s, c):
    """Coordinates of the lateral arrow head points C/D at perpendicular offset c."""
    root = np.sqrt(
        -s**2 * y_F**2
        + (c**2 - i_p**2) * s**2
        + 2 * i_p * s * x_F
        + c**2
        - x_F**2
        + 2 * (i_p * s**2 - s * x_F) * y_F
    )
    return [
        [
            (s**2 * x_F + i_p * s - s * y_F - root * s) / (s**2 + 1),
            (i_p * s**2 - s * x_F + y_F + root) / (s**2 + 1),
        ],
        [
            (s**2 * x_F + i_p * s - s * y_F + root * s) / (s**2 + 1),
            (i_p * s**2 - s * x_F + y_F - root) / (s**2 + 1),
        ],
    ]


def _draw_arrow_2d(start_point, end_point, fig, ax, arrow_size, arrow_width, arrow_color, zorder):
    """Creates a polygon defined by the shape of the arrow"""
    cf=arrow_width
    af=arrow_size
    x0= start_point
    xA= end_point
    xB= [0, 0]
    xF= [0, 0]
    if(x0[0]==xA[0]):
        xB[0] = xA[0]
        xF[0] = xA[0]
        if(x0[1]>=xA[1]):
            xF[1]=af+xA[1]
            xB[1]=-cf+xF[1]
        else:
            xF[1]=-af+xA[1]
            xB[1]=cf+xF[1]
        xC = [xF[0]-cf,xF[1]]
        xD = [xF[0]+cf,xF[1]]
    elif(x0[1]==xA[1]):
        xF[1]=xA[1]
        xB[1]=xA[1]
        if(x0[0]>=xA[0]):
            xF[0]=af+xA[0]
            xB[0]=-cf+xF[0]
        else:
            xF[0]=-af+xA[0]
            xB[0]=cf+xF[0]
        xC = [xF[0],xF[1]-cf]
        xD = [xF[0],xF[1]+cf]
    elif(xA[0]>x0[0]):
        sf = (xA[1]-x0[1])/(xA[0]-x0[0])
        xF = [_arrow_tip_candidates(xA[0], xA[1], sf, af)[0][0], _arrow_tip_candidates(xA[0], xA[1], sf, af)[0][1]]
        xB = [_arrow_base_candidates(xF[0], xF[1], sf, cf)[1][0], _arrow_base_candidates(xF[0], xF[1], sf, cf)[1][1]]
        xC = [_arrow_side_candidates(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[0][0], _arrow_side_candidates(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[0][1]]
        xD = [_arrow_side_candidates(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[1][0], _arrow_side_candidates(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[1][1]]
    elif(xA[0]<x0[0]):
        sf = (xA[1]-x0[1])/(xA[0]-x0[0])
        xF = [_arrow_tip_candidates(xA[0], xA[1], sf, af)[1][0], _arrow_tip_candidates(xA[0], xA[1], sf, af)[1][1]]
        xB = [_arrow_base_candidates(xF[0], xF[1], sf, cf)[0][0], _arrow_base_candidates(xF[0], xF[1], sf, cf)[0][1]]
        xC = [_arrow_side_candidates(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[0][0], _arrow_side_candidates(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[0][1]]
        xD = [_arrow_side_candidates(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[1][0], _arrow_side_candidates(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[1][1]]
    xs = [x0[0], xA[0]]
    ys = [x0[1], xA[1]]
    arrLine = plt.plot(xs, ys, color=arrow_color, zorder=zorder, clip_on=False)
    arrow = [xA, xC, xB, xD]
    verts = []
    patches = []
    for pt in arrow:
        verts.append([pt[0], pt[1]])
    arrHead = Polygon(verts)
    patches.append(arrHead)
    p = PatchCollection(patches, facecolor=arrow_color, edgecolor=arrow_color, alpha=1, zorder=zorder)
    ax.add_collection(p)
    return arrLine+[arrHead]


def _draw_arrow_3d(start_point, end_point, fig, ax, arrow_size, arrow_width, arrow_color, zorder):
    """Creates arrow with the default quiver3d from matplotlib"""
    u = end_point[0] - start_point[0]
    v = end_point[1] - start_point[1]
    w = end_point[2] - start_point[2]
    quiv = ax.quiver(
        start_point[0],
        start_point[1],
        start_point[2],
        u,
        v,
        w,
        length=0.002,
        arrow_length_ratio=15,
        pivot='tip',
        color=arrow_color,
        zorder=zorder,
        normalize=True,
    )
    return [quiv]


def _is_three_player_cube(payoff_data):
    return isinstance(payoff_data, (tuple, list)) and hasattr(payoff_data[0], "ndim") and payoff_data[0].ndim == 3



def draw_state_space(strategy_labels, payoff_data, ax, font_size, zorder):
    """Draws the simplex frame."""
    if payoff_data[0].shape == (3,):
        pt1 = p_to_sim(1, 0)
        pt2 = p_to_sim(0, 1)
        pt3 = p_to_sim(0, 0)
        lbl1 = ax.annotate(strategy_labels[0], (pt1[0] - 0.01, pt1[1] + 0.04), fontsize=font_size, zorder=zorder)
        lbl2 = ax.annotate(strategy_labels[1], (pt2[0] - 0.05, pt2[1] - 0.01), fontsize=font_size, zorder=zorder)
        lbl3 = ax.annotate(strategy_labels[2], (pt3[0] + 0.03, pt3[1] - 0.01), fontsize=font_size, zorder=zorder)
        xs = ([pt1[0], pt2[0]], [pt1[0], pt3[0]], [pt2[0], pt3[0]])
        ys = ([pt1[1], pt2[1]], [pt1[1], pt3[1]], [pt2[1], pt3[1]])
        lines = []
        for xpair, ypair in zip(xs, ys):
            lines += plt.plot(xpair, ypair, color='black', zorder=zorder, alpha=1, clip_on=False)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, (3 ** 0.5) / 2 + 0.05)
        ax.set_aspect('equal', adjustable='box')
        return lines + [lbl1, lbl2, lbl3]
    if payoff_data[0].shape == (2, 2):
        ax.set_xlabel(strategy_labels[0], fontsize=font_size)
        ax.set_ylabel(strategy_labels[1], fontsize=font_size)
        edges = [([0, 1], [0, 0]), ([1, 1], [0, 1]), ([1, 0], [1, 1]), ([0, 0], [1, 0])]
        lines = []
        for xs, ys in edges:
            lines += plt.plot(xs, ys, color='black', zorder=zorder, alpha=1, clip_on=False)
        return lines
    if _is_three_player_cube(payoff_data):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel(strategy_labels[0], fontsize=font_size)
        ax.set_ylabel(strategy_labels[1], fontsize=font_size)
        ax.set_zlabel(strategy_labels[2], fontsize=font_size)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_zticks([0, 1])
        ax.set_xticklabels(["0", "1"], fontsize=font_size - 2)
        ax.set_yticklabels(["0", "1"], fontsize=font_size - 2)
        ax.set_zticklabels(["0", "1"], fontsize=font_size - 2)
        # Hide panes/axis lines but keep ticks/labels for a minimal look.
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis.pane.set_edgecolor((1, 1, 1, 0))
        # Hide axis lines using _axinfo (mplot3d-friendly)
        for axis in ("xaxis", "yaxis", "zaxis"):
            try:
                getattr(ax, axis)._axinfo["axisline"]["linewidth"] = 0.0
            except Exception:
                pass
        corners = [
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
        ]
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 3),
            (4, 5), (4, 6), (5, 7), (6, 7),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        artists = []
        for start_idx, end_idx in edges:
            s = corners[start_idx]
            e = corners[end_idx]
            artists += ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color='black', zorder=zorder, alpha=1)
        return artists
    if payoff_data[0].shape == (4,):
        pt1 = sim_to_p_2p4s(1, 0, 0)
        pt2 = sim_to_p_2p4s(0, 1, 0)
        pt3 = sim_to_p_2p4s(0, 0, 1)
        pt4 = sim_to_p_2p4s(0, 0, 0)
        ax.grid(False)
        lbl1 = ax.text(pt1[0], pt1[1] + 0.05, pt1[2], strategy_labels[0], fontsize=font_size, zorder=zorder)
        lbl2 = ax.text(pt2[0] - 0.05, pt2[1], pt2[2], strategy_labels[1], fontsize=font_size, zorder=zorder)
        lbl3 = ax.text(pt3[0] + 0.05, pt3[1] - 0.022, pt3[2], strategy_labels[2], fontsize=font_size, zorder=zorder)
        lbl4 = ax.text(pt4[0] - 0.02, pt4[1] - 0.022, pt4[2] + 0.05, strategy_labels[3], fontsize=font_size, zorder=zorder)
        xs = [[pt1[0], pt2[0]], [pt2[0], pt3[0]], [pt3[0], pt1[0]], [pt4[0], pt1[0]], [pt4[0], pt2[0]], [pt4[0], pt3[0]]]
        ys = [[pt1[1], pt2[1]], [pt2[1], pt3[1]], [pt3[1], pt1[1]], [pt4[1], pt1[1]], [pt4[1], pt2[1]], [pt4[1], pt3[1]]]
        zs = [[pt1[2], pt2[2]], [pt2[2], pt3[2]], [pt3[2], pt1[2]], [pt4[2], pt1[2]], [pt4[2], pt2[2]], [pt4[2], pt3[2]]]
        lines = []
        for xpair, ypair, zpair in zip(xs, ys, zs):
            lines += plt.plot(xpair, ypair, zpair, color='black', zorder=zorder, alpha=1, clip_on=False)
        return lines + [lbl1, lbl2, lbl3, lbl4]
    return []


def plot_trajectory(
    initial_state,
    payoff_data,
    time_step,
    arrow_positions,
    tmax,
    fig,
    ax,
    trajectory_color,
    arrow_size,
    arrow_width,
    zorder,
    arrow_color=None,
):
    """Draws trajectories in the simplex, given a starting point."""
    t = np.linspace(0, tmax, int(tmax / time_step))
    line_color = trajectory_color if trajectory_color is not None else 'black'
    arrow_col = arrow_color if arrow_color is not None else line_color

    if payoff_data[0].shape == (3,):  # symmetric 2P3S
        x0, y0 = initial_state
        sol = odeint(dynamics.replicator_2p3s, [x0, y0], t, (payoff_data,))
        solRev = odeint(dynamics.reverse_replicator_2p3s, [x0, y0], t, (payoff_data,))
        solX = []
        solY = []
        solXrev = []
        solYrev = []
        for pt in sol:
            cPt = p_to_sim(pt[0], pt[1])
            solX.append(cPt[0])
            solY.append(cPt[1])
        for pt in solRev:
            cPt = p_to_sim(pt[0], pt[1])
            solXrev.append(cPt[0])
            solYrev.append(cPt[1])
        psol = plt.plot(solX, solY, color=line_color, zorder=zorder, clip_on=False)
        psolRev = plt.plot(solXrev, solYrev, color=line_color, zorder=zorder, clip_on=False)
        dirs = []
        for frac in arrow_positions:
            base = min(max(int(frac * (len(solX) - 1)), 0), len(solX) - 2)
            dirs += _draw_arrow_2d(
                [solX[base], solY[base]],
                [solX[base + 1], solY[base + 1]],
                fig,
                ax,
                arrow_width=arrow_width,
                arrow_size=arrow_size,
                arrow_color=arrow_col,
                zorder=zorder,
            )
        return psol + psolRev + dirs

    if payoff_data[0].shape == (2, 2):  # asymmetric 2P2S
        x0, y0 = initial_state
        sol = odeint(dynamics.replicator_2p2s, [x0, y0], t, (payoff_data,))
        solRev = odeint(dynamics.reverse_replicator_2p2s, [x0, y0], t, (payoff_data,))
        solX, solY = sol[:, 0], sol[:, 1]
        solXrev, solYrev = solRev[:, 0], solRev[:, 1]
        psol = plt.plot(solX, solY, color=line_color, zorder=zorder, clip_on=False)
        psolRev = plt.plot(solXrev, solYrev, color=line_color, zorder=zorder, clip_on=False)
        dirs = []
        for frac in arrow_positions:
            base = min(max(int(frac * (len(solX) - 1)), 0), len(solX) - 2)
            dirs += _draw_arrow_2d(
                [solX[base], solY[base]],
                [solX[base + 1], solY[base + 1]],
                fig,
                ax,
                arrow_width=arrow_width,
                arrow_size=arrow_size,
                arrow_color=arrow_col,
                zorder=zorder,
            )
        return psol + psolRev + dirs

    if _is_three_player_cube(payoff_data):
        x0, y0, z0 = initial_state
        sol = odeint(dynamics.replicator_3p2s, [x0, y0, z0], t, (payoff_data,))
        solRev = odeint(dynamics.reverse_replicator_3p2s, [x0, y0, z0], t, (payoff_data,))
        solX, solY, solZ = sol[:, 0], sol[:, 1], sol[:, 2]
        solXrev, solYrev, solZrev = solRev[:, 0], solRev[:, 1], solRev[:, 2]
        psol = ax.plot(solX, solY, solZ, linewidth=0.8, color=line_color, zorder=zorder)
        psolRev = ax.plot(solXrev, solYrev, solZrev, linewidth=0.8, color=line_color, zorder=zorder)
        dirs = []
        for frac in arrow_positions:
            base = min(max(int(frac * (len(solX) - 1)), 0), len(solX) - 2)
            dirs += _draw_arrow_3d(
                [solX[base], solY[base], solZ[base]],
                [solX[base + 1], solY[base + 1], solZ[base + 1]],
                fig,
                ax,
                arrow_width=arrow_width,
                arrow_size=arrow_size,
                arrow_color=arrow_col,
                zorder=zorder,
            )
        return psol + psolRev + dirs

    if payoff_data[0].shape == (4,):
        x0, y0, z0 = initial_state
        sol = odeint(dynamics.replicator_2p4s, [x0, y0, z0], t, (payoff_data,))
        solRev = odeint(dynamics.reverse_replicator_2p4s, [x0, y0, z0], t, (payoff_data,))
        solX, solY, solZ = [], [], []
        solXrev, solYrev, solZrev = [], [], []
        for pt in sol:
            cPt = sim_to_p_2p4s(pt[0], pt[1], pt[2])
            solX.append(cPt[0])
            solY.append(cPt[1])
            solZ.append(cPt[2])
        for pt in solRev:
            cPt = sim_to_p_2p4s(pt[0], pt[1], pt[2])
            solXrev.append(cPt[0])
            solYrev.append(cPt[1])
            solZrev.append(cPt[2])
        psol = ax.plot(solX, solY, solZ, linewidth=0.8, color=line_color, zorder=zorder)
        psolRev = ax.plot(solXrev, solYrev, solZrev, linewidth=0.8, color=line_color, zorder=zorder)
        dirs = []
        for frac in arrow_positions:
            base = min(max(int(frac * (len(solX) - 1)), 0), len(solX) - 2)
            dirs += _draw_arrow_3d(
                [solX[base], solY[base], solZ[base]],
                [solX[base + 1], solY[base + 1], solZ[base + 1]],
                fig,
                ax,
                arrow_width=arrow_width,
                arrow_size=arrow_size,
                arrow_color=arrow_col,
                zorder=zorder,
            )
        return psol + psolRev + dirs

    return None


def plot_equilibria(
    payoff_data,
    ax,
    sink_color,
    saddle_color,
    source_color,
    equilibrium_size,
    zorder,
    center_color=None,
):
    """Classify and plot equilibria for the given game/payoff_data."""
    source, sink, saddle, center, undetermined = [], [], [], [], []
    three_player = _is_three_player_cube(payoff_data)
    center_color = source_color if center_color is None else center_color

    result = analysis.analyze_equilibria(payoff_data)
    unstable_plotted_as_source = False

    def _point_to_plot(record):
        if result.game_class == "2P3S":
            return np.array(p_to_sim(record.reduced_position[0], record.reduced_position[1]))
        if result.game_class == "2P4S":
            return np.array(
                sim_to_p_2p4s(
                    record.reduced_position[0],
                    record.reduced_position[1],
                    record.reduced_position[2],
                )
            )
        return np.asarray(record.full_position, dtype=float)

    for record in result.records:
        point_to_plot = _point_to_plot(record)
        if record.stability == 'sink':
            sink.append(point_to_plot)
        elif record.stability == 'source':
            source.append(point_to_plot)
        elif record.stability == 'unstable':
            source.append(point_to_plot)
            unstable_plotted_as_source = True
        elif record.stability == 'saddle':
            saddle.append(point_to_plot)
        elif record.stability == 'center':
            center.append(point_to_plot)
        else:
            undetermined.append(point_to_plot)

    if unstable_plotted_as_source:
        warnings.warn(
            "At least one equilibrium is classified as 'unstable': linearization "
            "proves it is not stable, but does not distinguish source from saddle. "
            "For plotting compatibility, unstable equilibria are drawn with the "
            "source color and returned in the source bucket.",
            PlottingWarning,
            stacklevel=2,
        )

    def _to_raw(point):
        if result.game_class == "2P3S":
            r, p = sim_to_p(point[0], point[1])
            return [r, p, 1 - r - p]
        if result.game_class == "2P2S":
            return point.tolist()
        if three_player or result.game_class == "2P4S":
            return point.tolist()
        return point.tolist()

    def _sphere_radius(size):
        return np.clip(np.sqrt(size) / 200.0, 0.01, 0.08) * (2.0 / 3.0)

    def _plot_spheres(points, color, alpha=0.85):
        if points.size == 0:
            return
        radius = _sphere_radius(equilibrium_size)
        rgba = mcolors.to_rgba(color, alpha=alpha)
        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 12)
        for center in points:
            cx, cy, cz = center
            X = radius * np.outer(np.cos(u), np.sin(v)) + cx
            Y = radius * np.outer(np.sin(u), np.sin(v)) + cy
            Z = radius * np.outer(np.ones_like(u), np.cos(v)) + cz
            facecolors = np.empty(X.shape + (4,))
            facecolors[..., :] = rgba
            ax.plot_surface(
                X,
                Y,
                Z,
                rstride=1,
                cstride=1,
                facecolors=facecolors,
                linewidth=0,
                antialiased=True,
                shade=True,
                zorder=zorder,
            )

    def _scatter(points, color, marker='o'):
        if not points:
            return
        pts = np.array(points)
        if three_player or payoff_data[0].shape == (4,):
            _plot_spheres(pts, color)
        else:
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=equilibrium_size,
                color=color,
                marker=marker,
                edgecolors='black',
                alpha=1,
                zorder=zorder,
                clip_on=False,
            )

    def _scatter_undetermined(points):
        if not points:
            return
        pts = np.array(points)
        if three_player or payoff_data[0].shape == (4,):
            _plot_spheres(pts, 'gray', alpha=0.45)
        else:
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=equilibrium_size,
                facecolors='none',
                edgecolors='black',
                alpha=1,
                zorder=zorder,
                clip_on=False,
            )

    raw_source = [_to_raw(pt) for pt in source]
    raw_saddle = [_to_raw(pt) for pt in saddle]
    raw_sink = [_to_raw(pt) for pt in sink]
    raw_center = [_to_raw(pt) for pt in center]
    raw_undetermined = [_to_raw(pt) for pt in undetermined]

    _scatter(source, source_color)
    _scatter(saddle, saddle_color)
    _scatter(sink, sink_color)
    _scatter(center, center_color)
    _scatter_undetermined(undetermined)

    return [raw_source, raw_saddle, raw_sink, raw_center, raw_undetermined]


def plot_vector_field(
    payoff_data,
    ax,
    grid=15,
    margin=0.02,
    color="black",
    alpha=0.75,
    length=0.04,
    width=0.003,
    zorder=15,
    normalize=True,
):
    """Plot a sparse vector field for supported games."""
    game_class = infer_game_class(payoff_data)

    if game_class == "2P2S":
        points, vectors = _vector_field_2p2s(payoff_data, grid, margin)
        return _quiver_2d(ax, points, vectors, color, alpha, length, width, zorder, normalize)

    if game_class == "2P3S":
        points, vectors = _vector_field_2p3s(payoff_data, grid, margin)
        return _quiver_2d(ax, points, vectors, color, alpha, length, width, zorder, normalize)

    if game_class == "3P2S":
        points, vectors = _vector_field_3p2s(payoff_data, grid, margin)
        return _quiver_3d(ax, points, vectors, color, alpha, length, zorder, normalize)

    if game_class == "2P4S":
        points, vectors = _vector_field_2p4s(payoff_data, grid, margin)
        return _quiver_3d(ax, points, vectors, color, alpha, length, zorder, normalize)

    raise ValueError(f"Unsupported game class: {game_class}")


def plot_game(
    game,
    *,
    fig=None,
    ax=None,
    figsize=DEFAULT_PLOT_STYLE["figsize"],
    view_elev=DEFAULT_PLOT_STYLE["view_elev"],
    view_azim=DEFAULT_PLOT_STYLE["view_azim"],
    xlabel=None,
    ylabel=None,
    zlabel=None,
    starts=None,
    random_state=None,
    simplex_font_size=DEFAULT_PLOT_STYLE["simplex_font_size"],
    simplex_zorder=DEFAULT_PLOT_STYLE["simplex_zorder"],
    show_speed=DEFAULT_PLOT_STYLE["show_speed"],
    speed_grid=DEFAULT_PLOT_STYLE["speed_grid"],
    speed_cmap=DEFAULT_PLOT_STYLE["speed_cmap"],
    speed_levels=DEFAULT_PLOT_STYLE["speed_levels"],
    speed_zorder=DEFAULT_PLOT_STYLE["speed_zorder"],
    show_vector_field=DEFAULT_PLOT_STYLE["show_vector_field"],
    vector_grid=DEFAULT_PLOT_STYLE["vector_grid"],
    vector_margin=DEFAULT_PLOT_STYLE["vector_margin"],
    vector_color=DEFAULT_PLOT_STYLE["vector_color"],
    vector_alpha=DEFAULT_PLOT_STYLE["vector_alpha"],
    vector_length=DEFAULT_PLOT_STYLE["vector_length"],
    vector_width=DEFAULT_PLOT_STYLE["vector_width"],
    vector_zorder=DEFAULT_PLOT_STYLE["vector_zorder"],
    vector_normalize=DEFAULT_PLOT_STYLE["vector_normalize"],
    show_trajectories=DEFAULT_PLOT_STYLE["show_trajectories"],
    trajectory_step=DEFAULT_PLOT_STYLE["trajectory_step"],
    trajectory_arrows=None,
    tmax=DEFAULT_PLOT_STYLE["tmax"],
    trajectory_color=DEFAULT_PLOT_STYLE["trajectory_color"],
    arrow_size=DEFAULT_PLOT_STYLE["arrow_size"],
    arrow_width=DEFAULT_PLOT_STYLE["arrow_width"],
    trajectory_zorder=DEFAULT_PLOT_STYLE["trajectory_zorder"],
    show_equilibria=DEFAULT_PLOT_STYLE["show_equilibria"],
    sink_color=DEFAULT_PLOT_STYLE["sink_color"],
    saddle_color=DEFAULT_PLOT_STYLE["saddle_color"],
    source_color=DEFAULT_PLOT_STYLE["source_color"],
    center_color=DEFAULT_PLOT_STYLE["center_color"],
    equilibrium_size=DEFAULT_PLOT_STYLE["equilibrium_size"],
    equilibrium_zorder=DEFAULT_PLOT_STYLE["equilibrium_zorder"],
):
    """Plot a supported game with simplex/cube, trajectories, and equilibria.

    This is the high-level plotting entry point. The lower-level functions
    remain available for users who need full manual control.
    """
    payoff_data = _payoff_data(game)
    game_class = infer_game_class(game)
    labels = _strategy_labels(game, game_class)
    title = getattr(game, "name", None)
    trajectory_arrows = (
        list(DEFAULT_PLOT_STYLE["trajectory_arrows"])
        if trajectory_arrows is None
        else trajectory_arrows
    )

    fig, ax = _get_or_create_axes(fig, ax, game_class, figsize, view_elev, view_azim)

    draw_state_space(labels, payoff_data, ax, simplex_font_size, simplex_zorder)
    _set_default_axes_style(ax, game_class, labels)
    _apply_axis_label_overrides(ax, xlabel, ylabel, zlabel)

    if show_speed and game_class in ("2P2S", "2P3S"):
        x_region, y_region = _speed_regions(game_class)
        plot_speed_field(
            x_region=x_region,
            y_region=y_region,
            step=speed_grid,
            ax=ax,
            payoff_data=payoff_data,
            cmap=speed_cmap,
            levels=speed_levels,
            zorder=speed_zorder,
        )

    if show_vector_field:
        plot_vector_field(
            payoff_data,
            ax,
            grid=vector_grid,
            margin=vector_margin,
            color=vector_color,
            alpha=vector_alpha,
            length=vector_length,
            width=vector_width,
            zorder=vector_zorder,
            normalize=vector_normalize,
        )

    if show_equilibria:
        plot_equilibria(
            payoff_data,
            ax,
            sink_color=sink_color,
            saddle_color=saddle_color,
            source_color=source_color,
            center_color=center_color,
            equilibrium_size=equilibrium_size,
            zorder=equilibrium_zorder,
        )

    if show_trajectories:
        starts = _default_starts(game_class, random_state) if starts is None else starts
        trajectory_colors = _trajectory_colors(trajectory_color, len(starts))
        for start, color in zip(starts, trajectory_colors):
            plot_trajectory(
                start,
                payoff_data,
                time_step=trajectory_step,
                arrow_positions=trajectory_arrows,
                tmax=tmax,
                fig=fig,
                ax=ax,
                trajectory_color=color,
                arrow_size=arrow_size,
                arrow_width=arrow_width,
                zorder=trajectory_zorder,
                arrow_color=color,
            )

    if title:
        ax.set_title(title)

    return fig, ax


def _payoff_data(game):
    return getattr(game, "payoff_data", game)


def _strategy_labels(game, game_class):
    labels = getattr(game, "strategy_labels", None)
    if labels:
        return labels
    if game_class == "2P2S":
        return ["Strategy 1", "Strategy 2"]
    if game_class == "3P2S":
        return ["Population 1", "Population 2", "Population 3"]
    if game_class == "2P4S":
        return ["Strategy 1", "Strategy 2", "Strategy 3", "Strategy 4"]
    return ["Strategy 1", "Strategy 2", "Strategy 3"]


def _trajectory_colors(trajectory_color, count):
    if mcolors.is_color_like(trajectory_color):
        return [trajectory_color] * count
    if len(trajectory_color) != count:
        raise ValueError(
            "trajectory_color must be a single Matplotlib color or a list of "
            "colors with the same length as starts."
        )
    if not all(mcolors.is_color_like(color) for color in trajectory_color):
        raise ValueError("Every entry in trajectory_color must be a Matplotlib color.")
    return trajectory_color


def _apply_axis_label_overrides(ax, xlabel, ylabel, zlabel):
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if zlabel is not None and hasattr(ax, "set_zlabel"):
        ax.set_zlabel(zlabel)


def _get_or_create_axes(fig, ax, game_class, figsize, view_elev, view_azim):
    if ax is not None:
        return ax.figure, ax

    if fig is None:
        fig = plt.figure(figsize=figsize)

    if game_class in ("2P4S", "3P2S"):
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=view_elev, azim=view_azim)
    else:
        ax = fig.add_subplot(111)
        ax.set_aspect(1)

    return fig, ax


def _set_default_axes_style(ax, game_class, labels):
    if game_class == "2P2S":
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Population 1: Pr({labels[0]})")
        ax.set_ylabel(f"Population 2: Pr({labels[0]})")
    elif game_class == "2P3S":
        ax.axis("off")
    elif game_class == "2P4S":
        ax.set_axis_off()


def _speed_regions(game_class):
    if game_class == "2P3S":
        return [0, 1], [0, np.sqrt(3 / 4)]
    return [0, 1], [0, 1]


def _default_starts(game_class, random_state):
    rng = np.random.default_rng(random_state)
    if game_class == "2P3S":
        return rng.dirichlet(np.ones(3), size=4)[:, :2].tolist()
    if game_class == "2P4S":
        return rng.dirichlet(np.ones(4), size=4)[:, :3].tolist()
    if game_class in ("2P2S", "3P2S"):
        return rng.uniform(0.05, 0.95, size=(4, 2 if game_class == "2P2S" else 3)).tolist()
    raise ValueError(f"Unsupported game class: {game_class}")


def _vector_field_2p2s(payoff_data, grid, margin):
    values = np.linspace(margin, 1.0 - margin, grid)
    points = []
    vectors = []
    for x_val in values:
        for y_val in values:
            point = np.array([x_val, y_val])
            vector = np.asarray(dynamics.replicator_2p2s(point, 0, payoff_data), dtype=float)
            points.append(point)
            vectors.append(vector)
    return np.asarray(points), np.asarray(vectors)


def _vector_field_2p3s(payoff_data, grid, margin):
    bary_points = _simplex_grid_3(grid, margin)
    points = []
    vectors = []
    for r, p, _ in bary_points:
        vector = np.asarray(dynamics.replicator_2p3s([r, p], 0, payoff_data), dtype=float)
        start = np.asarray(p_to_sim(r, p), dtype=float)
        end = np.asarray(p_to_sim(r + vector[0], p + vector[1]), dtype=float)
        points.append(start)
        vectors.append(end - start)
    return np.asarray(points), np.asarray(vectors)


def _vector_field_3p2s(payoff_data, grid, margin):
    values = np.linspace(margin, 1.0 - margin, grid)
    points = []
    vectors = []
    for x_val in values:
        for y_val in values:
            for z_val in values:
                point = np.array([x_val, y_val, z_val])
                vector = np.asarray(dynamics.replicator_3p2s(point, 0, payoff_data), dtype=float)
                points.append(point)
                vectors.append(vector)
    return np.asarray(points), np.asarray(vectors)


def _vector_field_2p4s(payoff_data, grid, margin):
    bary_points = _simplex_grid_4(grid, margin)
    points = []
    vectors = []
    for a, b, c, _ in bary_points:
        vector = np.asarray(dynamics.replicator_2p4s([a, b, c], 0, payoff_data), dtype=float)
        start = np.asarray(sim_to_p_2p4s(a, b, c), dtype=float)
        end = np.asarray(sim_to_p_2p4s(a + vector[0], b + vector[1], c + vector[2]), dtype=float)
        points.append(start)
        vectors.append(end - start)
    return np.asarray(points), np.asarray(vectors)


def _simplex_grid_3(grid, margin):
    points = []
    values = np.linspace(margin, 1.0 - margin, grid)
    for a in values:
        for b in values:
            c = 1.0 - a - b
            if c >= margin:
                points.append([a, b, c])
    return np.asarray(points)


def _simplex_grid_4(grid, margin):
    points = []
    values = np.linspace(margin, 1.0 - margin, grid)
    for a in values:
        for b in values:
            for c in values:
                d = 1.0 - a - b - c
                if d >= margin:
                    points.append([a, b, c, d])
    return np.asarray(points)


def _normalize_vectors(vectors):
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    nonzero = norms > 1e-12
    normalized = np.zeros_like(vectors)
    normalized[nonzero] = vectors[nonzero] / norms[nonzero, None]
    return normalized, nonzero


def _quiver_2d(ax, points, vectors, color, alpha, length, width, zorder, normalize):
    if normalize:
        vectors, keep = _normalize_vectors(vectors)
        points = points[keep]
        vectors = vectors[keep]
        vectors = vectors * length
    if len(points) == 0:
        return None
    return ax.quiver(
        points[:, 0],
        points[:, 1],
        vectors[:, 0],
        vectors[:, 1],
        color=color,
        alpha=alpha,
        width=width,
        zorder=zorder,
        angles="xy",
        scale_units="xy",
        scale=1,
    )


def _quiver_3d(ax, points, vectors, color, alpha, length, zorder, normalize):
    if normalize:
        vectors, keep = _normalize_vectors(vectors)
        points = points[keep]
        vectors = vectors[keep]
    if len(points) == 0:
        return None
    return ax.quiver(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        vectors[:, 0],
        vectors[:, 1],
        vectors[:, 2],
        color=color,
        alpha=alpha,
        length=length,
        normalize=False,
        zorder=zorder,
    )


def _matrix_to_colors(matrix, cmap):
    """Converts a matrix into a RGBA color map."""
    color_dimension = matrix # It must be in 2D - as for "X, Y, Z".
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)
    return fcolors, m


def _outside_2p3s_simplex(X, Y):
    """Boolean mask for points lying outside the 2P3S simplex in plotting coordinates."""
    mask = np.zeros(X.shape, dtype=bool)
    for i in range(len(X)):
        for j in range(len(X)):
            if X[i, j] <= 0.5:
                mask[i, j] = Y[i, j] > 2 * np.sqrt(3 / 4) * X[i, j]
            elif X[i, j] > 0.5:
                mask[i, j] = Y[i, j] > 2 * np.sqrt(3 / 4) * (1 - X[i, j])
            else:
                mask[i, j] = False
    return mask


def _project_to_2p3s_simplex(X, Y):
    """Orthogonally project out-of-bounds grid points back to the 2P3S simplex."""
    mask = _outside_2p3s_simplex(X, Y)
    for i in range(len(X)):
        for j in range(len(Y)):
            if mask[i][j]:
                if X[i, j] < 0.5:
                    xB, yB = 0, 0
                    xV, yV = 1, 2 * np.sqrt(3 / 4)
                    BH = ((X[i, j] - xB) * xV + (Y[i, j] - yB) * yV) / (np.sqrt(xV**2 + yV**2))
                    X[i, j] = xB + (BH / np.sqrt(xV**2 + yV**2)) * xV
                    Y[i, j] = 2 * np.sqrt(3 / 4) * X[i, j]
                elif X[i, j] > 0.5:
                    xB, yB = 0.5, np.sqrt(3 / 4)
                    xV, yV = 1, -2 * np.sqrt(3 / 4)
                    BH = ((X[i, j] - xB) * xV + (Y[i, j] - yB) * yV) / (np.sqrt(xV**2 + yV**2))
                    X[i, j] = xB + (BH / np.sqrt(xV**2 + yV**2)) * xV
                    Y[i, j] = 2 * np.sqrt(3 / 4) * (1 - X[i, j])
    return X, Y


def _speed_2p3s(x, y, payoff_data):
    """Speed magnitude of the 2P3S replicator dynamics at plotting coordinates (x, y)."""
    r, p = sim_to_p(x, y)
    vector = np.asarray(dynamics.replicator_2p3s([r, p], 0, payoff_data), dtype=float)
    start = np.asarray(p_to_sim(r, p), dtype=float)
    end = np.asarray(p_to_sim(r + vector[0], p + vector[1]), dtype=float)
    return np.linalg.norm(end - start)


def _speed_grid_2p3s(X, Y, payoff_data):
    """Fill a grid with speeds for 2P3S replicator dynamics."""
    CALC = np.zeros(X.shape)
    for i in range(len(X)):
        for j in range(len(Y)):
            CALC[i][j] = _speed_2p3s(X[i][j], Y[i][j], payoff_data)
    return CALC


def _speed_grid_2p2s(U, V, payoff_data):
    """Fill a grid with speeds for 2P2S replicator dynamics."""
    CALC = np.zeros(U.shape)
    for i in range(len(U)):
        for j in range(len(V)):
            x = U[i][j]
            y = V[i][j]
            vector = dynamics.replicator_2p2s([x, y], 0, payoff_data)
            CALC[i][j] = np.linalg.norm(vector)
    return CALC


def plot_speed_field(x_region, y_region, step, payoff_data, ax, cmap, levels, zorder):
    """Plots movement speed for supported games."""
    x = np.linspace(x_region[0], x_region[1], step)
    y = np.linspace(y_region[0], y_region[1], step)
    X, Y = np.meshgrid(x, y)

    if payoff_data[0].shape == (3,):  # symmetric 2P3S
        X, Y = _project_to_2p3s_simplex(X, Y)
        C = _speed_grid_2p3s(X, Y, payoff_data)
        surf = ax.contourf(
            X,
            Y,
            C,
            levels=levels,
            cmap=cmap,
            corner_mask=False,
            alpha=0.9,
            zorder=zorder,
        )
        return surf

    if payoff_data[0].shape == (2, 2):  # asymmetric 2P2S
        C = _speed_grid_2p2s(X, Y, payoff_data)
        surf = ax.contourf(
            X,
            Y,
            C,
            levels=levels,
            cmap=cmap,
            corner_mask=False,
            alpha=0.9,
            zorder=zorder,
        )
        return surf

    raise NotImplementedError("Speed plot currently supports only 2P3S and 2P2S games.")
