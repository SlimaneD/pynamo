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
import numpy as np
from scipy.integrate import odeint
from sympy import Matrix
from sympy.abc import x, y, z

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D

import equationsolver as eqsol
import dynamics


def arrow_dyn2(xStart, xEnd, fig, ax, arrow_size, arrow_width, arrow_color, zOrder):
    """Creates a polygon defined by the shape of the arrow"""
    cf=arrow_width
    af=arrow_size
    x0= xStart
    xA= xEnd
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
        xF = [eqsol.solF(xA[0], xA[1], sf, af)[0][0], eqsol.solF(xA[0], xA[1], sf, af)[0][1]]
        xB = [eqsol.solB(xF[0], xF[1], sf, cf)[1][0], eqsol.solB(xF[0], xF[1], sf, cf)[1][1]]
        xC = [eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[0][0], eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[0][1]]
        xD = [eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[1][0], eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[1][1]]
    elif(xA[0]<x0[0]):
        sf = (xA[1]-x0[1])/(xA[0]-x0[0])
        xF = [eqsol.solF(xA[0], xA[1], sf, af)[1][0], eqsol.solF(xA[0], xA[1], sf, af)[1][1]]
        xB = [eqsol.solB(xF[0], xF[1], sf, cf)[0][0], eqsol.solB(xF[0], xF[1], sf, cf)[0][1]]
        xC = [eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[0][0], eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[0][1]]
        xD = [eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[1][0], eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[1][1]]
    xs = [x0[0], xA[0]]
    ys = [x0[1], xA[1]]
    arrLine = plt.plot(xs, ys, color=arrow_color, zorder=zOrder, clip_on=False)
    arrow = [xA, xC, xB, xD]
    verts = []
    patches = []
    for pt in arrow:
        verts.append([pt[0], pt[1]])
    arrHead = Polygon(verts)
    patches.append(arrHead)
    p = PatchCollection(patches, facecolor=arrow_color, edgecolor=arrow_color, alpha=1, zorder=zOrder)
    ax.add_collection(p)
    return arrLine+[arrHead]


def arrow_dyn3(xStart, xEnd, fig, ax, arrow_size, arrow_width, arrow_color, zOrder):
    """Creates arrow with the default quiver3d from matplotlib"""
    u = xEnd[0] - xStart[0]
    v = xEnd[1] - xStart[1]
    w = xEnd[2] - xStart[2]
    quiv = ax.quiver(xStart[0], xStart[1], xStart[2], u, v, w, length=0.002, arrow_length_ratio=15, pivot='tip', color = arrow_color, zorder=zOrder, normalize = True)
    return [quiv]


def _is_three_population_cube(payMtx):
    return isinstance(payMtx, (tuple, list)) and hasattr(payMtx[0], "ndim") and payMtx[0].ndim == 3



def setSimplex(strat, payMtx, ax, fontSize, zOrder):
    """Draws the simplex frame."""
    if payMtx[0].shape == (3,):
        pt1 = eqsol.p_to_sim(1, 0)
        pt2 = eqsol.p_to_sim(0, 1)
        pt3 = eqsol.p_to_sim(0, 0)
        lbl1 = ax.annotate(strat[0], (pt1[0] - 0.01, pt1[1] + 0.04), fontsize=fontSize, zorder=zOrder)
        lbl2 = ax.annotate(strat[1], (pt2[0] - 0.05, pt2[1] - 0.01), fontsize=fontSize, zorder=zOrder)
        lbl3 = ax.annotate(strat[2], (pt3[0] + 0.03, pt3[1] - 0.01), fontsize=fontSize, zorder=zOrder)
        xs = ([pt1[0], pt2[0]], [pt1[0], pt3[0]], [pt2[0], pt3[0]])
        ys = ([pt1[1], pt2[1]], [pt1[1], pt3[1]], [pt2[1], pt3[1]])
        lines = []
        for xpair, ypair in zip(xs, ys):
            lines += plt.plot(xpair, ypair, color='black', zorder=zOrder, alpha=1, clip_on=False)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, (3 ** 0.5) / 2 + 0.05)
        ax.set_aspect('equal', adjustable='box')
        return lines + [lbl1, lbl2, lbl3]
    if payMtx[0].shape == (2, 2):
        ax.set_xlabel(strat[0], fontsize=fontSize)
        ax.set_ylabel(strat[1], fontsize=fontSize)
        edges = [([0, 1], [0, 0]), ([1, 1], [0, 1]), ([1, 0], [1, 1]), ([0, 0], [1, 0])]
        lines = []
        for xs, ys in edges:
            lines += plt.plot(xs, ys, color='black', zorder=zOrder, alpha=1, clip_on=False)
        return lines
    if _is_three_population_cube(payMtx):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel(strat[0], fontsize=fontSize)
        ax.set_ylabel(strat[1], fontsize=fontSize)
        ax.set_zlabel(strat[2], fontsize=fontSize)
        ax.grid(False)
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
            artists += ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color='black', zorder=zOrder, alpha=1)
        return artists
    if payMtx[0].shape == (4,):
        pt1 = eqsol.sim_to_p_2P4S(1, 0, 0)
        pt2 = eqsol.sim_to_p_2P4S(0, 1, 0)
        pt3 = eqsol.sim_to_p_2P4S(0, 0, 1)
        pt4 = eqsol.sim_to_p_2P4S(0, 0, 0)
        lbl1 = ax.text(pt1[0], pt1[1] + 0.05, pt1[2], strat[0], fontsize=fontSize, zorder=zOrder)
        lbl2 = ax.text(pt2[0] - 0.05, pt2[1], pt2[2], strat[1], fontsize=fontSize, zorder=zOrder)
        lbl3 = ax.text(pt3[0] + 0.05, pt3[1] - 0.022, pt3[2], strat[2], fontsize=fontSize, zorder=zOrder)
        lbl4 = ax.text(pt4[0] - 0.02, pt4[1] - 0.022, pt4[2] + 0.05, strat[3], fontsize=fontSize, zorder=zOrder)
        xs = [[pt1[0], pt2[0]], [pt2[0], pt3[0]], [pt3[0], pt1[0]], [pt4[0], pt1[0]], [pt4[0], pt2[0]], [pt4[0], pt3[0]]]
        ys = [[pt1[1], pt2[1]], [pt2[1], pt3[1]], [pt3[1], pt1[1]], [pt4[1], pt1[1]], [pt4[1], pt2[1]], [pt4[1], pt3[1]]]
        zs = [[pt1[2], pt2[2]], [pt2[2], pt3[2]], [pt3[2], pt1[2]], [pt4[2], pt1[2]], [pt4[2], pt2[2]], [pt4[2], pt3[2]]]
        lines = []
        for xpair, ypair, zpair in zip(xs, ys, zs):
            lines += plt.plot(xpair, ypair, zpair, color='black', zorder=zOrder, alpha=1, clip_on=False)
        return lines + [lbl1, lbl2, lbl3, lbl4]
    return []


def trajectory(X0, payMtx, step, parr, Tmax, fig, ax, col, arrSize, arrWidth, zd, arrow_color=None):
    """Draws trajectories in the simplex, given a starting point."""
    t = np.linspace(0, Tmax, int(Tmax / step))
    line_color = col or 'black'
    arrow_col = arrow_color if arrow_color is not None else (col if col is not None else 'k')

    if payMtx[0].shape == (3,):  # symmetric 2P3S
        x0, y0 = X0
        sol = odeint(dynamics.repDyn3, [x0, y0], t, (payMtx,))
        solRev = odeint(dynamics.repDyn3Rev, [x0, y0], t, (payMtx,))
        solX = []
        solY = []
        solXrev = []
        solYrev = []
        for pt in sol:
            cPt = eqsol.p_to_sim(pt[0], pt[1])
            solX.append(cPt[0])
            solY.append(cPt[1])
        for pt in solRev:
            cPt = eqsol.p_to_sim(pt[0], pt[1])
            solXrev.append(cPt[0])
            solYrev.append(cPt[1])
        psol = plt.plot(solX, solY, color=line_color, zorder=zd, clip_on=False)
        psolRev = plt.plot(solXrev, solYrev, color=line_color, zorder=zd, clip_on=False)
        dirs = []
        for frac in parr:
            base = min(max(int(frac * (len(solX) - 1)), 0), len(solX) - 2)
            dirs += arrow_dyn2(
                [solX[base], solY[base]],
                [solX[base + 1], solY[base + 1]],
                fig,
                ax,
                arrow_width=arrWidth,
                arrow_size=arrSize,
                arrow_color=arrow_col,
                zOrder=zd,
            )
        return psol + psolRev + dirs

    if payMtx[0].shape == (2, 2):  # asymmetric 2P2S
        x0, y0 = X0
        sol = odeint(dynamics.testrep, [x0, y0], t, (payMtx,))
        solRev = odeint(dynamics.testrepRev, [x0, y0], t, (payMtx,))
        solX, solY = sol[:, 0], sol[:, 1]
        solXrev, solYrev = solRev[:, 0], solRev[:, 1]
        psol = plt.plot(solX, solY, color=line_color, zorder=zd, clip_on=False)
        psolRev = plt.plot(solXrev, solYrev, color=line_color, zorder=zd, clip_on=False)
        dirs = []
        for frac in parr:
            base = min(max(int(frac * (len(solX) - 1)), 0), len(solX) - 2)
            dirs += arrow_dyn2(
                [solX[base], solY[base]],
                [solX[base + 1], solY[base + 1]],
                fig,
                ax,
                arrow_width=arrWidth,
                arrow_size=arrSize,
                arrow_color=arrow_col,
                zOrder=zd,
            )
        return psol + psolRev + dirs

    if _is_three_population_cube(payMtx):
        x0, y0, z0 = X0
        sol = odeint(dynamics.repDyn3Pop2, [x0, y0, z0], t, (payMtx,))
        solRev = odeint(dynamics.repDyn3Pop2Rev, [x0, y0, z0], t, (payMtx,))
        solX, solY, solZ = sol[:, 0], sol[:, 1], sol[:, 2]
        solXrev, solYrev, solZrev = solRev[:, 0], solRev[:, 1], solRev[:, 2]
        psol = ax.plot(solX, solY, solZ, linewidth=0.8, color=line_color, zorder=zd)
        psolRev = ax.plot(solXrev, solYrev, solZrev, linewidth=0.4, color=line_color, zorder=zd)
        dirs = []
        for frac in parr:
            base = min(max(int(frac * (len(solX) - 1)), 0), len(solX) - 2)
            dirs += arrow_dyn3(
                [solX[base], solY[base], solZ[base]],
                [solX[base + 1], solY[base + 1], solZ[base + 1]],
                fig,
                ax,
                arrow_width=arrWidth,
                arrow_size=arrSize,
                arrow_color=arrow_col,
                zOrder=zd,
            )
        return psol + psolRev + dirs

    if payMtx[0].shape == (4,):
        x0, y0, z0 = X0
        sol = odeint(dynamics.repDyn4, [x0, y0, z0], t, (payMtx,))
        solRev = odeint(dynamics.repDyn4Rev, [x0, y0, z0], t, (payMtx,))
        solX, solY, solZ = [], [], []
        solXrev, solYrev, solZrev = [], [], []
        for pt in sol:
            cPt = eqsol.sim_to_p_2P4S(pt[0], pt[1], pt[2])
            solX.append(cPt[0])
            solY.append(cPt[1])
            solZ.append(cPt[2])
        for pt in solRev:
            cPt = eqsol.sim_to_p_2P4S(pt[0], pt[1], pt[2])
            solXrev.append(cPt[0])
            solYrev.append(cPt[1])
            solZrev.append(cPt[2])
        psol = ax.plot(solX, solY, solZ, linewidth=0.8, color=line_color, zorder=zd)
        psolRev = ax.plot(solXrev, solYrev, solZrev, linewidth=0.4, color=line_color, zorder=zd)
        dirs = []
        for frac in parr:
            base = min(max(int(frac * (len(solX) - 1)), 0), len(solX) - 2)
            dirs += arrow_dyn3(
                [solX[base], solY[base], solZ[base]],
                [solX[base + 1], solY[base + 1], solZ[base + 1]],
                fig,
                ax,
                arrow_width=arrWidth,
                arrow_size=arrSize,
                arrow_color=arrow_col,
                zOrder=zd,
            )
        return psol + psolRev + dirs

    return None


def equilibria(payMtx, ax, colSnk, colSdl, colSce, ptSize, zd):
    """Computes the equilibrium points of the game and characterizes them."""
    source, sink, saddle, centre, undet = [], [], [], [], []
    numEqs, numEig = [], []
    three_pop = _is_three_population_cube(payMtx)

    nuEqsRaw = eqsol.solGame(payMtx)

    if payMtx[0].shape == (3,):
        for eq in nuEqsRaw:
            if 0 <= eq[0] <= 1 and 0 <= eq[1] <= 1 and eq[0] + eq[1] <= 1:
                numEqs.append([eq[0], eq[1]])
        for eq in numEqs:
            if getattr(eq[0], 'imag', 0) != 0 or getattr(eq[1], 'imag', 0) != 0:
                eq[0], eq[1] = 99, 99
        for eq in numEqs:
            t = 0
            X = Matrix(dynamics.repDyn3([x, y], t, payMtx))
            Y = Matrix([x, y])
            JC = X.jacobian(Y)
            valuedJC = np.array(JC.subs([(x, eq[0]), (y, eq[1])]))
            M = np.array(valuedJC, dtype=float)
            w, _ = np.linalg.eig(M)
            numEig.append(w)

    elif payMtx[0].shape == (2, 2):
        for eq in nuEqsRaw:
            if 0 <= eq[0] <= 1 and 0 <= eq[1] <= 1:
                numEqs.append([eq[0], eq[1]])
        for eq in numEqs:
            if getattr(eq[0], 'imag', 0) != 0 or getattr(eq[1], 'imag', 0) != 0:
                eq[0], eq[1] = 99, 99
        for eq in numEqs:
            t = 0
            X = Matrix(dynamics.testrep([x, y], t, payMtx))
            Y = Matrix([x, y])
            JC = X.jacobian(Y)
            valuedJC = np.array(JC.subs([(x, eq[0]), (y, eq[1])]))
            M = np.array(valuedJC, dtype=float)
            w, _ = np.linalg.eig(M)
            numEig.append(w)

    elif three_pop:
        for eq in nuEqsRaw:
            coords = [float(np.real(c)) for c in eq]
            if all(0 <= c <= 1 for c in coords):
                numEqs.append(coords)
                numEig.append(np.zeros(3))

    elif payMtx[0].shape == (4,):
        for eq in nuEqsRaw:
            if 0 <= eq[0] <= 1 and 0 <= eq[1] <= 1 and 0 <= eq[2] <= 1 and eq[0] + eq[1] + eq[2] <= 1:
                numEqs.append([eq[0], eq[1], eq[2]])
        for eq in numEqs:
            if any(getattr(val, 'imag', 0) != 0 for val in eq):
                eq[0], eq[1], eq[2] = 99, 99, 99
        for eq in numEqs:
            t = 0
            X = Matrix(dynamics.repDyn4([x, y, z], t, payMtx))
            Y = Matrix([x, y, z])
            JC = X.jacobian(Y)
            valuedJC = np.array(JC.subs([(x, eq[0]), (y, eq[1]), (z, eq[2])]))
            M = np.array(valuedJC, dtype=float)
            w, _ = np.linalg.eig(M)
            numEig.append(w)

    def _round_value(val, ndigits):
        if np.iscomplexobj(val):
            real_part = round(float(np.real(val)), ndigits)
            imag_part = round(float(np.imag(val)), ndigits)
            if imag_part == 0:
                return real_part
            return complex(real_part, imag_part)
        return round(float(val), ndigits)

    def _jacobian_three_pop(point):
        base = np.clip(np.asarray(point, dtype=float), 0.0, 1.0)
        f0 = dynamics.repDyn3Pop2(base, 0, payMtx)
        jac = np.zeros((3, 3), dtype=float)
        eps = 1e-6
        for axis in range(3):
            pert = base.copy()
            pert[axis] = np.clip(pert[axis] + eps, 0.0, 1.0)
            f1 = dynamics.repDyn3Pop2(pert, 0, payMtx)
            jac[:, axis] = (f1 - f0) / eps
        return jac

    def _classify_three_pop(point):
        try:
            jac = _jacobian_three_pop(point)
            eigvals = np.linalg.eigvals(jac)
        except Exception:
            return 'undet'
        real_parts = np.real(eigvals)
        tol = 1e-6
        if np.all(real_parts < -tol):
            return 'sink'
        if np.all(real_parts > tol):
            return 'source'
        if np.any(real_parts > tol) and np.any(real_parts < -tol):
            return 'saddle'
        if np.all(np.abs(real_parts) <= tol):
            return 'centre'
        return 'undet'

    for i in range(len(numEqs)):
        numEqs[i] = [_round_value(val, 10) for val in numEqs[i]]
        if i < len(numEig):
            numEig[i] = [_round_value(val, 12) for val in numEig[i]]
        if payMtx[0].shape == (2, 2):
            point_to_plot = np.array([numEqs[i][0], numEqs[i][1]])
        elif payMtx[0].shape == (3,):
            point_to_plot = np.array(eqsol.p_to_sim(numEqs[i][0], numEqs[i][1]))
            numEqs[i].append(1 - numEqs[i][0] - numEqs[i][1])
        elif three_pop:
            point_to_plot = np.array(numEqs[i])
            stability = _classify_three_pop(point_to_plot)
            if stability == 'sink':
                sink.append(point_to_plot)
            elif stability == 'source':
                source.append(point_to_plot)
            elif stability == 'saddle':
                saddle.append(point_to_plot)
            elif stability == 'centre':
                centre.append(point_to_plot)
            else:
                undet.append(point_to_plot)
            continue
        elif payMtx[0].shape == (4,):
            point_to_plot = np.array(eqsol.sim_to_p_2P4S(numEqs[i][0], numEqs[i][1], numEqs[i][2]))
        else:
            continue

        if payMtx[0].shape == (2, 2) or payMtx[0].shape == (3,):
            l1, l2 = numEig[i][0], numEig[i][1]
            suml, prodl = l1 + l2, l1 * l2
            if isinstance(prodl, complex):
                if np.imag(prodl) != 0:
                    centre.append(point_to_plot)
                    continue
                prodl = np.real(prodl)
            if isinstance(suml, complex):
                if np.imag(suml) != 0:
                    centre.append(point_to_plot)
                    continue
                suml = np.real(suml)
            if prodl < 0:
                saddle.append(point_to_plot)
            else:
                if suml > 0:
                    source.append(point_to_plot)
                elif suml < 0:
                    sink.append(point_to_plot)
                else:
                    if getattr(l1, 'imag', 0) != 0:
                        centre.append(point_to_plot)
                    else:
                        undet.append(point_to_plot)
        else:
            centre.append(point_to_plot)

    def _to_raw(point):
        if payMtx[0].shape == (3,):
            r, p = eqsol.sim_to_p(point[0], point[1])
            return [r, p, 1 - r - p]
        if payMtx[0].shape == (2, 2):
            return point.tolist()
        if three_pop or payMtx[0].shape == (4,):
            return point.tolist()
        return point.tolist()

    def _sphere_radius(size):
        return np.clip(np.sqrt(size) / 200.0, 0.01, 0.08)

    def _plot_spheres(points, color):
        if points.size == 0:
            return
        radius = _sphere_radius(ptSize)
        rgba = mcolors.to_rgba(color, alpha=0.85)
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
                zorder=zd,
            )

    def _scatter(points, color, marker='o'):
        if not points:
            return
        pts = np.array(points)
        if three_pop or payMtx[0].shape == (4,):
            _plot_spheres(pts, color)
        else:
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=ptSize,
                color=color,
                marker=marker,
                edgecolors='black',
                alpha=1,
                zorder=zd,
                clip_on=False,
            )

    raw_source = [_to_raw(pt) for pt in source]
    raw_saddle = [_to_raw(pt) for pt in saddle]
    raw_sink = [_to_raw(pt) for pt in sink]
    raw_centre = [_to_raw(pt) for pt in centre]
    raw_undet = [_to_raw(pt) for pt in undet]

    _scatter(source, colSce)
    _scatter(saddle, colSdl)
    _scatter(sink, colSnk)
    _scatter(centre, colSce)
    _scatter(undet, colSnk)

    return [raw_source, raw_saddle, raw_sink, raw_centre, raw_undet]
def matrix_to_colors(matrix, cmap):
    """Converts a matrix into a RGBA color map."""
    color_dimension = matrix # It must be in 2D - as for "X, Y, Z".
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)
    return fcolors, m

def speed_plot(x_region, y_region, step, payMtx, ax, cmap, levels, zd):
    """Plots movement speed for supported games."""
    x = np.linspace(x_region[0], x_region[1], step)
    y = np.linspace(y_region[0], y_region[1], step)
    X, Y = np.meshgrid(x, y)

    if payMtx[0].shape == (3,):  # symmetric 2P3S
        X, Y = eqsol.outofbounds_reproject(X, Y)
        C = eqsol.speedGrid(X, Y, payMtx)
        surf = ax.contourf(
            X,
            Y,
            C,
            levels=levels,
            cmap=cmap,
            corner_mask=False,
            alpha=0.9,
            zorder=zd,
        )
        return surf

    if payMtx[0].shape == (2, 2):  # asymmetric 2P2S
        C = eqsol.speedGrid2P2S(X, Y, payMtx)
        surf = ax.contourf(
            X,
            Y,
            C,
            levels=levels,
            cmap=cmap,
            corner_mask=False,
            alpha=0.9,
            zorder=zd,
        )
        return surf

    raise NotImplementedError("Speed plot currently supports only 2P3S and 2P2S games.")
