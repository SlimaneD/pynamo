# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:18:00 2020

@author: Benjamin Giraudon

Status : OK
"""

import warnings

import numpy as np


class DegenerateEquilibriumWarning(RuntimeWarning):
    """Warning raised when equilibrium sets are non-isolated."""


warnings.simplefilter("always", DegenerateEquilibriumWarning)


class EquilibriumResult(list):
    """List of isolated equilibria with metadata about non-isolated solutions."""

    def __init__(self, equilibria=None, *, degenerate=False, message=None):
        super().__init__(equilibria or [])
        self.degenerate = degenerate
        self.message = message

# Replicator dynamics for a symmetric 2P3S game
def w3(x1, x2, y_1, y_2, payMtx):
    ''' Average payoff of a (x1, x2) strategy against a (y1, y2) strategy in symmetric 2P3S games'''
    return x1*(y_1*payMtx[0][0] + y_2*payMtx[0][1] + (1-y_1-y_2)*payMtx[0][2]) + x2*(y_1*payMtx[1][0] + y_2*payMtx[1][1] + (1-y_1-y_2)*payMtx[1][2]) + (1-x1-x2)*(y_1*payMtx[2][0] + y_2*payMtx[2][1] + (1-y_1-y_2)*payMtx[2][2])

def repDyn3(X, t, payMtx):
    '''Computes the replicator dynamics for a symmetric 2-player 3-strategy game (2P3S)'''
    x1, x2 = X
    Pbar = w3(x1, x2, x1, x2, payMtx)
    return np.array([x1*(w3(1, 0, x1, x2, payMtx) - Pbar), x2*(w3(0, 1, x1, x2, payMtx) - Pbar)])

def repDyn3Speed(x1, x2, payMtx):
    '''Computes the replicator dynamics for a symmetric 2-player 3-strategy game (2P3S) | adapted for speed grids'''
    Pbar = w3(x1, x2, x1, x2, payMtx)
    return np.array([x1*(w3(1, 0, x1, x2, payMtx) - Pbar), x2*(w3(0, 1, x1, x2, payMtx) - Pbar)])

def repDyn3Rev(X, t, payMtx):
    '''Computes the opposite replicator dynamics for a symmetric 2-player 3-strategy game (2P3S)'''
    x1, x2 = X
    Pbar = w3(x1, x2, x1, x2, payMtx)
    return np.array([-x1*(w3(1, 0, x1, x2, payMtx) - Pbar), -x2*(w3(0, 1, x1, x2, payMtx) - Pbar)])

# Replicator dynamics for an asymmetric 2P2S game
def repDyn22(X, t, payMtx):
    '''Computes the replicator dynamics for an asymmetric 2P2S game'''
    x, y = X
    return x*(1-x)*( (y*payMtx[0][0] + (1-y)*payMtx[0][1]) - (y*payMtx[1][0] + (1-y)*payMtx[1][1]))

def repDyn22Rev(X, t, payMtx):
    '''Computes the opposite replicator dynamics for an asymmetric 2P2S game'''
    x, y = X
    return -x*(1-x)*( (y*payMtx[0][0] + (1-y)*payMtx[0][1]) - (y*payMtx[1][0] + (1-y)*payMtx[1][1]))

def repDyn22_vector(X, t, payMtx):
    """Vector field for asymmetric 2P2S replicator dynamics (both populations)."""
    x, y = X[0], X[1]
    return [repDyn22([x, y], t, payMtx[0]), repDyn22([y, x], t, payMtx[1])]

def repDyn22_vector_rev(X, t, payMtx):
    """Opposite vector field for asymmetric 2P2S replicator dynamics (both populations)."""
    x, y = X[0], X[1]
    return [repDyn22Rev([x, y], t, payMtx[0]), repDyn22Rev([y, x], t, payMtx[1])]

# Replicator dynamics for 2P4S game

def w4(x1, x2, x3, y1, y2, y3, payMtx):
    ''' Average payoff of a (x1, x2, x3) strategy against a (y1, y2, y3) strategy in symmetric 2P4S games'''
#    X = np.array([x1, x2, x3, 1 - x1 - x2 - x3])
#    Y = np.array([ [y1], [y2], [y3], [1 - y1 - y2 - y3] ])
#    PY = np.dot(payMtx, Y)
#    sumT = np.dot(X, PY)[0]
#    print("sumT", sumT)
#    test = x1*(y1*payMtx[0, 0] + y2*payMtx[0, 1] + y3*payMtx[0, 2] + (1 - y1 - y2 - y3)*payMtx[0, 3]) + x2*(y1*payMtx[1, 0] + y2*payMtx[1, 1] + y3*payMtx[1, 2] + (1 - y1 - y2 - y3)*payMtx[1, 3]) + x3*(y1*payMtx[2, 0] + y2*payMtx[2, 1] + y3*payMtx[2, 2] + (1 - y1 - y2 - y3)*payMtx[2, 3]) + (1 - x1 - x2 - x3)*(y1*payMtx[3, 0] + y2*payMtx[3, 1] + y3*payMtx[3, 2] + (1 - y1 - y2 - y3)*payMtx[3, 3])
#    print("test", test)
    return x1*(y1*payMtx[0, 0] + y2*payMtx[0, 1] + y3*payMtx[0, 2] + (1 - y1 - y2 - y3)*payMtx[0, 3]) + x2*(y1*payMtx[1, 0] + y2*payMtx[1, 1] + y3*payMtx[1, 2] + (1 - y1 - y2 - y3)*payMtx[1, 3]) + x3*(y1*payMtx[2, 0] + y2*payMtx[2, 1] + y3*payMtx[2, 2] + (1 - y1 - y2 - y3)*payMtx[2, 3]) + (1 - x1 - x2 - x3)*(y1*payMtx[3, 0] + y2*payMtx[3, 1] + y3*payMtx[3, 2] + (1 - y1 - y2 - y3)*payMtx[3, 3])

def repDyn4(X, t, payMtx):
    '''Computes the replicator dynamics for a symmetric 2P4S game'''
    x1, x2, x3 = X
    Pbar = w4(x1, x2, x3, x1, x2, x3, payMtx)
    return np.array([x1*(w4(1, 0, 0, x1, x2, x3, payMtx) - Pbar), x2*(w4(0, 1, 0, x1, x2, x3, payMtx) - Pbar), x3*(w4(0, 0, 1, x1, x2, x3, payMtx) - Pbar)])

def repDyn4Rev(X, t, payMtx):
    '''Computes the opposite replicator dynamics for a symmetric 2P4S game'''
    x1, x2, x3 = X
    Pbar = w4(x1, x2, x3, x1, x2, x3, payMtx)
    return np.array([-x1*(w4(1, 0, 0, x1, x2, x3, payMtx) - Pbar), -x2*(w4(0, 1, 0, x1, x2, x3, payMtx) - Pbar), -x3*(w4(0, 0, 1, x1, x2, x3, payMtx) - Pbar)])


# Replicator dynamics for a 3-population 2-action game
def _expected_payoff(pay_tensor, probs, player_index, action):
    """Compute expected payoff for a given player and action in a 3-population 2-action game."""
    total = 0.0
    for a0 in (0, 1):
        for a1 in (0, 1):
            for a2 in (0, 1):
                actions = [a0, a1, a2]
                if actions[player_index] != action:
                    continue
                prob = 1.0
                for idx, a_val in enumerate(actions):
                    if idx == player_index:
                        continue
                    p = probs[idx]
                    prob *= p if a_val == 1 else (1 - p)
                total += pay_tensor[a0, a1, a2] * prob
    return total


def repDyn3Pop2(X, t, payTensors):
    """Replicator dynamics for three populations with two actions each."""
    x, y, z = X
    probs = [x, y, z]
    tensors = payTensors

    payoffs = []
    for idx, tensor in enumerate(tensors):
        u0 = _expected_payoff(tensor, probs, idx, 0)
        u1 = _expected_payoff(tensor, probs, idx, 1)
        payoffs.append((u0, u1))

    dx = x * (payoffs[0][1] - (x * payoffs[0][1] + (1 - x) * payoffs[0][0]))
    dy = y * (payoffs[1][1] - (y * payoffs[1][1] + (1 - y) * payoffs[1][0]))
    dz = z * (payoffs[2][1] - (z * payoffs[2][1] + (1 - z) * payoffs[2][0]))
    return np.array([dx, dy, dz])


def repDyn3Pop2Rev(X, t, payTensors):
    """Opposite replicator dynamics for the three-population 2-action game."""
    return -repDyn3Pop2(X, t, payTensors)


def compute_equilibria(payoffs):
    """Return rest points (equilibria) of the replicator dynamics for the given payoff tensors."""
    from sympy import Symbol
    from sympy.solvers import solve
    from itertools import product

    time_0 = 0
    x_sym = Symbol('x')
    y_sym = Symbol('y')
    equilibria = []
    degenerate = False
    message = None

    def _extract_isolated_solutions(solutions, symbols):
        isolated = []
        found_degeneracy = False
        for sol in solutions:
            if not all(sym in sol for sym in symbols):
                found_degeneracy = True
                continue
            try:
                isolated.append([float(sol[sym]) for sym in symbols])
            except (TypeError, ValueError):
                found_degeneracy = True
        return isolated, found_degeneracy

    def _mark_degenerate():
        nonlocal degenerate, message
        degenerate = True
        message = (
            "Degenerate equilibrium solutions detected: SymPy returned a "
            "parametric/non-isolated solution set. Non-isolated equilibrium "
            "sets are not plotted; isolated equilibria are still shown when "
            "they can be identified."
        )
        warnings.warn(message, DegenerateEquilibriumWarning, stacklevel=3)

    def _contains_point(point, acc):
        for sol in acc:
            if all(abs(sol[i] - point[i]) < 1e-9 for i in range(len(point))):
                return True
        return False

    def _add_missing_points(points):
        for point in points:
            if not _contains_point(point, equilibria):
                equilibria.append(list(point))

    if payoffs[0].shape == (3,):
        dx, dy = repDyn3([x_sym, y_sym], time_0, payoffs)
        mass_constraint = dx + dy
        solutions = solve([dx, dy, mass_constraint], x_sym, y_sym, dict=True)
        equilibria, degenerate = _extract_isolated_solutions(solutions, (x_sym, y_sym))
        if degenerate:
            _mark_degenerate()
        _add_missing_points([(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)])

    elif payoffs[0].shape == (2, 2):
        pay_p1, pay_p2 = payoffs
        dx = repDyn22([x_sym, y_sym], time_0, pay_p1)
        dy = repDyn22([y_sym, x_sym], time_0, pay_p2)
        solutions = solve([dx, dy], x_sym, y_sym, dict=True)
        equilibria, degenerate = _extract_isolated_solutions(solutions, (x_sym, y_sym))
        if degenerate:
            _mark_degenerate()
        _add_missing_points(product((0.0, 1.0), repeat=2))

    elif payoffs[0].shape == (2, 2, 2):
        z_sym = Symbol('z')
        dx, dy, dz = repDyn3Pop2([x_sym, y_sym, z_sym], time_0, payoffs)
        solutions = solve([dx, dy, dz], x_sym, y_sym, z_sym, dict=True)
        equilibria, degenerate = _extract_isolated_solutions(
            solutions, (x_sym, y_sym, z_sym)
        )
        if degenerate:
            _mark_degenerate()

        for vertex in product((0.0, 1.0), repeat=3):
            if not _contains_point(vertex, equilibria):
                equilibria.append(list(vertex))

    elif payoffs[0].shape == (4,):
        z_sym = Symbol('z')
        dx, dy, dz = repDyn4([x_sym, y_sym, z_sym], time_0, payoffs)
        mass_constraint = dx + dy + dz
        solutions = solve([dx, dy, dz, mass_constraint], x_sym, y_sym, z_sym, dict=True)
        equilibria, degenerate = _extract_isolated_solutions(
            solutions, (x_sym, y_sym, z_sym)
        )
        if degenerate:
            _mark_degenerate()
        _add_missing_points([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)])

    return EquilibriumResult(equilibria, degenerate=degenerate, message=message)
