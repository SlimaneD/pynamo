# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:18:00 2020

@author: Benjamin Giraudon

Status : OK
"""

import warnings

import numpy as np

__all__ = [
    "DegenerateEquilibriumWarning",
    "EquilibriumResult",
    "replicator_2p2s",
    "reverse_replicator_2p2s",
    "replicator_2p3s",
    "reverse_replicator_2p3s",
    "replicator_2p4s",
    "reverse_replicator_2p4s",
    "replicator_3pop2s",
    "reverse_replicator_3pop2s",
    "compute_equilibria",
]


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
def _average_payoff_2p3s(x1, x2, y1, y2, payoff_data):
    """Average payoff of strategy (x1, x2) against strategy (y1, y2)."""
    return (
        x1 * (y1 * payoff_data[0][0] + y2 * payoff_data[0][1] + (1 - y1 - y2) * payoff_data[0][2])
        + x2 * (y1 * payoff_data[1][0] + y2 * payoff_data[1][1] + (1 - y1 - y2) * payoff_data[1][2])
        + (1 - x1 - x2)
        * (y1 * payoff_data[2][0] + y2 * payoff_data[2][1] + (1 - y1 - y2) * payoff_data[2][2])
    )


def replicator_2p3s(state, t, payoff_data):
    """Replicator dynamics for a symmetric 2-player 3-strategy game."""
    x1, x2 = state
    average_payoff = _average_payoff_2p3s(x1, x2, x1, x2, payoff_data)
    return np.array(
        [
            x1 * (_average_payoff_2p3s(1, 0, x1, x2, payoff_data) - average_payoff),
            x2 * (_average_payoff_2p3s(0, 1, x1, x2, payoff_data) - average_payoff),
        ]
    )


def reverse_replicator_2p3s(state, t, payoff_data):
    """Reverse-time replicator dynamics for a symmetric 2P3S game."""
    return -replicator_2p3s(state, t, payoff_data)


# Replicator dynamics for an asymmetric 2P2S game
def _replicator_2p2s_population(state, t, payoff_data):
    """One-population component of asymmetric 2P2S replicator dynamics."""
    x, y = state
    payoff_action_1 = y * payoff_data[0][0] + (1 - y) * payoff_data[0][1]
    payoff_action_2 = y * payoff_data[1][0] + (1 - y) * payoff_data[1][1]
    return x * (1 - x) * (payoff_action_1 - payoff_action_2)


def replicator_2p2s(state, t, payoff_data):
    """Vector field for asymmetric 2-player 2-strategy replicator dynamics."""
    x, y = state
    return [
        _replicator_2p2s_population([x, y], t, payoff_data[0]),
        _replicator_2p2s_population([y, x], t, payoff_data[1]),
    ]


def reverse_replicator_2p2s(state, t, payoff_data):
    """Reverse-time vector field for asymmetric 2P2S replicator dynamics."""
    return -np.asarray(replicator_2p2s(state, t, payoff_data))

# Replicator dynamics for 2P4S game

def _average_payoff_2p4s(x1, x2, x3, y1, y2, y3, payoff_data):
    """Average payoff of strategy (x1, x2, x3) against strategy (y1, y2, y3)."""
#    X = np.array([x1, x2, x3, 1 - x1 - x2 - x3])
#    Y = np.array([ [y1], [y2], [y3], [1 - y1 - y2 - y3] ])
#    PY = np.dot(payoff_data, Y)
#    sumT = np.dot(X, PY)[0]
#    print("sumT", sumT)
#    test = x1*(y1*payoff_data[0, 0] + y2*payoff_data[0, 1] + y3*payoff_data[0, 2] + (1 - y1 - y2 - y3)*payoff_data[0, 3]) + x2*(y1*payoff_data[1, 0] + y2*payoff_data[1, 1] + y3*payoff_data[1, 2] + (1 - y1 - y2 - y3)*payoff_data[1, 3]) + x3*(y1*payoff_data[2, 0] + y2*payoff_data[2, 1] + y3*payoff_data[2, 2] + (1 - y1 - y2 - y3)*payoff_data[2, 3]) + (1 - x1 - x2 - x3)*(y1*payoff_data[3, 0] + y2*payoff_data[3, 1] + y3*payoff_data[3, 2] + (1 - y1 - y2 - y3)*payoff_data[3, 3])
#    print("test", test)
    return x1*(y1*payoff_data[0, 0] + y2*payoff_data[0, 1] + y3*payoff_data[0, 2] + (1 - y1 - y2 - y3)*payoff_data[0, 3]) + x2*(y1*payoff_data[1, 0] + y2*payoff_data[1, 1] + y3*payoff_data[1, 2] + (1 - y1 - y2 - y3)*payoff_data[1, 3]) + x3*(y1*payoff_data[2, 0] + y2*payoff_data[2, 1] + y3*payoff_data[2, 2] + (1 - y1 - y2 - y3)*payoff_data[2, 3]) + (1 - x1 - x2 - x3)*(y1*payoff_data[3, 0] + y2*payoff_data[3, 1] + y3*payoff_data[3, 2] + (1 - y1 - y2 - y3)*payoff_data[3, 3])


def replicator_2p4s(state, t, payoff_data):
    """Replicator dynamics for a symmetric 2-player 4-strategy game."""
    x1, x2, x3 = state
    average_payoff = _average_payoff_2p4s(x1, x2, x3, x1, x2, x3, payoff_data)
    return np.array([x1*(_average_payoff_2p4s(1, 0, 0, x1, x2, x3, payoff_data) - average_payoff), x2*(_average_payoff_2p4s(0, 1, 0, x1, x2, x3, payoff_data) - average_payoff), x3*(_average_payoff_2p4s(0, 0, 1, x1, x2, x3, payoff_data) - average_payoff)])


def reverse_replicator_2p4s(state, t, payoff_data):
    """Reverse-time replicator dynamics for a symmetric 2P4S game."""
    return -replicator_2p4s(state, t, payoff_data)


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


def replicator_3pop2s(state, t, payoff_tensors):
    """Replicator dynamics for three populations with two actions each."""
    x, y, z = state
    probs = [x, y, z]
    tensors = payoff_tensors

    payoff_data = []
    for idx, tensor in enumerate(tensors):
        u0 = _expected_payoff(tensor, probs, idx, 0)
        u1 = _expected_payoff(tensor, probs, idx, 1)
        payoff_data.append((u0, u1))

    dx = x * (payoff_data[0][1] - (x * payoff_data[0][1] + (1 - x) * payoff_data[0][0]))
    dy = y * (payoff_data[1][1] - (y * payoff_data[1][1] + (1 - y) * payoff_data[1][0]))
    dz = z * (payoff_data[2][1] - (z * payoff_data[2][1] + (1 - z) * payoff_data[2][0]))
    return np.array([dx, dy, dz])


def reverse_replicator_3pop2s(state, t, payoff_tensors):
    """Opposite replicator dynamics for the three-population 2-action game."""
    return -replicator_3pop2s(state, t, payoff_tensors)


def compute_equilibria(payoff_data):
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

    if payoff_data[0].shape == (3,):
        dx, dy = replicator_2p3s([x_sym, y_sym], time_0, payoff_data)
        mass_constraint = dx + dy
        solutions = solve([dx, dy, mass_constraint], x_sym, y_sym, dict=True)
        equilibria, degenerate = _extract_isolated_solutions(solutions, (x_sym, y_sym))
        if degenerate:
            _mark_degenerate()
        _add_missing_points([(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)])

    elif payoff_data[0].shape == (2, 2):
        pay_p1, pay_p2 = payoff_data
        dx = _replicator_2p2s_population([x_sym, y_sym], time_0, pay_p1)
        dy = _replicator_2p2s_population([y_sym, x_sym], time_0, pay_p2)
        solutions = solve([dx, dy], x_sym, y_sym, dict=True)
        equilibria, degenerate = _extract_isolated_solutions(solutions, (x_sym, y_sym))
        if degenerate:
            _mark_degenerate()
        _add_missing_points(product((0.0, 1.0), repeat=2))

    elif payoff_data[0].shape == (2, 2, 2):
        z_sym = Symbol('z')
        dx, dy, dz = replicator_3pop2s([x_sym, y_sym, z_sym], time_0, payoff_data)
        solutions = solve([dx, dy, dz], x_sym, y_sym, z_sym, dict=True)
        equilibria, degenerate = _extract_isolated_solutions(
            solutions, (x_sym, y_sym, z_sym)
        )
        if degenerate:
            _mark_degenerate()

        for vertex in product((0.0, 1.0), repeat=3):
            if not _contains_point(vertex, equilibria):
                equilibria.append(list(vertex))

    elif payoff_data[0].shape == (4,):
        z_sym = Symbol('z')
        dx, dy, dz = replicator_2p4s([x_sym, y_sym, z_sym], time_0, payoff_data)
        mass_constraint = dx + dy + dz
        solutions = solve([dx, dy, dz, mass_constraint], x_sym, y_sym, z_sym, dict=True)
        equilibria, degenerate = _extract_isolated_solutions(
            solutions, (x_sym, y_sym, z_sym)
        )
        if degenerate:
            _mark_degenerate()
        _add_missing_points([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)])

    return EquilibriumResult(equilibria, degenerate=degenerate, message=message)
