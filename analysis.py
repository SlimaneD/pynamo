"""Structured equilibrium analysis for pyNamo games.

This module separates mathematical analysis from plotting.  Plotting functions
can consume these records, while notebooks can display them as tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import warnings

import numpy as np
from sympy import Matrix
from sympy.abc import x, y, z

import dynamics
from game import infer_game_class

__all__ = [
    "InconclusiveStabilityWarning",
    "EquilibriumRecord",
    "EquilibriumAnalysis",
    "analyze_equilibria",
    "equilibrium_table",
]


class InconclusiveStabilityWarning(RuntimeWarning):
    """Warning raised when stability cannot be classified by implemented tests."""


warnings.simplefilter("always", InconclusiveStabilityWarning)


@dataclass
class EquilibriumRecord:
    """Analysis data for one isolated equilibrium."""

    reduced_position: np.ndarray
    full_position: np.ndarray
    stability: str
    eigenvalues: Optional[np.ndarray]
    eigenvectors: Optional[np.ndarray]
    admissible_eigenvalues: Optional[np.ndarray]
    admissible_eigenvectors: Optional[np.ndarray]
    nash: Optional[bool]
    strict_nash: Optional[bool]
    ess: Optional[bool] = None
    warning: Optional[str] = None


@dataclass
class EquilibriumAnalysis:
    """Collection of equilibrium records plus global analysis metadata."""

    records: List[EquilibriumRecord]
    game_class: str
    degenerate: bool = False
    message: Optional[str] = None

    def to_rows(self, ndigits: int = 6) -> List[dict]:
        """Return a list of dictionaries suitable for table display."""
        return [_record_to_row(record, ndigits) for record in self.records]

    def to_dataframe(self, ndigits: int = 6):
        """Return a pandas DataFrame if pandas is installed."""
        import pandas as pd

        return pd.DataFrame(self.to_rows(ndigits=ndigits))


def analyze_equilibria(game) -> EquilibriumAnalysis:
    """Compute equilibrium positions, stability, eigenvalues, and eigenvectors."""
    payoff_data = _payoff_data(game)
    game_class = infer_game_class(game)
    raw_equilibria = dynamics.compute_equilibria(payoff_data)
    records = []

    for eq in raw_equilibria:
        reduced = np.asarray(eq, dtype=float)
        if not _is_state_in_domain(reduced, game_class):
            continue

        nash, strict_nash = _nash_status(reduced, payoff_data, game_class)
        ess = _is_ess_symmetric(reduced, payoff_data, game_class)
        eig_data = _eigen_data(reduced, payoff_data, game_class)
        if eig_data is None:
            stability = "undetermined"
            eigvals = eigvecs = admissible_vals = admissible_vecs = None
            warning = "No eigenvalue analysis is implemented for this game class."
        else:
            eigvals, eigvecs = eig_data
            admissible_vals, admissible_vecs, skipped_complex = _admissible_eigenpairs(
                reduced, eig_data, game_class
            )
            stability = _classify_from_admissible_eigenvalues(
                admissible_vals, skipped_complex
            )
            warning = None

            if stability in ("center", "undetermined"):
                if ess:
                    stability = "sink"
                else:
                    warning = _unresolved_warning(reduced, game_class)
                    warnings.warn(
                        warning, InconclusiveStabilityWarning, stacklevel=2
                    )

        records.append(
            EquilibriumRecord(
                reduced_position=reduced,
                full_position=_full_position(reduced, game_class),
                stability=stability,
                eigenvalues=eigvals,
                eigenvectors=eigvecs,
                admissible_eigenvalues=admissible_vals,
                admissible_eigenvectors=admissible_vecs,
                nash=nash,
                strict_nash=strict_nash,
                ess=ess,
                warning=warning,
            )
        )

    return EquilibriumAnalysis(
        records=records,
        game_class=game_class,
        degenerate=getattr(raw_equilibria, "degenerate", False),
        message=getattr(raw_equilibria, "message", None),
    )


def equilibrium_table(game, ndigits: int = 6):
    """Convenience wrapper returning a pandas table of equilibrium analysis."""
    return analyze_equilibria(game).to_dataframe(ndigits=ndigits)


def _payoff_data(game):
    return getattr(game, "payoff_data", game)


def _eigen_data(reduced, payoff_data, game_class):
    if game_class == "2P3S":
        field = Matrix(dynamics.replicator_2p3s([x, y], 0, payoff_data))
        jacobian = field.jacobian(Matrix([x, y]))
        matrix = np.array(jacobian.subs([(x, reduced[0]), (y, reduced[1])]), dtype=float)
        return np.linalg.eig(matrix)

    if game_class == "2P2S":
        field = Matrix(dynamics.replicator_2p2s([x, y], 0, payoff_data))
        jacobian = field.jacobian(Matrix([x, y]))
        matrix = np.array(jacobian.subs([(x, reduced[0]), (y, reduced[1])]), dtype=float)
        return np.linalg.eig(matrix)

    if game_class == "2P4S":
        field = Matrix(dynamics.replicator_2p4s([x, y, z], 0, payoff_data))
        jacobian = field.jacobian(Matrix([x, y, z]))
        matrix = np.array(
            jacobian.subs([(x, reduced[0]), (y, reduced[1]), (z, reduced[2])]),
            dtype=float,
        )
        return np.linalg.eig(matrix)

    if game_class == "3P2S":
        matrix = _numeric_jacobian(
            lambda state: dynamics.replicator_3p2s(state, 0, payoff_data), reduced
        )
        return np.linalg.eig(matrix)

    return None


def _numeric_jacobian(field_function, point, eps: float = 1e-6):
    point = np.asarray(point, dtype=float)
    f0 = np.asarray(field_function(point), dtype=float)
    jacobian = np.zeros((point.size, point.size), dtype=float)
    for axis in range(point.size):
        if point[axis] <= eps:
            perturbed = point.copy()
            perturbed[axis] += eps
            f1 = np.asarray(field_function(perturbed), dtype=float)
            jacobian[:, axis] = (f1 - f0) / eps
        elif point[axis] >= 1.0 - eps:
            perturbed = point.copy()
            perturbed[axis] -= eps
            f1 = np.asarray(field_function(perturbed), dtype=float)
            jacobian[:, axis] = (f0 - f1) / eps
        else:
            forward = point.copy()
            backward = point.copy()
            forward[axis] += eps
            backward[axis] -= eps
            f_forward = np.asarray(field_function(forward), dtype=float)
            f_backward = np.asarray(field_function(backward), dtype=float)
            jacobian[:, axis] = (f_forward - f_backward) / (2.0 * eps)
    return jacobian


def _full_position(reduced, game_class):
    reduced = np.asarray(reduced, dtype=float)
    if game_class == "2P3S":
        return np.array([reduced[0], reduced[1], 1.0 - reduced.sum()])
    if game_class == "2P4S":
        return np.array([reduced[0], reduced[1], reduced[2], 1.0 - reduced.sum()])
    return reduced.copy()


def _nash_status(reduced, payoff_data, game_class, tol: float = 1e-8):
    """Return whether a state in the game domain is Nash and strict Nash."""
    if game_class in ("2P3S", "2P4S"):
        return _symmetric_nash_status(_full_position(reduced, game_class), payoff_data, tol)
    if game_class == "2P2S":
        return _asymmetric_2p2s_nash_status(reduced, payoff_data, tol)
    if game_class == "3P2S":
        return _three_player_two_strategy_nash_status(reduced, payoff_data, tol)
    return None, None


def _symmetric_nash_status(strategy, payoff_matrix, tol: float):
    p = np.asarray(strategy, dtype=float)
    payoff_data = payoff_matrix @ p
    return _mixed_strategy_best_response_status([p], [payoff_data], tol)


def _asymmetric_2p2s_nash_status(reduced, payoff_matrices, tol: float):
    x_prob_action_0, y_prob_action_0 = np.asarray(reduced, dtype=float)
    p0 = np.array([x_prob_action_0, 1.0 - x_prob_action_0])
    p1 = np.array([y_prob_action_0, 1.0 - y_prob_action_0])

    payoff_player_0 = payoff_matrices[0] @ p1
    payoff_player_1 = payoff_matrices[1] @ p0
    return _mixed_strategy_best_response_status(
        [p0, p1], [payoff_player_0, payoff_player_1], tol
    )


def _three_player_two_strategy_nash_status(reduced, payoff_tensors, tol: float):
    probabilities_action_1 = np.asarray(reduced, dtype=float)
    mixed_strategies = [
        np.array([1.0 - prob, prob]) for prob in probabilities_action_1
    ]

    payoff_vectors = []
    for player_index, tensor in enumerate(payoff_tensors):
        payoff_vectors.append(
            np.array(
                [
                    _expected_payoff_for_action(
                        tensor, probabilities_action_1, player_index, 0
                    ),
                    _expected_payoff_for_action(
                        tensor, probabilities_action_1, player_index, 1
                    ),
                ]
            )
        )

    return _mixed_strategy_best_response_status(mixed_strategies, payoff_vectors, tol)


def _expected_payoff_for_action(payoff_tensor, probabilities_action_1, player_index, action):
    total = 0.0
    for a0 in (0, 1):
        for a1 in (0, 1):
            for a2 in (0, 1):
                actions = [a0, a1, a2]
                if actions[player_index] != action:
                    continue

                probability = 1.0
                for idx, action_idx in enumerate(actions):
                    if idx == player_index:
                        continue
                    prob_action_1 = probabilities_action_1[idx]
                    probability *= prob_action_1 if action_idx == 1 else 1.0 - prob_action_1

                total += payoff_tensor[a0, a1, a2] * probability
    return total


def _mixed_strategy_best_response_status(strategies, payoff_vectors, tol: float):
    is_nash = True
    is_strict = True

    for strategy, payoff_data in zip(strategies, payoff_vectors):
        strategy = np.asarray(strategy, dtype=float)
        payoff_data = np.asarray(payoff_data, dtype=float)

        support = strategy > tol
        max_payoff = np.max(payoff_data)
        if np.any(np.abs(payoff_data[support] - max_payoff) > tol):
            is_nash = False

        if np.count_nonzero(support) != 1:
            is_strict = False
        else:
            chosen = int(np.flatnonzero(support)[0])
            alternatives = np.delete(payoff_data, chosen)
            if alternatives.size and not np.all(payoff_data[chosen] > alternatives + tol):
                is_strict = False

    return bool(is_nash), bool(is_nash and is_strict)


def _is_state_in_domain(reduced, game_class, tol: float = 1e-9) -> bool:
    reduced = np.asarray(reduced, dtype=float)
    if game_class in ("2P3S", "2P4S"):
        return np.all(reduced >= -tol) and reduced.sum() <= 1.0 + tol
    if game_class in ("2P2S", "3P2S"):
        return np.all(reduced >= -tol) and np.all(reduced <= 1.0 + tol)
    return False


def _is_interior_point(point, game_class, tol: float = 1e-9) -> bool:
    point = np.asarray(point, dtype=float)
    if game_class in ("2P3S", "2P4S"):
        return np.all(point > tol) and point.sum() < 1.0 - tol
    if game_class in ("2P2S", "3P2S"):
        return np.all(point > tol) and np.all(point < 1.0 - tol)
    return False


def _is_admissible_direction(point, direction, game_class, tol: float = 1e-9) -> bool:
    point = np.asarray(point, dtype=float)
    direction = np.asarray(direction, dtype=float)

    if game_class == "2P3S":
        if abs(point[0]) <= tol and direction[0] < -tol:
            return False
        if abs(point[1]) <= tol and direction[1] < -tol:
            return False
        if abs(1.0 - point.sum()) <= tol and direction.sum() > tol:
            return False
        return True

    if game_class in ("2P2S", "3P2S"):
        for coord, delta in zip(point, direction):
            if abs(coord) <= tol and delta < -tol:
                return False
            if abs(1.0 - coord) <= tol and delta > tol:
                return False
        return True

    if game_class == "2P4S":
        if any(abs(point[idx]) <= tol and direction[idx] < -tol for idx in range(3)):
            return False
        if abs(1.0 - point.sum()) <= tol and direction.sum() > tol:
            return False
        return True

    return False


def _admissible_eigenpairs(point, eig_data, game_class, tol: float = 1e-9):
    eigvals, eigvecs = eig_data
    if _is_interior_point(point, game_class, tol):
        return np.asarray(eigvals, dtype=complex), np.asarray(eigvecs), False

    values = []
    vectors = []
    skipped_complex_boundary = False
    for idx, eigval in enumerate(eigvals):
        eigvec = eigvecs[:, idx]
        if np.linalg.norm(np.imag(eigvec)) > tol:
            skipped_complex_boundary = True
            continue

        direction = np.real(eigvec)
        if (
            _is_admissible_direction(point, direction, game_class, tol)
            or _is_admissible_direction(point, -direction, game_class, tol)
        ):
            values.append(eigval)
            vectors.append(eigvec)

    if vectors:
        vectors = np.column_stack(vectors)
    else:
        vectors = np.empty((len(point), 0))
    return np.asarray(values, dtype=complex), vectors, skipped_complex_boundary


def _classify_from_admissible_eigenvalues(admissible, skipped_complex_boundary, tol: float = 1e-9):
    if admissible is None or admissible.size == 0:
        return "undetermined"

    real_parts = np.real(admissible)
    if np.all(real_parts < -tol):
        return "sink"
    if np.all(real_parts > tol):
        return "source"
    if np.any(real_parts > tol) and np.any(real_parts < -tol):
        return "saddle"
    if np.all(np.abs(real_parts) <= tol):
        if skipped_complex_boundary:
            return "undetermined"
        if np.any(np.abs(np.imag(admissible)) > tol):
            return "center"
        return "undetermined"

    if np.any(real_parts > tol):
        # Positive real parts prove instability, but zero real parts prevent a
        # sharper source/saddle classification by linearization alone.
        return "unstable"

    # Negative real parts together with zero real parts do not prove asymptotic
    # stability. A later ESS check may still classify symmetric-game cases.
    return "undetermined"


def _is_ess_symmetric(reduced, payoff_data, game_class, tol: float = 1e-8) -> Optional[bool]:
    if game_class not in ("2P3S", "2P4S"):
        return None

    p = np.clip(_full_position(reduced, game_class), 0.0, 1.0)
    total = p.sum()
    if total <= tol:
        return False
    p = p / total

    pure_payoff_data = payoff_data @ p
    payoff_at_p = float(p @ pure_payoff_data)
    if np.any(pure_payoff_data > payoff_at_p + tol):
        return False

    support = np.flatnonzero(p > tol)
    best_responses = np.flatnonzero(np.abs(pure_payoff_data - payoff_at_p) <= tol)
    if best_responses.size != support.size or set(best_responses) != set(support):
        return False
    if support.size == 1:
        return True

    payoff_submatrix = payoff_data[np.ix_(support, support)]
    symmetric_part = 0.5 * (payoff_submatrix + payoff_submatrix.T)
    basis = _tangent_basis(support.size)
    restricted = basis.T @ symmetric_part @ basis
    return bool(np.all(np.linalg.eigvalsh(restricted) < -tol))


def _tangent_basis(size):
    if size <= 1:
        return np.empty((size, 0))
    basis = np.zeros((size, size - 1))
    for idx in range(size - 1):
        basis[idx, idx] = 1.0
        basis[-1, idx] = -1.0
    return basis


def _unresolved_warning(reduced, game_class):
    full = _full_position(reduced, game_class)
    return (
        "Stability classification is inconclusive for equilibrium "
        f"{np.round(full, 10).tolist()}: admissible linearization has "
        "eigenvalues with zero real part or unsupported eigendirections, "
        "and the implemented fallback did not establish asymptotic stability."
    )


def _record_to_row(record: EquilibriumRecord, ndigits: int) -> dict:
    return {
        "Position": _format_array(record.full_position, ndigits),
        "Stability Status": record.stability,
        "Nash": record.nash,
        "ESS": record.ess,
        "Strict Nash": record.strict_nash,
        "Eigenvalues": _format_array(record.admissible_eigenvalues, ndigits),
        "Eigenvectors": _format_matrix(record.admissible_eigenvectors, ndigits),
        "Warning": record.warning,
    }


def _format_number(value, ndigits: int):
    if value is None:
        return None
    value = complex(value)
    real = round(float(value.real), ndigits)
    imag = round(float(value.imag), ndigits)
    if imag == 0:
        return real
    return complex(real, imag)


def _format_array(values, ndigits: int):
    if values is None:
        return None
    return [_format_number(value, ndigits) for value in np.asarray(values).ravel()]


def _format_matrix(values, ndigits: int):
    if values is None:
        return None
    matrix = np.asarray(values)
    return [
        [_format_number(matrix[row, col], ndigits) for col in range(matrix.shape[1])]
        for row in range(matrix.shape[0])
    ]
