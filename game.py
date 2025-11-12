"""Core abstractions for representing normal-form games."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


PayoffMatrix = np.ndarray
PayoffPair = Tuple[PayoffMatrix, PayoffMatrix]
RawPayoff = Union[
    PayoffMatrix,
    Sequence[Sequence[float]],
    Sequence[PayoffMatrix],
]


class Game:
    """Lightweight container for matrix games used throughout pyNamo."""

    def __init__(
        self,
        name: str,
        payoff_matrices: RawPayoff,
        *,
        strategy_labels: Optional[Sequence[str]] = None,
        description: str = "",
        symmetric: Optional[bool] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.strategy_labels: List[str] = list(strategy_labels or [])
        self._payoff = self._normalise_payoff(payoff_matrices)

        inferred_symmetry = not isinstance(self._payoff, tuple)
        if symmetric is None:
            self.symmetric = inferred_symmetry
        else:
            if symmetric != inferred_symmetry:
                raise ValueError(
                    f"Incompatible symmetry flag for game '{name}'. "
                    f"Expected {'symmetric' if inferred_symmetry else 'asymmetric'} matrices."
                )
            self.symmetric = symmetric

        self._validate_dimensions()
        self._ensure_label_defaults()

    @property
    def payoff_data(self) -> Union[PayoffMatrix, PayoffPair]:
        """Return the underlying payoff representation."""
        return self._payoff

    def payoff_for_player(self, player: int = 0) -> PayoffMatrix:
        """Return the payoff matrix relevant for the requested player."""
        if self.symmetric:
            return self._payoff  # type: ignore[return-value]
        try:
            return self._payoff[player]  # type: ignore[index]
        except IndexError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Player index {player} out of range for game '{self.name}'."
            ) from exc

    def num_strategies(self) -> int:
        """Number of strategies available to each player."""
        if self.symmetric:
            return self._payoff.shape[0]  # type: ignore[return-value]
        return self._payoff[0].shape[0]  # type: ignore[index]

    def expected_payoffs(self, mixed_strategies: Sequence[Sequence[float]]) -> np.ndarray:
        """Expected payoff to each pure strategy under the supplied mixed strategies."""
        strategy_count = self.num_strategies()

        if self.symmetric:
            if len(mixed_strategies) != 1:
                raise ValueError(
                    "Symmetric games expect a single population mixed strategy."
                )
            sigma = self._as_probability_vector(mixed_strategies[0], strategy_count)
            return self._payoff @ sigma  # type: ignore[return-value]

        players = len(self._payoff)  # type: ignore[arg-type]
        if len(mixed_strategies) != players:
            raise ValueError(
                f"Asymmetric games expect {players} mixed strategies (one per population)."
            )

        vectors = [
            self._as_probability_vector(sigma, strategy_count) for sigma in mixed_strategies
        ]

        payoffs = np.empty((players, strategy_count), dtype=float)
        for idx, matrix in enumerate(self._payoff):  # type: ignore[iterable]
            result = matrix
            for axis in reversed(range(players)):
                if axis == idx:
                    continue
                result = np.tensordot(result, vectors[axis], axes=([axis], [0]))
            payoffs[idx] = result
        return payoffs

    def _normalise_payoff(
        self, payoff_matrices: RawPayoff
    ) -> Union[PayoffMatrix, PayoffPair]:
        if isinstance(payoff_matrices, tuple):
            matrices = tuple(self._to_numpy(matrix) for matrix in payoff_matrices)
            return matrices  # type: ignore[return-value]
        if isinstance(payoff_matrices, list) and payoff_matrices and isinstance(
            payoff_matrices[0], np.ndarray
        ):
            matrices_tuple = tuple(self._to_numpy(matrix) for matrix in payoff_matrices)  # type: ignore[arg-type]
            if len(matrices_tuple) == 1:
                return matrices_tuple[0]
            return matrices_tuple  # type: ignore[return-value]
        return self._to_numpy(payoff_matrices)  # type: ignore[arg-type]

    def _to_numpy(self, matrix: Union[PayoffMatrix, Sequence[Sequence[float]]]) -> PayoffMatrix:
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim == 2:
            rows, cols = arr.shape
            if rows != cols:
                raise ValueError("Payoff matrices must be square.")
        elif arr.ndim < 2:
            raise ValueError("Payoff arrays must have at least two dimensions.")
        return arr

    def _validate_dimensions(self) -> None:
        if self.symmetric:
            if self._payoff.shape[0] < 2:  # type: ignore[index]
                raise ValueError("Symmetric games must have at least two strategies.")
        else:
            shapes = {matrix.shape for matrix in self._payoff}  # type: ignore[iterable]
            if len(shapes) != 1:
                raise ValueError("All payoff tensors must share identical dimensions.")

    def _ensure_label_defaults(self) -> None:
        if self.strategy_labels:
            return

        count = self.num_strategies()
        default_labels = [f"S{i+1}" for i in range(count)]
        self.strategy_labels = default_labels

    def _as_probability_vector(
        self, strategy: Sequence[float], expected_size: int
    ) -> np.ndarray:
        vector = np.asarray(strategy, dtype=float)
        if vector.ndim != 1:
            raise ValueError("Mixed strategies must be one-dimensional vectors.")
        if vector.size != expected_size:
            raise ValueError(
                f"Mixed strategy length {vector.size} incompatible with {expected_size} strategies."
            )
        if np.any(vector < -1e-12):
            raise ValueError("Mixed strategies cannot have negative probabilities.")

        total = vector.sum()
        if not np.isclose(total, 1.0, atol=1e-8):
            raise ValueError("Mixed strategies must sum to 1.")
        return vector


def game_names(games: Iterable[Tuple[int, Game]]) -> dict:
    """Return a simple id->name mapping derived from a catalogue of games."""
    return {idx: game.name for idx, game in games}
