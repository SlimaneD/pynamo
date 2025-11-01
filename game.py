"""Core abstractions for representing normal-form games."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


PayoffMatrix = np.ndarray
PayoffPair = Tuple[PayoffMatrix, PayoffMatrix]
RawPayoff = Union[PayoffMatrix, Sequence[Sequence[float]], PayoffPair]


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
        if player not in (0, 1):
            raise ValueError(f"Asymmetric games only support players 0 and 1. Received: {player}")
        return self._payoff[player]  # type: ignore[index]

    def num_strategies(self) -> int:
        """Number of strategies available to each player."""
        if self.symmetric:
            return self._payoff.shape[0]  # type: ignore[return-value]
        return self._payoff[0].shape[0]  # type: ignore[index]

    def _normalise_payoff(
        self, payoff_matrices: RawPayoff
    ) -> Union[PayoffMatrix, PayoffPair]:
        if isinstance(payoff_matrices, tuple):
            matrices = tuple(self._to_numpy(matrix) for matrix in payoff_matrices)
            if len(matrices) != 2:
                raise ValueError("Asymmetric games must provide exactly two payoff matrices.")
            return matrices  # type: ignore[return-value]
        if isinstance(payoff_matrices, list) and payoff_matrices and isinstance(
            payoff_matrices[0], np.ndarray
        ):
            matrices_tuple = tuple(self._to_numpy(matrix) for matrix in payoff_matrices)  # type: ignore[arg-type]
            if len(matrices_tuple) == 1:
                return matrices_tuple[0]
            if len(matrices_tuple) == 2:
                return matrices_tuple  # type: ignore[return-value]
        return self._to_numpy(payoff_matrices)  # type: ignore[arg-type]

    def _to_numpy(self, matrix: Union[PayoffMatrix, Sequence[Sequence[float]]]) -> PayoffMatrix:
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Payoff matrices must be two-dimensional.")
        rows, cols = arr.shape
        if rows != cols:
            raise ValueError("Payoff matrices must be square.")
        return arr

    def _validate_dimensions(self) -> None:
        if self.symmetric:
            if self._payoff.shape[0] < 2:  # type: ignore[index]
                raise ValueError("Symmetric games must have at least two strategies.")
        else:
            left, right = self._payoff  # type: ignore[assignment]
            if left.shape != right.shape:
                raise ValueError("Asymmetric games require payoff matrices of identical dimensions.")

    def _ensure_label_defaults(self) -> None:
        if self.strategy_labels:
            return

        count = self.num_strategies()
        default_labels = [f"S{i+1}" for i in range(count)]
        self.strategy_labels = default_labels


def game_names(games: Iterable[Tuple[int, Game]]) -> dict:
    """Return a simple id->name mapping derived from a catalogue of games."""
    return {idx: game.name for idx, game in games}

