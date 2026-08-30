"""Built-in example games shipped with pyNamo."""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Tuple

import numpy as np

from game import Game

__all__ = ["GameCatalog", "games"]


class GameCatalog:
    """Named collection of built-in example games.

    Games can be accessed by attribute, by string lookup, or by game class.

    Examples
    --------
    >>> import examples
    >>> examples.games.good_rps
    >>> examples.games("good_rps")
    >>> examples.games.by_class("2P2S")
    """

    def __init__(self, games_by_name: Dict[str, Game]) -> None:
        """Create a game catalog.

        Parameters
        ----------
        games_by_name : dict of str to game.Game
            Mapping from stable example identifiers to Game objects.
        """
        self._games = dict(games_by_name)

    def __call__(self, name: str) -> Game:
        """Return a game by its string identifier.

        Parameters
        ----------
        name : str
            Identifier of a built-in example game.

        Returns
        -------
        game.Game
            The requested example game.
        """
        return self._games[name]

    def __getattr__(self, name: str) -> Game:
        try:
            return self._games[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __contains__(self, name: str) -> bool:
        return name in self._games

    def __iter__(self) -> Iterator[str]:
        return iter(self._games)

    def items(self) -> Iterable[Tuple[str, Game]]:
        """Return `(name, game)` pairs for all examples."""
        return self._games.items()

    def names(self) -> list[str]:
        """Return the list of available example names."""
        return list(self._games)

    def by_class(self, game_class: str) -> Dict[str, Game]:
        """Return all examples matching a supported game class.

        Parameters
        ----------
        game_class : str
            One of "2P2S", "2P3S", "2P4S", or "3P2S".

        Returns
        -------
        dict of str to game.Game
            Examples whose inferred game class matches `game_class`.
        """
        return {
            name: game
            for name, game in self._games.items()
            if game.game_class == game_class
        }


def _coordination_tensor():
    tensor = np.zeros((2, 2, 2))
    for a1 in range(2):
        for a2 in range(2):
            for a3 in range(2):
                if a1 == a2 == a3:
                    tensor[a1, a2, a3] = 1
    return tensor


def _cyclic_mismatching_pennies_tensors():
    payoff_player_1 = np.zeros((2, 2, 2))
    payoff_player_2 = np.zeros((2, 2, 2))
    payoff_player_3 = np.zeros((2, 2, 2))

    for action_1 in (0, 1):
        for action_2 in (0, 1):
            for action_3 in (0, 1):
                payoff_player_1[action_1, action_2, action_3] = (
                    1.0 if action_1 != action_2 else 0.0
                )
                payoff_player_2[action_1, action_2, action_3] = (
                    1.0 if action_2 != action_3 else 0.0
                )
                payoff_player_3[action_1, action_2, action_3] = (
                    1.0 if action_3 != action_1 else 0.0
                )

    return payoff_player_1, payoff_player_2, payoff_player_3


def _cyclic_matching_pennies_tensors():
    payoff_player_1 = np.zeros((2, 2, 2))
    payoff_player_2 = np.zeros((2, 2, 2))
    payoff_player_3 = np.zeros((2, 2, 2))

    for action_1 in (0, 1):
        for action_2 in (0, 1):
            for action_3 in (0, 1):
                payoff_player_1[action_1, action_2, action_3] = (
                    1.0 if action_1 == action_2 else 0.0
                )
                payoff_player_2[action_1, action_2, action_3] = (
                    1.0 if action_2 == action_3 else 0.0
                )
                payoff_player_3[action_1, action_2, action_3] = (
                    1.0 if action_3 == action_1 else 0.0
                )

    return payoff_player_1, payoff_player_2, payoff_player_3


def _hawk_dove_retaliator(v=2.0, c=3.0):
    """McElreath & Boyd Hawk-Dove-Retaliator game, Figure 2.3."""
    return np.array(
        [
            [(v - c) / 2.0, v, (v - c) / 2.0],
            [0.0, v / 2.0, v / 2.0],
            [(v - c) / 2.0, v / 2.0, v / 2.0],
        ],
        dtype=float,
    )


def _repeated_pd_tft_allc_alld(b=4.0, c=1.0, w=0.5):
    """McElreath & Boyd repeated PD example, Figure 4.1."""
    return np.array(
        [
            [(b - c) / (1.0 - w), (b - c) / (1.0 - w), -c],
            [(b - c) / (1.0 - w), (b - c) / (1.0 - w), -c / (1.0 - w)],
            [b, b / (1.0 - w), 0.0],
        ],
        dtype=float,
    )


def _predator_prey_prey_payoffs():
    """Prey payoffs for the predator-prey behavioral-conflict example."""
    return np.array(
        [[0.0, 3.0], [2.0, 1.0]],
        dtype=float,
    )


def _predator_prey_predator_payoffs():
    """Predator payoffs for the predator-prey behavioral-conflict example."""
    return np.array(
        [[4.0, 1.0], [3.0, 0.0]],
        dtype=float,
    )


def _ownership_game(v=2.0, c=3.0):
    """Maynard Smith-Parker ownership game with Hawk, Dove, Bourgeois, Anti-Bourgeois."""
    return np.array(
        [
            [(v - c) / 2.0, v, (3.0 * v - c) / 4.0, (3.0 * v - c) / 4.0],
            [0.0, v / 2.0, v / 4.0, v / 4.0],
            [(v - c) / 4.0, 3.0 * v / 4.0, v / 2.0, v / 2.0 - c / 4.0],
            [(v - c) / 4.0, 3.0 * v / 4.0, v / 2.0 - c / 4.0, v / 2.0],
        ],
        dtype=float,
    )


games = GameCatalog(
    {
        "good_rps": Game(
            "Good RPS",
            np.array([[0, -1, 2], [2, 0, -1], [-1, 2, 0]]),
            strategy_labels=["$R$", "$P$", "$S$"],
            description="Rock-Paper-Scissors game with interior equilibrium.",
        ),
        "hawk_dove_retaliator": Game(
            "Hawk-Dove-Retaliator",
            _hawk_dove_retaliator(),
            strategy_labels=["Hawk", "Dove", "Retaliator"],
            description="McElreath & Boyd Figure 2.3, with v/c = 2/3.",
        ),
        "standard_rps": Game(
            "Standard RPS",
            np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]]),
            strategy_labels=["R", "P", "S"],
        ),
        "coordination_123": Game(
            "123 Coordination",
            np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            strategy_labels=["1", "2", "3"],
        ),
        "pure_coordination": Game(
            "Pure Coordination",
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            strategy_labels=["1", "2", "3"],
        ),
        "repeated_pd_tft_allc_alld": Game(
            "Repeated PD: TFT, ALLC, ALLD",
            _repeated_pd_tft_allc_alld(),
            strategy_labels=["TFT", "ALLC", "ALLD"],
            description="McElreath & Boyd Figure 4.1, with b/c = 4 and w = 0.5.",
        ),
        "matching_pennies": Game(
            "Matching Pennies",
            (
                np.array([[1, -1], [-1, 1]]),
                np.array([[-1, 1], [1, -1]]),
            ),
            strategy_labels=["H", "T"],
            symmetric=False,
        ),
        "two_player_hawk_dove": Game(
            "2-player Hawk-Dove",
            (
                np.array([[-1, 5], [0, 2.5]]),
                np.array([[-1, 5], [0, 2.5]]),
            ),
            strategy_labels=["H", "D"],
            symmetric=False,
        ),
        "battle_of_the_sexes": Game(
            "Battle of the Sexes",
            (
                np.array([[2, 0], [0, 1]]),
                np.array([[1, 0], [0, 2]]),
            ),
            strategy_labels=["B", "S"],
            symmetric=False,
        ),
        "predator_prey_behavioral_conflict": Game(
            "Predator-Prey Behavioral Conflict",
            (
                _predator_prey_prey_payoffs(),
                _predator_prey_predator_payoffs(),
            ),
            strategy_labels=["Fight", "Flee"],
            player_strategy_labels=[
                ["Fight", "Flee"],
                ["Aggressive", "Cautious"],
            ],
            player_labels=["Prey", "Predator"],
            description=(
                "Two-population predator-prey trait-frequency model. Prey fight "
                "or flee; predators are aggressive or cautious."
            ),
            symmetric=False,
        ),
        "hofbauer_swinkels": Game(
            "Hofbauer-Swinkels",
            np.array(
                [
                    [0, 0, -1, 0],
                    [0, 0, 0, -1],
                    [-1, 0, 0, 0],
                    [0, -1, 0, 0],
                ]
            ),
            strategy_labels=["$R$", "$P$", "$S$", "$T$"],
        ),
        "skyrms_1992": Game(
            "Skyrms 1992",
            np.array(
                [
                    [0, -12, 0, 22],
                    [20, 0, 0, -10],
                    [-21, -4, 0, 35],
                    [10, -2, 2, 0],
                ]
            ),
            strategy_labels=["1", "2", "3", "4"],
        ),
        "ownership_game": Game(
            "Ownership Game",
            _ownership_game(),
            strategy_labels=["Hawk", "Dove", "Bourgeois", "Anti-Bourgeois"],
            description=(
                "Maynard Smith-Parker ownership asymmetry game with v = 2 and c = 3."
            ),
        ),
        "coordination_cube": Game(
            "Coordination Cube",
            tuple(_coordination_tensor() for _ in range(3)),
            strategy_labels=["$x_A$", "$x_B$", "$x_C$"],
            description="Three-player coordination where everyone prefers matching actions.",
            symmetric=False,
        ),
        "cyclic_mismatching_pennies": Game(
            "Cyclic Mismatching Pennies",
            _cyclic_mismatching_pennies_tensors(),
            strategy_labels=["$x = P_1(T)$", "$y = P_2(T)$", "$z = P_3(T)$"],
            description=(
                "Three-player cyclic mismatch game: player 1 mismatches player 2, "
                "player 2 mismatches player 3, and player 3 mismatches player 1."
            ),
            symmetric=False,
        ),
        "cyclic_matching_pennies": Game(
            "Cyclic Matching Pennies",
            _cyclic_matching_pennies_tensors(),
            strategy_labels=["$x = P_1(T)$", "$y = P_2(T)$", "$z = P_3(T)$"],
            description=(
                "Three-player cyclic matching game: player 1 matches player 2, "
                "player 2 matches player 3, and player 3 matches player 1."
            ),
            symmetric=False,
        ),
    }
)
