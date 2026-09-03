"""Built-in example games shipped with pyNamo."""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Tuple

import numpy as np

from game import Game, format_game_description

__all__ = ["GameCatalog", "describe", "games"]


def describe(game: Game) -> None:
    """Print a readable summary of a game and its catalogue metadata.

    This is a convenience wrapper around `game.Game.describe`.

    Examples
    --------
    >>> import examples
    >>> examples.describe(examples.games.good_rps)
    """
    print(format_game_description(game))


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
            reference="Sandholm (2010), Examples 3.3.2 and 3.3.7.",
            reference_note="Good RPS is the case w > l; pyNamo uses w = 2 and l = 1.",
            parameters="win value w = 2, loss cost l = 1.",
            illustrates="Globally attracting interior equilibrium in good RPS under the replicator dynamic.",
        ),
        "hawk_dove_retaliator": Game(
            "Hawk-Dove-Retaliator",
            _hawk_dove_retaliator(),
            strategy_labels=["Hawk", "Dove", "Retaliator"],
            description="Hawk-Dove-Retaliator contest game with v/c = 2/3.",
            reference=(
                "Maynard Smith & Price (1973); Maynard Smith & Parker (1976); "
                "McElreath & Boyd (2007), Figure 2.3."
            ),
            reference_note="pyNamo uses the payoff convention presented by McElreath & Boyd.",
            parameters="resource value v = 2, fighting cost c = 3.",
            illustrates="Degenerate equilibrium edge and transverse stability in a biologically motivated contest game.",
        ),
        "standard_rps": Game(
            "Standard RPS",
            np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]]),
            strategy_labels=["R", "P", "S"],
            description="Standard zero-sum Rock-Paper-Scissors.",
            reference="Sandholm (2010), Example 3.3.2 and Example 6.1.1.",
            reference_note="Standard RPS is the case w = l.",
            parameters="win value w = 1, loss cost l = 1.",
            illustrates="Closed orbits/center-like behavior under the replicator dynamic.",
        ),
        "coordination_123": Game(
            "123 Coordination",
            np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            strategy_labels=["1", "2", "3"],
            description="Three-strategy coordination game with unequal diagonal payoffs.",
            reference="Sandholm (2010), 123 Coordination game; Examples 3.1.5 and 7.1.1.",
            reference_note="Exact payoff matrix used by Sandholm for 123 Coordination.",
            parameters="diagonal payoffs 1, 2, and 3.",
            illustrates="Multiple strict equilibria and potential-game structure.",
        ),
        "pure_coordination": Game(
            "Pure Coordination",
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            strategy_labels=["1", "2", "3"],
            description="Pure three-strategy coordination game.",
            reference="Sandholm (2010), Figure 9.7.",
            reference_note="Identity payoff matrix for pure coordination.",
            parameters="unit diagonal payoffs and zero off-diagonal payoffs.",
            illustrates="Multiple equivalent strict equilibria and an unstable interior equilibrium.",
        ),
        "repeated_pd_tft_allc_alld": Game(
            "Repeated PD: TFT, ALLC, ALLD",
            _repeated_pd_tft_allc_alld(),
            strategy_labels=["TFT", "ALLC", "ALLD"],
            description="Repeated Prisoner's Dilemma with TFT, ALLC, and ALLD.",
            reference="McElreath & Boyd (2007), Figure 4.1.",
            reference_note="Payoffs are generated from the repeated-game expressions used in the text.",
            parameters="benefit b = 4, cost c = 1, continuation probability w = 0.5.",
            illustrates="Repeated-interaction cooperation dynamics and degenerate TFT-ALLC structure.",
        ),
        "matching_pennies": Game(
            "Matching Pennies",
            (
                np.array([[1, -1], [-1, 1]]),
                np.array([[-1, 1], [1, -1]]),
            ),
            strategy_labels=["H", "T"],
            description="Canonical zero-sum two-population Matching Pennies game.",
            reference="Sandholm (2010), Example 7.2.3.",
            reference_note="Standard two-player zero-sum payoff convention.",
            parameters="payoffs +1 for matching to player 1 and -1 to player 2; signs reversed on mismatches.",
            illustrates="Interior center and cycling behavior in asymmetric two-population dynamics.",
            symmetric=False,
        ),
        "two_player_hawk_dove": Game(
            "2-player Hawk-Dove",
            (
                np.array([[-1, 5], [0, 2.5]]),
                np.array([[-1, 5], [0, 2.5]]),
            ),
            strategy_labels=["H", "D"],
            description="Two-population Hawk-Dove contest game.",
            reference="Maynard Smith & Price (1973); Sandholm (2010), Chapter 2.",
            reference_note="Classical Hawk-Dove payoff convention in two-population form.",
            parameters="resource value v = 5, fighting cost c = 7 in the equivalent Hawk-Dove normalization.",
            illustrates="Stable mixed equilibrium in a two-strategy biological contest game.",
            symmetric=False,
        ),
        "battle_of_the_sexes": Game(
            "Battle of the Sexes",
            (
                np.array([[2, 0], [0, 1]]),
                np.array([[1, 0], [0, 2]]),
            ),
            strategy_labels=["B", "S"],
            description="Canonical asymmetric Battle of the Sexes coordination/conflict game.",
            reference="Standard game-theory example.",
            reference_note="Included as a familiar asymmetric 2-player / 2-strategy benchmark.",
            parameters="coordination payoffs favor different coordinated outcomes for the two players.",
            illustrates="Two pure Nash equilibria and an unstable mixed equilibrium in asymmetric dynamics.",
            symmetric=False,
        ),
        "stag_hunt": Game(
            "Stag Hunt",
            (
                np.array([[4.0, 0.0], [3.0, 3.0]]),
                np.array([[4.0, 0.0], [3.0, 3.0]]),
            ),
            strategy_labels=["Stag", "Hare"],
            player_strategy_labels=[
                ["Stag", "Hare"],
                ["Stag", "Hare"],
            ],
            player_labels=["Player 1", "Player 2"],
            description=(
                "Standard two-population Stag Hunt with payoff-dominant and "
                "risk-dominant conventions."
            ),
            reference="Skyrms (2004), The Stag Hunt and the Evolution of Social Structure.",
            reference_note="Canonical two-strategy coordination game implemented in two-population form.",
            parameters="payoff-dominant Stag/Stag outcome and safer Hare option.",
            illustrates="Bistability, risk dominance, and basin dependence in coordination dynamics.",
            symmetric=False,
        ),
        "rps_with_twin": Game(
            "RPS with a Twin",
            np.array(
                [
                    [0, -1, 1, 1],
                    [1, 0, -1, -1],
                    [-1, 1, 0, 0],
                    [-1, 1, 0, 0],
                ]
            ),
            strategy_labels=["$R$", "$P$", "$S$", "$T$"],
            description=(
                "Sandholm's RPS-with-a-twin example: standard Rock-Paper-Scissors "
                "with an exact twin of Scissors, illustrating invariant planes under "
                "the replicator dynamic."
            ),
            reference="Sandholm (2010), Section 9.4.3 and Figure 9.19a.",
            reference_note="Standard RPS with an exact duplicate of Scissors.",
            parameters="standard RPS payoffs with strategy T an exact twin of S.",
            illustrates="Invariant planes and neutral directions created by duplicate strategies.",
        ),
        "chaotic_four_strategy_game": Game(
            "Chaotic Four-Strategy Game",
            np.array(
                [
                    [0, -12, 0, 22],
                    [20, 0, 0, -10],
                    [-21, -4, 0, 35],
                    [10, -2, 2, 0],
                ]
            ),
            strategy_labels=["1", "2", "3", "4"],
            description=(
                "Four-strategy game exhibiting complex/chaotic-looking "
                "replicator dynamics; provenance verified in Sandholm's Example 9.3.1."
            ),
            reference="Sandholm (2010), Example 9.3.1.",
            reference_note=(
                "Sandholm attributes the example to Arneodo, Coullet, and Tresser, "
                "with related discussion by Schnabl et al. and Skyrms (1992)."
            ),
            parameters="exact 4x4 payoff matrix from Sandholm's Example 9.3.1.",
            illustrates="Chaotic attractor under the replicator dynamic in a four-strategy game.",
        ),
        "ownership_game": Game(
            "Ownership Game",
            _ownership_game(),
            strategy_labels=["Hawk", "Dove", "Bourgeois", "Anti-Bourgeois"],
            description=(
                "Classical Hawk-Dove ownership game with Bourgeois and Anti-Bourgeois strategies."
            ),
            reference=(
                "Maynard Smith & Parker (1976); see also Mesterton-Gibbons & "
                "Sherratt (2014)."
            ),
            reference_note=(
                "Standard derived one-shot ownership game; Mesterton-Gibbons & "
                "Sherratt provide a modern treatment of Bourgeois and Anti-Bourgeois conventions."
            ),
            parameters="resource value v = 2, fighting cost c = 3.",
            illustrates="Ownership conventions, Bourgeois/Anti-Bourgeois equilibria, and asymmetric contest logic.",
        ),
        "coordination_cube": Game(
            "Coordination Cube",
            tuple(_coordination_tensor() for _ in range(3)),
            strategy_labels=["$x_A$", "$x_B$", "$x_C$"],
            description="Three-player coordination where everyone prefers matching actions.",
            reference="Standard coordination-game construction.",
            reference_note="Included as a simple symmetric game represented in the 3P2S class.",
            parameters="payoff 1 when all three players choose the same action; payoff 0 otherwise.",
            illustrates="Three-player cube geometry and coordination basins.",
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
            reference="Sandholm (2010), Example 9.2.3; Jordan (1993).",
            reference_note="Sandholm defines the three-player cyclic mismatching-pennies game.",
            parameters="payoff 1 for mismatching the next player in the cycle; payoff 0 otherwise.",
            illustrates="Nonconvergent three-player dynamics in the cube.",
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
            reference="pyNamo companion variant of Sandholm/Jordan cyclic mismatching pennies.",
            reference_note="This matching version is included as the sign/convention companion to cyclic mismatching pennies.",
            parameters="payoff 1 for matching the next player in the cycle; payoff 0 otherwise.",
            illustrates="Contrast between cyclic matching and cyclic mismatching incentives in 3P2S dynamics.",
            symmetric=False,
        ),
    }
)
