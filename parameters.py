# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:52:00 2020

@author: Benjamin Giraudon

Status : OK
"""
import numpy as np

from game import Game, game_names

__all__ = [
    "GAME_CLASS_MENU",
    "GAME_CATALOG",
    "GAME_NAMES_2P3S",
    "GAME_NAMES_2P2S",
    "GAME_NAMES_2P4S",
    "GAME_NAMES_3P2S",
    "arrow_size",
    "arrow_width",
    "time_step",
    "available_games",
]


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


def _inclusive_fitness_pd(b=4.0, c=1.0, r=0.5):
    """Prisoner's Dilemma with relatedness included in inclusive-fitness payoffs."""
    return np.array(
        [
            [(1.0 + r) * (b - c), -c + r * b],
            [b - r * c, 0.0],
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


# Example game catalogue, indexed by supported game-class identifier.
GAME_CLASS_MENU = {1: "arrow", 2: "2P3S", 3: "2P2S", 4: "2P4S", 5: "3P2S"}

GAME_CATALOG = {
    "2P3S": {
        1: Game(
            "Good RPS",
            np.array([[0, -1, 2], [2, 0, -1], [-1, 2, 0]]),
            strategy_labels=["$R$", "$P$", "$S$"],
            description="Rock–Paper–Scissors game with interior equilibrium.",
        ),
        2: Game(
            "Hawk-Dove-Retaliator",
            _hawk_dove_retaliator(),
            strategy_labels=["Hawk", "Dove", "Retaliator"],
            description="McElreath & Boyd Figure 2.3, with v/c = 2/3.",
        ),
        3: Game(
            "Standard RPS",
            np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]]),
            strategy_labels=["R", "P", "S"],
        ),
        4: Game(
            "123 Coordination",
            np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            strategy_labels=["1", "2", "3"],
        ),
        5: Game(
            "Pure Coordination",
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            strategy_labels=["1", "2", "3"],
        ),
        6: Game(
            "Repeated PD: TFT, ALLC, ALLD",
            _repeated_pd_tft_allc_alld(),
            strategy_labels=["TFT", "ALLC", "ALLD"],
            description="McElreath & Boyd Figure 4.1, with b/c = 4 and w = 0.5.",
        ),
    },
    "2P2S": {
        1: Game(
            "Matching Pennies",
            (
                np.array([[1, -1], [-1, 1]]),
                np.array([[-1, 1], [1, -1]]),
            ),
            strategy_labels=["H", "T"],
            symmetric=False,
        ),
        2: Game(
            "2-player Hawk-Dove",
            (
                np.array([[-1, 5], [0, 2.5]]),
                np.array([[-1, 5], [0, 2.5]]),
            ),
            strategy_labels=["H", "D"],
            symmetric=False,
        ),
        3: Game(
            "Battle of the Sexes",
            (
                np.array([[2, 0], [0, 1]]),
                np.array([[1, 0], [0, 2]]),
            ),
            strategy_labels=["B", "S"],
            symmetric=False,
        ),
        4: Game(
            "PD with Relatedness",
            (
                _inclusive_fitness_pd(),
                _inclusive_fitness_pd(),
            ),
            strategy_labels=["C", "D"],
            description=(
                "Prisoner's Dilemma with relatedness folded into inclusive-fitness "
                "payoffs; b = 4, c = 1, r = 0.5, so r > c/b."
            ),
            symmetric=False,
        ),
    },
    "2P4S": {
        1: Game(
            "Hofbauer-Swinkels",
            np.array([[0, 0, -1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, -1, 0, 0]]),
            strategy_labels=["$R$", "$P$", "$S$", "$T$"],
        ),
        2: Game(
            "Skyrms 1992",
            np.array([[0, -12, 0, 22], [20, 0, 0, -10], [-21, -4, 0, 35], [10, -2, 2, 0]]),
            strategy_labels=["1", "2", "3", "4"],
        ),
        3: Game(
            "Ownership Game",
            _ownership_game(),
            strategy_labels=["Hawk", "Dove", "Bourgeois", "Anti-Bourgeois"],
            description=(
                "Maynard Smith-Parker ownership asymmetry game with v = 2 and c = 3."
            ),
        ),
    },
    "3P2S": {
        1: Game(
            "Coordination Cube",
            tuple(_coordination_tensor() for _ in range(3)),
            strategy_labels=["$x_A$", "$x_B$", "$x_C$"],
            description="Three-player coordination where everyone prefers matching actions.",
            symmetric=False,
        ),
        2: Game(
            "Cyclic Mismatching Pennies",
            _cyclic_mismatching_pennies_tensors(),
            strategy_labels=["$x = P_1(T)$", "$y = P_2(T)$", "$z = P_3(T)$"],
            description=(
                "Three-player cyclic mismatch game: player 1 mismatches player 2, "
                "player 2 mismatches player 3, and player 3 mismatches player 1."
            ),
            symmetric=False,
        ),
        3: Game(
            "Cyclic Matching Pennies",
            _cyclic_matching_pennies_tensors(),
            strategy_labels=["$x = P_1(T)$", "$y = P_2(T)$", "$z = P_3(T)$"],
            description=(
                "Three-player cyclic matching game: player 1 matches player 2, "
                "player 2 matches player 3, and player 3 matches player 1."
            ),
            symmetric=False,
        ),
    },
}

GAME_NAMES_2P3S = game_names(GAME_CATALOG["2P3S"].items())
GAME_NAMES_2P2S = game_names(GAME_CATALOG["2P2S"].items())
GAME_NAMES_2P4S = game_names(GAME_CATALOG["2P4S"].items())
GAME_NAMES_3P2S = game_names(GAME_CATALOG["3P2S"].items())

# Drawer parameters
arrow_size = 1 / 25.0
arrow_width = (1 / 2) * arrow_size
time_step = 0.01


def available_games(game_class):
    """Return the mapping of example IDs to Game instances for the requested game class."""
    return GAME_CATALOG.get(game_class, {})
