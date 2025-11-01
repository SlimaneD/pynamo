# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:52:00 2020

@author: Benjamin Giraudon

Status : OK
"""
import numpy as np

from game import Game, game_names

# Simulation dictionaries
dict_test = {1: "arrow", 2: "2P3S", 3: "2P2S", 4: "2P4S"}

GAMES_BY_TEST = {
    "2P3S": {
        1: Game(
            "Good RPS",
            np.array([[0, -1, 2], [2, 0, -1], [-1, 2, 0]]),
            strategy_labels=["$R$", "$P$", "$S$"],
            description="Rock–Paper–Scissors game with interior equilibrium.",
        ),
        2: Game(
            "Zeeman",
            np.array([[0, 6, -4], [-3, 0, 5], [-1, 3, 0]]),
            strategy_labels=["1", "2", "3"],
            description="Zeeman's example illustrating complex dynamics.",
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
            "Webb 9.9",
            np.array([[3, 0, 1], [0, 3, 1], [1, 1, 1]]),
            strategy_labels=["A", "B", "C"],
        ),
    },
    "2P2S": {
        1: Game(
            "Matching Pennies",
            (
                np.array([[1, -1], [-1, 1]]),
                np.array([[-1, 1], [1, -1]]),
            ),
            strategy_labels=["$p_1$", "$p_2$"],
            symmetric=False,
        ),
        2: Game(
            "2-pop Hawk-Dove",
            (
                np.array([[-1, 5], [0, 2.5]]),
                np.array([[-1, 5], [0, 2.5]]),
            ),
            strategy_labels=["$p_H$", "$p_D$"],
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
            "Bad RPS with a twin",
            np.array([[0, -2, 1, 1], [1, 0, -2, -2], [-2, 1, 0, 0], [-2, 1, 0, 0]]),
            strategy_labels=["$R$", "$P$", "$S$", "$T$"],
        ),
    },
}

dict_2P3S = game_names(GAMES_BY_TEST["2P3S"].items())
dict_2P2S = game_names(GAMES_BY_TEST["2P2S"].items())
dict_2P4S = game_names(GAMES_BY_TEST["2P4S"].items())

# Drawer parameters
arrowSize = 1 / 25.0
arrowWidth = (1 / 2) * arrowSize
step = 0.01


def available_games(test_key):
    """Return the mapping of example IDs to Game instances for the requested test."""
    return GAMES_BY_TEST.get(test_key, {})
