import numpy as np
import pytest

import game


def test_symmetric_3_strategy_game_class():
    g = game.Game(
        "RPS",
        np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float),
        strategy_labels=["R", "P", "S"],
    )

    assert g.symmetric is True
    assert g.game_class == "2P3S"
    assert g.num_strategies() == 3
    assert g.strategy_labels == ["R", "P", "S"]


def test_asymmetric_2_strategy_game_class():
    g = game.Game(
        "Matching Pennies",
        (
            np.array([[1, -1], [-1, 1]], dtype=float),
            np.array([[-1, 1], [1, -1]], dtype=float),
        ),
        symmetric=False,
    )

    assert g.symmetric is False
    assert g.game_class == "2P2S"
    assert g.num_strategies() == 2


def test_incompatible_symmetry_flag_raises():
    with pytest.raises(ValueError):
        game.Game(
            "Bad symmetry flag",
            (
                np.array([[1, 0], [0, 1]], dtype=float),
                np.array([[1, 0], [0, 1]], dtype=float),
            ),
            symmetric=True,
        )


def test_expected_payoffs_for_symmetric_game():
    g = game.Game(
        "Coordination",
        np.array([[1, 0], [0, 1]], dtype=float),
    )

    np.testing.assert_allclose(g.expected_payoffs([[0.25, 0.75]]), [0.25, 0.75])
