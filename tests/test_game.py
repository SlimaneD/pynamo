import numpy as np
import pytest

from pynamo_egt import game
def test_symmetric_3_strategy_game_class():
    g = game.Game(
        "RPS",
        np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float),
        strategy_labels=["R", "P", "S"],
    )

    assert g.symmetric is True
    assert g.game_class == "1Pop3S"
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
    assert g.game_class == "2Pop2S"
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


@pytest.mark.parametrize('size,expected', [(2, '1Pop2S'), (3, '1Pop3S'), (4, '1Pop4S')])
def test_population_class_for_symmetric_games(size, expected):
    assert game.Game('symmetric', np.eye(size)).game_class == expected


def test_three_population_class():
    assert game.Game('three populations', tuple(np.zeros((2, 2, 2)) for _ in range(3))).game_class == '3Pop2S'


@pytest.mark.parametrize('old,new', [
    ('2P2S', '2Pop2S'), ('2P3S', '1Pop3S'),
    ('2P4S', '1Pop4S'), ('3P2S', '3Pop2S'),
])
def test_legacy_catalogue_lookup(old, new):
    from pynamo_egt.examples import games
    current = games.by_class(new)
    assert current
    assert games.by_class(old) == current
    assert all(g.game_class == new for g in current.values())
