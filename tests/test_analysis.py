import numpy as np
import pytest

import analysis
import dynamics
import game
import parameters as param


EXPECTED_COLUMNS = [
    "Position",
    "Stability Status",
    "Nash",
    "ESS",
    "Strict Nash",
    "Eigenvalues",
    "Eigenvectors",
    "Warning",
]


def test_analysis_rows_have_expected_columns():
    g = param.available_games("2P3S")[1]
    rows = analysis.analyze_equilibria(g).to_rows()

    assert rows
    assert list(rows[0].keys()) == EXPECTED_COLUMNS


def test_matching_pennies_interior_equilibrium_is_found():
    g = param.available_games("2P2S")[1]
    rows = analysis.analyze_equilibria(g).to_rows()
    positions = np.asarray([row["Position"] for row in rows], dtype=float)

    assert np.any(np.all(np.isclose(positions, [0.5, 0.5]), axis=1))


def test_degenerate_equilibrium_warning_is_emitted():
    degenerate_edge_game = game.Game(
        "Degenerate Edge",
        np.array(
            [
                [1, 1, 0],
                [1, 1, 0],
                [2, 2, 0],
            ],
            dtype=float,
        ),
        strategy_labels=["A", "B", "C"],
    )

    with pytest.warns(dynamics.DegenerateEquilibriumWarning):
        result = analysis.analyze_equilibria(degenerate_edge_game)

    assert result.degenerate is True
    assert result.message is not None


def test_static_equilibrium_concepts_are_present():
    g = param.available_games("2P3S")[1]
    record = analysis.analyze_equilibria(g).records[0]

    assert isinstance(record.nash, bool)
    assert isinstance(record.strict_nash, bool)
    assert isinstance(record.ess, bool)
