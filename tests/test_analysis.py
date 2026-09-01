import numpy as np
import pytest

import analysis
import dynamics
import game
import examples


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
    g = examples.games.good_rps
    rows = analysis.analyze_equilibria(g).to_rows()

    assert rows
    assert list(rows[0].keys()) == EXPECTED_COLUMNS


def test_matching_pennies_interior_equilibrium_is_found():
    g = examples.games.matching_pennies
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
    g = examples.games.good_rps
    equilibrium = analysis.analyze_equilibria(g).equilibria[0]

    assert isinstance(equilibrium.nash, bool)
    assert isinstance(equilibrium.strict_nash, bool)
    assert isinstance(equilibrium.ess, bool)


def test_hawk_dove_retaliator_edge_ess_has_eigenpairs():
    result = analysis.analyze_equilibria(examples.games.hawk_dove_retaliator)
    equilibrium = next(
        eq
        for eq in result.equilibria
        if np.allclose(eq.full_position, [2 / 3, 1 / 3, 0])
    )

    assert equilibrium.ess is True
    assert equilibrium.stability == "sink"
    assert equilibrium.admissible_eigenvalues.size > 0
    assert equilibrium.admissible_eigenvectors.shape[1] > 0
