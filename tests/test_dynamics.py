import numpy as np

from pynamo_egt import dynamics
from pynamo_egt import examples
def test_replicator_2p2s_shape():
    payoff_data = examples.games.matching_pennies.payoff_data
    vector = dynamics.replicator_2p2s([0.4, 0.6], 0, payoff_data)

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (2,)
    assert np.all(np.isfinite(vector))


def test_replicator_2p2s_uses_each_players_own_strategies_as_rows():
    player_1_first_strategy_dominant = np.array([[1, 1], [0, 0]], dtype=float)
    player_2_second_strategy_dominant = np.array([[0, 0], [1, 1]], dtype=float)

    vector = dynamics.replicator_2p2s(
        [0.5, 0.5],
        0,
        (player_1_first_strategy_dominant, player_2_second_strategy_dominant),
    )

    np.testing.assert_allclose(vector, [0.25, -0.25])


def test_replicator_2p3s_shape():
    payoff_data = examples.games.good_rps.payoff_data
    vector = dynamics.replicator_2p3s([0.3, 0.4], 0, payoff_data)

    assert np.asarray(vector).shape == (2,)
    assert np.all(np.isfinite(vector))


def test_replicator_2p4s_shape():
    payoff_data = examples.games.rps_with_twin.payoff_data
    vector = dynamics.replicator_2p4s([0.2, 0.3, 0.1], 0, payoff_data)

    assert np.asarray(vector).shape == (3,)
    assert np.all(np.isfinite(vector))


def test_replicator_3p2s_shape():
    payoff_data = examples.games.coordination_cube.payoff_data
    vector = dynamics.replicator_3p2s([0.2, 0.5, 0.8], 0, payoff_data)

    assert np.asarray(vector).shape == (3,)
    assert np.all(np.isfinite(vector))


def test_replicator_3p2s_coordinates_are_first_strategy_probabilities():
    payoff_tensors = []
    for player in range(3):
        tensor = np.zeros((2, 2, 2), dtype=float)
        for action_profile in np.ndindex(tensor.shape):
            tensor[action_profile] = 1.0 if action_profile[player] == 0 else 0.0
        payoff_tensors.append(tensor)

    vector = dynamics.replicator_3p2s([0.5, 0.5, 0.5], 0, tuple(payoff_tensors))

    np.testing.assert_allclose(vector, [0.25, 0.25, 0.25])


def test_compute_equilibria_returns_metadata():
    payoff_data = examples.games.matching_pennies.payoff_data
    equilibria = dynamics.compute_equilibria(payoff_data)

    assert isinstance(equilibria, dynamics.EquilibriumResult)
    assert hasattr(equilibria, "degenerate")
    assert hasattr(equilibria, "message")


def test_compute_equilibria_supports_one_population_hawk_dove():
    equilibria = dynamics.compute_equilibria(examples.games.hawk_dove.payoff_data)

    np.testing.assert_allclose(equilibria, [[0.0], [5 / 7], [1.0]])


def test_standard_one_population_games_cover_four_phase_line_regimes():
    games = examples.games

    assert all(
        game.game_class == "1Pop2S"
        for game in (
            games.prisoners_dilemma,
            games.hawk_dove,
            games.coordination_game,
            games.mutualism_game,
        )
    )
    assert dynamics.replicator_1d(0.25, games.prisoners_dilemma.payoff_data) < 0
    assert dynamics.replicator_1d(0.75, games.prisoners_dilemma.payoff_data) < 0
    assert dynamics.replicator_1d(0.25, games.hawk_dove.payoff_data) > 0
    assert dynamics.replicator_1d(0.75, games.hawk_dove.payoff_data) < 0
    assert dynamics.replicator_1d(0.25, games.coordination_game.payoff_data) < 0
    assert dynamics.replicator_1d(0.75, games.coordination_game.payoff_data) > 0
    assert dynamics.replicator_1d(0.25, games.mutualism_game.payoff_data) > 0
    assert dynamics.replicator_1d(0.75, games.mutualism_game.payoff_data) > 0


def test_compute_equilibria_filters_points_on_parametric_manifold():
    equilibria = dynamics.compute_equilibria(examples.games.hawk_dove_retaliator.payoff_data)

    assert equilibria.degenerate is True
    assert not any(np.allclose(eq, [0.0, 2 / 3]) for eq in equilibria)
    assert any(np.allclose(eq, [2 / 3, 1 / 3]) for eq in equilibria)
    assert any(np.allclose(eq, [1.0, 0.0]) for eq in equilibria)
    assert any(np.allclose(eq, [0.0, 1.0]) for eq in equilibria)
    assert any(np.allclose(eq, [0.0, 0.0]) for eq in equilibria)


def test_compute_equilibria_filters_repeated_pd_tft_allc_continuum_point():
    equilibria = dynamics.compute_equilibria(examples.games.repeated_pd_tft_allc_alld.payoff_data)

    assert equilibria.degenerate is True
    assert not any(np.allclose(eq, [2 / 3, 1 / 3]) for eq in equilibria)
    assert any(np.allclose(eq, [1.0, 0.0]) for eq in equilibria)
    assert any(np.allclose(eq, [0.0, 1.0]) for eq in equilibria)
