import numpy as np

import dynamics
import parameters as param


def test_replicator_2p2s_shape():
    payoff_data = param.available_games("2P2S")[1].payoff_data
    vector = dynamics.replicator_2p2s([0.4, 0.6], 0, payoff_data)

    assert np.asarray(vector).shape == (2,)
    assert np.all(np.isfinite(vector))


def test_replicator_2p3s_shape():
    payoff_data = param.available_games("2P3S")[1].payoff_data
    vector = dynamics.replicator_2p3s([0.3, 0.4], 0, payoff_data)

    assert np.asarray(vector).shape == (2,)
    assert np.all(np.isfinite(vector))


def test_replicator_2p4s_shape():
    payoff_data = param.available_games("2P4S")[1].payoff_data
    vector = dynamics.replicator_2p4s([0.2, 0.3, 0.1], 0, payoff_data)

    assert np.asarray(vector).shape == (3,)
    assert np.all(np.isfinite(vector))


def test_replicator_3pop2s_shape():
    payoff_data = param.available_games("3P2S")[1].payoff_data
    vector = dynamics.replicator_3pop2s([0.2, 0.5, 0.8], 0, payoff_data)

    assert np.asarray(vector).shape == (3,)
    assert np.all(np.isfinite(vector))


def test_compute_equilibria_returns_metadata():
    payoff_data = param.available_games("2P2S")[1].payoff_data
    equilibria = dynamics.compute_equilibria(payoff_data)

    assert isinstance(equilibria, dynamics.EquilibriumResult)
    assert hasattr(equilibria, "degenerate")
    assert hasattr(equilibria, "message")
