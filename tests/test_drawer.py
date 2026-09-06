import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import drawer
import examples


def test_phase_portrait_runs_for_all_supported_game_classes():
    for game_class in ("2P2S", "2P3S", "2P4S", "3P2S"):
        g = next(iter(examples.games.by_class(game_class).values()))
        fig, ax = drawer.phase_portrait(
            g,
            random_state=0,
            tmax=0.2,
            trajectory_arrows=[],
            show_speed=game_class in ("2P2S", "2P3S"),
            show_vector_field=True,
        )

        assert fig is not None
        assert ax is not None
        plt.close(fig)


def test_phase_portrait_accepts_one_color_per_trajectory():
    g = examples.games.matching_pennies
    fig, ax = drawer.phase_portrait(
        g,
        starts=[[0.2, 0.7], [0.8, 0.3]],
        trajectory_color=["tab:blue", "tab:orange"],
        tmax=0.2,
        trajectory_arrows=[],
    )

    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_plot_vector_field_only():
    g = examples.games.hawk_dove
    fig, ax = drawer.phase_portrait(
        g,
        show_trajectories=False,
        show_equilibria=False,
        show_speed=False,
        show_vector_field=True,
        vector_margin=0.0,
    )

    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_phase_portrait_accepts_equilibrium_edgecolor():
    g = examples.games.good_rps
    fig, ax = drawer.phase_portrait(
        g,
        show_trajectories=False,
        equilibrium_edgecolor="firebrick",
    )

    assert fig is not None
    assert ax is not None
    plt.close(fig)
