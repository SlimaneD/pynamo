"""Analytic reference cases and public plotting behavior for 1Pop2S."""
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import pynamo_egt as pn
from pynamo_egt import dynamics
from pynamo_egt._one_dimensional import _DisplayArrowhead


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.mark.parametrize('matrix,expected', [
    ([[1, 1], [0, 0]], .25),
    ([[0, 0], [1, 1]], -.25),
    ([[1, 0], [0, 1]], 0),
])
def test_vector_field(matrix, expected):
    assert dynamics.replicator_1d(.5, matrix) == expected


@pytest.mark.parametrize('matrix,stabilities', [
    ([[1, 0], [0, 1]], ['sink', 'source', 'sink']),
    ([[0, 1], [1, 0]], ['source', 'sink', 'source']),
])
def test_analysis(matrix, stabilities):
    game = pn.Game('reference', matrix)
    assert game.game_class == '1Pop2S'
    result = pn.analyze_equilibria(game)
    assert [p.stability for p in result.equilibria] == stabilities
    np.testing.assert_allclose([p.full_position for p in result.equilibria], [[0, 1], [.5, .5], [1, 0]])
    assert [p.ess for p in result.equilibria] == [s == 'sink' for s in stabilities]
    assert len(dynamics.compute_equilibria(game.payoff_data)) == 3


@pytest.mark.parametrize('matrix,endpoint,stable', [
    ([[-1, 0], [0, 0]], 0, True),
    ([[1, 0], [0, 0]], 0, False),
    ([[0, 1], [0, 0]], 1, True),
    ([[0, -1], [0, 0]], 1, False),
])
def test_nonhyperbolic_boundary(matrix, endpoint, stable):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = pn.analyze_equilibria(pn.Game('boundary', matrix))
    assert not caught
    p = next(p for p in result.equilibria if p.reduced_position[0] == endpoint)
    assert p.eigenvalues[0] == 0
    assert p.stability == ('sink' if stable else 'source')
    assert p.ess == stable


def test_continuum_and_color():
    game = pn.Game('neutral', [[2, 1], [2, 1]])
    result = pn.analyze_equilibria(game)
    assert result.degenerate and not result.equilibria
    fig, ax = pn.phase_portrait(game, continuum_color='orange')
    assert ax.lines[-1].get_color() == 'orange'
    assert not any(isinstance(a, _DisplayArrowhead) for a in ax.get_children())
    assert pn.rest_points_ess(game=game) == []


def test_centering_and_controls():
    game = pn.Game('coordination', [[1, 0], [0, 1]], strategy_labels=['A', 'B'])
    fig, ax = pn.phase_portrait(game, trajectory_color='navy', trajectory_linewidth=2,
                               show_equilibria=False, trajectory_zorder=1000)
    fig.canvas.draw()
    heads = [a for a in ax.get_children() if isinstance(a, _DisplayArrowhead)]
    assert len(heads) == 2
    for head, x in zip(heads, [.25, .75]):
        vertices = head.get_xy()[:4]
        np.testing.assert_allclose((vertices[:, 0].min()+vertices[:, 0].max())/2,
                                   ax.transData.transform((x, 0))[0])
    endpoint_labels = [t for t in ax.texts if t.get_text() in {'A', 'B'}]
    assert [(t.get_text(), t.xy, t.get_position()) for t in endpoint_labels] == [
        ('B', (0, 0), (0, -14)),
        ('A', (1, 0), (0, -14)),
    ]
    assert not ax.get_xticks().size
    assert ax.get_xlabel() == ''
    assert ax.lines[0].get_color() == 'navy'
    assert not ax.collections
    _, ax2 = pn.phase_portrait(game, trajectory_arrows=[], show_equilibria=False)
    assert not any(isinstance(a, _DisplayArrowhead) for a in ax2.get_children())


@pytest.mark.parametrize('positions', [[0], [1], [.5], [np.nan]])
def test_invalid_arrows(positions):
    with pytest.raises(ValueError):
        pn.phase_portrait(pn.Game('coordination', [[1, 0], [0, 1]]), trajectory_arrows=positions)


def test_bifurcation_labels_styles_and_reference_branch():
    fig, ax = pn.bifurcation_diagram(
        lambda s: [[s, 0], [1, 2]], parameter_range=(-1, 4),
        phase_line_values=[3, 0, 2], phase_line_labels=['II', '', 'II'],
        stable_color='navy', unstable_color='red', unstable_linestyle=':',
        stable_linewidth=3, show_equilibria=False, trajectory_arrows=[],
    )
    assert {t.xy[0]: t.get_text() for t in ax.texts} == {2: 'II', 3: 'II'}
    assert not ax.collections
    unstable = [line for line in ax.lines if line.get_label() == 'unstable branch']
    interior = unstable[1]
    x, y = interior.get_data()
    valid = np.isfinite(y)
    np.testing.assert_allclose(y[valid], 2/(x[valid]+1))
    assert interior.get_linestyle() == ':' and interior.get_color() == 'red'


def test_bifurcation_continuum_and_contextual_error():
    _, ax = pn.bifurcation_diagram(lambda s: [[s, 0], [0, s]], parameter_range=(-1, 1),
                                  parameter_values=[0], continuum_color='orange')
    np.testing.assert_allclose(ax.collections[0].get_colors()[0][:3], [1, 0.6470588235294118, 0])
    with pytest.raises(ValueError, match='parameter 3'):
        pn.bifurcation_diagram(lambda s: [[s, 0], [1, 2]], parameter_range=(0, 4),
                              phase_line_values=[3], trajectory_arrows=[.5])
    with pytest.raises(ValueError, match='one string'):
        pn.bifurcation_diagram(lambda s: [[s, 0], [1, 2]], parameter_range=(0, 4),
                              phase_line_values=[1, 2], phase_line_labels=['I'])
