import warnings
import numpy as np
import pytest
import analysis, examples, dynamics
from game import Game
from analysis import _ess_on_critical_cone as ess_on_critical_cone


def analyze(game):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return analysis.analyze_equilibria(game)


def test_boundary_ess_and_noness():
    for value, expected in [(1,True),(-1,False),(0,False)]:
        g=Game('tie',np.array([[0,value,0],[0,0,0],[-1,0,0]],dtype=float))
        result=analyze(g)
        p=next(e for e in result.equilibria if np.allclose(e.full_position,[1,0,0]))
        assert p.nash and not p.strict_nash
        assert p.ess == expected


def test_cone_allows_positive_form_outside_feasible_directions():
    # On z=(-u-v,u,v), u,v>=0: q=-u²-4uv-v²<0;
    # nevertheless the unrestricted form is indefinite.
    a=np.array([[0,0,0],[0,-1,-2],[0,-2,-1]],dtype=float)
    assert ess_on_critical_cone(a,[0],[0,1,2])
    a[1,2]=a[2,1]=2
    assert not ess_on_critical_cone(a,[0],[0,1,2])


@pytest.mark.parametrize('data',[np.zeros((3,3)),(np.zeros((2,2)),np.zeros((2,2))),tuple(np.zeros((2,2,2)) for _ in range(3))])
def test_zero_games_report_degeneracy(data):
    r=analyze(Game('zero',data))
    assert r.degenerate
    assert all(e.nash and not e.strict_nash for e in r.equilibria)


def test_analysis_preserves_catalogue_rest_points():
    for name, game in examples.games.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw = dynamics.compute_equilibria(game.payoff_data)
        expected = [p for p in raw if analysis._is_state_in_domain(p, game.game_class)]
        result = analyze(game)
        actual = [e.reduced_position for e in result.equilibria]
        assert len(actual) == len(expected), name
        assert all(np.allclose(p, q) for p, q in zip(actual, expected)), name
        assert not hasattr(result, 'nash_complete')
        assert not result.to_dataframe().attrs


def test_mixed_ess_and_neutral_control():
    for game, expected in [(examples.games.good_rps, True), (examples.games.standard_rps, False)]:
        result = analyze(game)
        interior = next(e for e in result.equilibria if np.allclose(e.full_position, [1/3]*3))
        assert interior.ess == expected


def test_boundary_ess_is_strategy_order_invariant():
    a = np.array([[0,1,0],[0,0,0],[-1,0,0]],dtype=float)
    for order in ([0,1,2],[2,0,1],[1,2,0]):
        resident = order.index(0)
        best = [order.index(0),order.index(1)]
        assert ess_on_critical_cone(a[np.ix_(order,order)],[resident],best)


def test_analysis_does_not_require_pygambit():
    import subprocess
    import sys
    from pathlib import Path

    subprocess.run(
        [sys.executable, '-c', '''
import sys
sys.modules['pygambit'] = None
import analysis, examples
result = analysis.analyze_equilibria(examples.games.good_rps)
assert any(e.nash and e.ess for e in result.equilibria)
'''],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
