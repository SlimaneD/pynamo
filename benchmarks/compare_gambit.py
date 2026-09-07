"""Run with an environment containing pygambit and pyNamo dependencies.
Outputs JSON to stdout; progress to stderr. Does not change library code.
"""
import sys, pathlib, json, time, warnings, itertools, platform
from importlib.metadata import version
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pygambit as gb
import examples, analysis
from game import Game

TOL = 1e-7

def arrays(g):
    a = g.payoff_data
    if g.symmetric:
        return [a, a.T]
    if g.game_class == '2P2S':
        return [a[0], a[1].T]  # pyNamo stores own action on each matrix's rows
    return list(a)

def profile(g, position):
    if g.symmetric:
        return [position, position]
    if g.game_class == '2P2S':
        return [np.array([p, 1-p]) for p in position]
    # Actual current dynamics/analysis convention: probability of tensor action 1.
    return [np.array([1-p, p]) for p in position]

def validate(payoffs, ps):
    valid = all(np.min(p) >= -TOL and abs(sum(p)-1) <= TOL for p in ps)
    regrets, strict = [], True
    for player, tensor in enumerate(payoffs):
        values = np.zeros(len(ps[player]))
        for actions in itertools.product(*(range(len(p)) for p in ps)):
            weight = np.prod([ps[j][a] for j,a in enumerate(actions) if j != player])
            values[actions[player]] += tensor[actions] * weight
        regrets.append(float(max(values) - np.dot(ps[player], values)))
        support = np.flatnonzero(ps[player] > TOL)
        strict &= len(support)==1 and all(values[support[0]] > values[j]+TOL for j in range(len(values)) if j != support[0])
    return {'valid': bool(valid and max(regrets)<=TOL), 'max_regret': max(regrets), 'strict': bool(strict)}

def same(a,b):
    return len(a)==len(b) and all(np.allclose(x,y,atol=TOL,rtol=0) for x,y in zip(a,b))

def run(name,g):
    print(name, file=sys.stderr, flush=True)
    payoffs = arrays(g)
    # Decimal string conversion preserves these catalogue payoffs exactly.
    gambit_game = gb.Game.from_arrays(*(np.array([[str(x) for x in row] for row in a]) if a.ndim==2 else a.astype(str) for a in payoffs))
    start=time.perf_counter()
    result = gb.nash.enummixed_solve(gambit_game, rational=True) if len(payoffs)==2 else gb.nash.enumpoly_solve(gambit_game, use_strategic=True, maxregret=1e-10)
    elapsed=time.perf_counter()-start
    gps=[ [np.array([float(p[s]) for s in player.strategies]) for player in gambit_game.players] for p in result.equilibria]
    raw_count = len(gps)
    unique = []
    for p in gps:
        if not any(same(p,q) for q in unique): unique.append(p)
    gps = unique
    relevant=[p for p in gps if not g.symmetric or same([p[0]],[p[1]])]
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        old=analysis.analyze_equilibria(g)
    oldtime=time.perf_counter()-start
    oldeq=[e for e in old.equilibria if e.nash]
    ops=[profile(g,e.full_position) for e in oldeq]
    witness_positions = {
        'zero_symmetric': [1/3,1/3,1/3],
        'zero_bimatrix': [.3,.7],
        'zero_three_player': [.2,.4,.6],
        'rps_with_twin': [1/3,1/3,1/6,1/6],
        'duplicate_coordination': [.25,.25,.5],
        'hawk_dove_retaliator': [0,.3,.7],
        'repeated_pd_tft_allc_alld': [.75,.25,0],
    }
    witnesses=[]
    if name in witness_positions:
        ps=profile(g,np.array(witness_positions[name]))
        witnesses.append(dict(position=witness_positions[name],check=validate(payoffs,ps),
            returned_by_gambit=any(same(ps,p) for p in relevant),
            returned_by_pynamo=any(same(ps,p) for p in ops)))
    return dict(name=name, gambit_raw_count=raw_count, witnesses=witnesses, game_class=g.game_class, symmetric=g.symmetric,
        gambit_seconds=elapsed,pynamo_seconds=oldtime,degenerate=old.degenerate,
        warnings=list(dict.fromkeys(str(w.message) for w in caught)),
        gambit_profiles=[[p.tolist() for p in ps] for ps in gps],
        gambit_checks=[validate(payoffs,p) for p in gps],
        symmetric_or_applicable_count=len(relevant),
        pynamo_profiles=[[p.tolist() for p in ps] for ps in ops],
        pynamo_checks=[validate(payoffs,p) for p in ops],
        pynamo_strict=[e.strict_nash for e in oldeq],pynamo_ess=[e.ess for e in oldeq],
        gambit_unmatched=[[p.tolist() for p in ps] for ps in relevant if not any(same(ps,o) for o in ops)],
        pynamo_unmatched=[[p.tolist() for p in ps] for ps in ops if not any(same(ps,o) for o in relevant)])

cases=list(examples.games.items()) + [
 ('zero_symmetric',Game('Zero',np.zeros((3,3)))),
 ('duplicate_coordination',Game('Duplicate',np.array([[1,1,0],[1,1,0],[0,0,1]]))),
 ('boundary_ess_tie',Game('Boundary ESS',np.array([[0,1,0],[0,0,0],[-1,0,0]]))),
 ('boundary_noness_tie',Game('Boundary nonESS',np.array([[0,-1,0],[0,0,0],[-1,0,0]]))),
 ('zero_bimatrix',Game('Zero bimatrix',(np.zeros((2,2)),np.zeros((2,2))))),
 ('zero_three_player',Game('Zero tensors',tuple(np.zeros((2,2,2)) for _ in range(3)))),
]
if __name__=='__main__':
    selected=set(sys.argv[1:])
    rows=[]
    for name,g in cases:
        if selected and name not in selected: continue
        try: rows.append(run(name,g))
        except Exception as exc: rows.append(dict(name=name,error=repr(exc)))
    print(json.dumps(dict(python=platform.python_version(),versions={p:version(p) for p in ['pygambit','numpy','scipy','sympy','pandas']},tolerance=TOL,results=rows),indent=2))
