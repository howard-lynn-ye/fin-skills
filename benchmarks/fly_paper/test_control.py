import math

import numpy as np
import pytest

from benchmarks.fly_paper.run_control import TASKS, execute, optimal, rollout, score_fixed


class Fixed:
    def __init__(self,actions):self.actions=iter(actions);self.targets=[]
    def choose(self,*args):return next(self.actions)
    def learn(self,state,action,receipt,now):
        from benchmarks.verified_memory.episode_credit import audit_option
        result=audit_option(receipt,now)
        assert result['status']=='ready'
        self.targets.append(result['target'])


def test_costs_are_self_financing_and_hold_preserves_units():
    cash,units,fee,_=execute(1.,0.,10.,1,.25,.001)
    assert cash+units*10+fee==pytest.approx(1.)
    assert cash==pytest.approx(.75)
    next_cash,next_units,fee,_=execute(cash,units,12.,0,.25,.001)
    assert (next_cash,next_units,fee)==(cash,units,0.)


def test_favorable_entry_receives_full_later_credit():
    task=TASKS[0]
    learner=Fixed([1,0,0,0])
    result=rollout(task,learner,0,train=True)
    expected=1.024*(1-.001)/(1+.001)
    assert result['terminal_wealth']==pytest.approx(expected)
    assert learner.targets[0]==pytest.approx(math.log(expected))


def test_costly_and_zero_budget_tasks_have_cash_optimum():
    for task in (TASKS[1],TASKS[6],TASKS[7]):
        best,_=optimal(task)
        assert best==pytest.approx(1.)


def test_exhaustive_evaluator_matches_episode_for_all_deterioration_paths():
    import itertools
    task=TASKS[3]
    for path in itertools.product((0,1),repeat=len(task.cues)):
        result=rollout(task,Fixed(path),0,train=False)
        assert result['terminal_wealth']==pytest.approx(score_fixed(task,path),abs=1e-12)


def test_future_price_change_cannot_change_first_trade():
    from dataclasses import replace
    task=TASKS[0]
    altered=replace(task,prices=(*task.prices[:2],.5,.2,.1))
    a=rollout(task,Fixed([1,0,0,0]),0,train=False)
    b=rollout(altered,Fixed([1,0,0,0]),0,train=False)
    assert a['actions'][0]==b['actions'][0]
    assert a['terminal_wealth']!=b['terminal_wealth']
