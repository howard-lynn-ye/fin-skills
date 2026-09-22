"""Small known-answer sequential-control assay using the adapted memory actor.

SPDX-License-Identifier: GPL-3.0-or-later
Original neural equations: Luo/Huang 2024. Actor/critic, financial environment,
state indexing and readout are engineering additions, not the published model.
"""
import argparse
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmarks.fly_paper.model import parameter_matrices,reset_valence
from benchmarks.fly_paper.online import CausalMemory
from benchmarks.verified_memory.episode_credit import CreditGate,IntervalCredit,NavMark,OptionReceipt


@dataclass(frozen=True)
class Task:
    name: str
    prices: tuple
    cues: tuple
    allocation: float=1.
    fee: float=.001


TASKS = (
    Task("entry_cost_then_gain",(1.,1.,1.008,1.016,1.024),(1,1,1,1)),
    Task("sweet_but_costly",(1.,1.,1.0002,1.0004,1.0006),(1,1,1,1)),
    Task("persistent_patch",(1.,1.,1.004,1.008,1.012,1.016,1.02),(1,1,1,1,1,1)),
    Task("patch_deteriorates",(1.,1.,1.008,1.016,1.008,.992,.984),(1,1,-1,-1,-1,-1)),
    Task("transient_bad_cue",(1.,1.,1.004,1.008,1.012,1.016,1.02),(1,1,-1,1,1,1)),
    Task("quarter_capital_budget",(1.,1.,1.008,1.016,1.024),(1,1,1,1),.25),
    Task("no_capital_budget",(1.,1.,1.008,1.016,1.024),(1,1,1,1),0.),
    Task("no_opportunity",(1.,1.,1.,1.,1.),(0,0,0,0)),
)


def execute(cash,units,price,action,allocation,fee):
    """Action 0 preserves current state; action 1 enters when flat/exits when held.

    Allocation is the fraction of available cash spent INCLUDING fees at entry.
    A held position is never rebalanced by HOLD. No leverage or shorting.
    """
    held=units>1e-14
    cost=turnover=0.
    if action==1 and held:
        notional=units*price; cost=notional*fee; turnover=notional
        cash+=notional-cost; units=0.
    elif action==1 and allocation>0:
        budget=cash*allocation; notional=budget/(1+fee)
        cost=budget-notional; turnover=notional
        units=notional/price; cash-=budget
    return cash,units,cost,turnover


def score_fixed(task,actions):
    """Independent evaluator: direct all-cash/share formulas, no learner imports."""
    cash,units=1.,0.
    for t,a in enumerate(actions):
        px=task.prices[t+1]
        if a and units>1e-14:
            cash+=units*px*(1-task.fee); units=0.
        elif a:
            spent=cash*task.allocation
            units=spent/(px*(1+task.fee)); cash-=spent
    return cash+units*task.prices[-1]*(1-task.fee)


def optimal(task):
    # The scorer sees the full path; the learner never calls this function.
    best=-math.inf; paths=[]
    for actions in itertools.product((0,1),repeat=len(task.cues)):
        value=score_fixed(task,actions)
        if value>best+1e-12:best=value;paths=[actions]
        elif abs(value-best)<1e-12:paths.append(actions)
    return best,paths


class Learner:
    def __init__(self,kind,params,seed):
        self.kind,self.params=kind,params
        self.rng=np.random.default_rng(seed)
        self.models={}; self.values={}; self.targets={}
        self.gate=CreditGate()
        self.applied=0;self.negative=0

    def scores(self,state):
        if self.kind=="fly_actor_critic":
            if state not in self.models:self.models[state]=CausalMemory(self.params)
            return self.models[state].cue_scores()
        if state not in self.targets:self.targets[state]=np.zeros(2)
        return self.targets[state]

    def choose(self,state,train,allocation):
        scores=self.scores(state)
        if allocation==0:return 0
        if train and self.rng.random()<.2:return int(self.rng.integers(2))
        # Stay/hold wins an exact tie in evaluation. During training, random ties
        # prevent the neutral initialized memory from fixing an action forever.
        best=np.flatnonzero(np.isclose(scores,scores.max(),atol=1e-12,rtol=0))
        return int(self.rng.choice(best)) if train else int(best[0])

    def learn(self,state,action,receipt,now):
        result=self.gate.consume(receipt,now)
        if result["status"]!="ready":raise AssertionError(result)
        target=result["target"]
        if self.kind=="fly_actor_critic":
            value=self.values.get(state,0.)
            advantage=target-value
            self.models[state].reinforce_return(advantage,cue=action)
            self.values[state]=value+.1*advantage
        else:
            self.targets[state][action]+=.1*(target-self.targets[state][action])
        self.applied+=1;self.negative+=target<0


def rollout(task,learner,episode,*,train):
    cash,units=1.,0.;total_fees=turnover=0.
    decisions=[];marks=[];navs=[1.];actions=[]
    origin=pd.Timestamp("2020-01-01")+pd.Timedelta(days=episode*20)
    def mark(event,nav):
        return NavMark(event,event+pd.Timedelta(hours=2),float(nav),"synthetic_cash_share_ledger")
    for t,cue in enumerate(task.cues):
        held=units>1e-14
        # Observable state only. Time to the known episode deadline is allowed.
        # This small lookup is an assay, not a general market representation.
        state=(len(task.cues)-t,int(cue),int(task.cues[t-1] if t else 0),held,task.allocation)
        action=learner.choose(state,train,task.allocation)
        stamp=origin+pd.Timedelta(days=t+1)
        pre=cash+units*task.prices[t+1]
        marks.append(mark(stamp,pre));begin=len(marks)-1
        before_cash,before_units=cash,units
        cash,units,fee,traded=execute(cash,units,task.prices[t+1],action,task.allocation,task.fee)
        post=cash+units*task.prices[t+1]
        assert abs(post+fee-pre)<1e-12
        assert cash>=-1e-12 and units>=0
        if not held and action:
            assert (before_cash-cash)<=before_cash*task.allocation+1e-12
        marks.append(mark(stamp+pd.Timedelta(minutes=1),post))
        decisions.append((state,action,begin,origin+pd.Timedelta(days=t,hours=12)))
        actions.append({"step":t,"cue":cue,"state_held":bool(held),"action":
            ("exit" if held else "enter") if action else ("hold" if held else "wait"),
            "code":action,"fee":fee,"cash":cash,"units":units})
        total_fees+=fee;turnover+=traded;navs.append(post)
    exit_fee=units*task.prices[-1]*task.fee
    cash+=units*task.prices[-1]-exit_fee;units=0.
    total_fees+=exit_fee
    marks.append(mark(origin+pd.Timedelta(days=len(task.cues),minutes=2),cash))
    independent=score_fixed(task,[a["code"] for a in actions])
    assert abs(independent-cash)<1e-12
    if train:
        for j,(state,action,start,decision_time) in enumerate(decisions):
            tail=marks[start:]
            intervals=tuple(IntervalCredit(a,b,math.log(b.nav/a.nav)) for a,b in zip(tail,tail[1:]))
            receipt=OptionReceipt(f"{task.name}-{episode}-{j}",decision_time,intervals,tail[-1].available_at)
            learner.learn(state,action,receipt,tail[-1].available_at+pd.Timedelta(hours=1))
    return {"terminal_wealth":cash,"log_return":math.log(cash),"fees":total_fees,
            "traded_notional_excluding_terminal":turnover,"actions":actions,"nav":navs+[cash]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parameters",type=Path,required=True)
    p.add_argument("--bridge",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if not json.loads(args.bridge.read_text())["all_checks_passed"]:raise RuntimeError("Interface gate failed")
    args.output.mkdir(parents=True,exist_ok=False)
    protocol={"training_episodes":400,"seeds":[11,23,37,53,71],"exploration_probability":.2,
        "critic_or_value_learning_rate":.1,"reward":"undiscounted suffix log wealth, all actual fees once",
        "arms":["fly_actor_critic","ordinary_mc_value"],"tasks":[t.__dict__ for t in TASKS],
        "success_rule":"greedy terminal wealth within 1e-10 of independently enumerated optimum",
        "exposure":"fixed known-answer development tasks, no generalization/market claim",
        "engineering_additions":["tabular observable states and independent circuits per state",
            "ordinary scalar Monte Carlo critic","signed advantage reinforcement of chosen cue/action",
            "persistent cash/share account and true HOLD","whole-episode suffix credit"],
        "limitations":["deterministic laboratory paths","time-to-deadline is observed",
            "independent task training","no learned representation or internal-state gate beyond constraints",
            "memory clock measures conditioning exposure, not market elapsed time",
            "same training episodes do not imply equal runtime or parameter count"]}
    (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2))
    d=loadmat(args.parameters)
    params=reset_valence(parameter_matrices(d["para_mu"].T,d["mat_lu_cell"]),0.)
    rows=[]
    for task in TASKS:
        optimum,paths=optimal(task)
        for arm in protocol["arms"]:
            for seed in protocol["seeds"]:
                learner=Learner(arm,params,seed)
                for episode in range(protocol["training_episodes"]):
                    rollout(task,learner,episode,train=True)
                evaluation=rollout(task,learner,protocol["training_episodes"],train=False)
                rows.append({"task":task.name,"arm":arm,"seed":seed,**evaluation,
                    "optimum_wealth":optimum,"regret":optimum-evaluation["terminal_wealth"],
                    "reaches_optimum":bool(abs(optimum-evaluation["terminal_wealth"])<=1e-10),
                    "updates":learner.applied,"negative_updates":learner.negative})
            print(task.name,arm,"completed",flush=True)
    summary=[]
    for task in TASKS:
        for arm in protocol["arms"]:
            selected=[r for r in rows if r["task"]==task.name and r["arm"]==arm]
            summary.append({"task":task.name,"arm":arm,
                "optimal_seeds":sum(r["reaches_optimum"] for r in selected),
                "total_seeds":len(selected),"mean_regret":float(np.mean([r["regret"] for r in selected])),
                "max_regret":max(r["regret"] for r in selected)})
    results={"rows":rows,"summary":summary,
        "all_fly_cases_reach_optimum":all(r["reaches_optimum"] for r in rows if r["arm"]=="fly_actor_critic"),
        "interpretation":"mechanism/control diagnostic; failure is reported, not a reason to tune this frozen run"}
    (args.output/"results.json").write_text(json.dumps(results,indent=2))
    print(json.dumps({"all_fly_cases_reach_optimum":results["all_fly_cases_reach_optimum"],"summary":summary}),flush=True)


if __name__=="__main__":main()
