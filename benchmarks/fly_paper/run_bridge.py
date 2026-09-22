"""Known-answer finance-to-memory interface assays, NOT a trading backtest.

SPDX-License-Identifier: GPL-3.0-or-later
The memory equations derive from Luo/Huang 2024; the financial mapping is new,
unvalidated engineering. Forced cue/outcome exposure is not policy learning.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmarks.fly_paper.model import UPSTREAM_COMMIT, parameter_matrices, reset_valence
from benchmarks.fly_paper.online import CausalMemory
from benchmarks.verified_memory.episode_credit import (
    CreditGate, IntervalCredit, NavMark, OptionReceipt,
)


def make_receipt(identifier, index, values, delay=2):
    origin = pd.Timestamp("2025-01-01")+pd.Timedelta(days=index*10)
    marks = [NavMark(origin+pd.Timedelta(days=i+1),origin+pd.Timedelta(days=i+1+delay),
                     float(nav),"synthetic_reconciled_account") for i,nav in enumerate(values)]
    intervals = tuple(IntervalCredit(a,b,math.log(b.nav/a.nav)) for a,b in zip(marks,marks[1:]))
    return OptionReceipt(identifier,origin,intervals,marks[-1].available_at)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parameters",type=Path,required=True)
    p.add_argument("--reproduction",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    reproduction=json.loads(args.reproduction.read_text(encoding="utf8"))
    if not reproduction["all_checks_passed"] or reproduction["upstream_commit"]!=UPSTREAM_COMMIT:
        raise RuntimeError("Native reproduction gate has not passed")
    args.output.mkdir(parents=True,exist_ok=False)
    protocol={"stage":"finance-to-memory interface assay, not policy evaluation",
        "parameter_sha256":hashlib.sha256(args.parameters.read_bytes()).hexdigest(),
        "reproduction_sha256":hashlib.sha256(args.reproduction.read_bytes()).hexdigest(),
        "conditioning_bouts_per_case":40,"reversal_blocks":24,"bouts_per_reversal_block":20,
        "fixed_log_return_scale":.01,"conditioning_seconds":30,"rest_seconds":135,
        "cases":{"delayed_gain":[1.,.998,1.005984],
                 "sweet_but_net_loss":[1.,.999, .9985],
                 "valid_large_loss":[1.,.99,.95],"neutral":[1.,1.,1.]},
        "engineering_changes":["past-only retention clock","signed return to punishment drive",
            "non-mutating mean normalized MBON probe","whole-option gamma=1 log-wealth target"],
        "exposure":"development assays; not sealed or unseen data; one central fitted parameter vector"}
    (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf8")
    d=loadmat(args.parameters)
    params=reset_valence(parameter_matrices(d["para_mu"].T,d["mat_lu_cell"]),0.)
    results={"cases":{},"checks":{},"reversal_blocks":[]}
    applied=negative=early=invalid=duplicates=0
    for name,values in protocol["cases"].items():
        brain=CausalMemory(params); gate=CreditGate(); trace=[]
        for i in range(protocol["conditioning_bouts_per_case"]):
            receipt=make_receipt(f"{name}-{i}",i,values)
            before=brain.weights.copy()
            pending=gate.consume(receipt,receipt.claimed_available)
            if pending["status"]=="ready": early+=1
            assert pending["target"] is None
            # Broken clocks are checked by fin-skills, and must not consume the ID.
            bad=replace(receipt,claimed_available=receipt.intervals[-1].after.event_time)
            rejected=gate.consume(bad,receipt.claimed_available+pd.Timedelta(days=1))
            invalid+=rejected["status"]=="invalid"
            assert rejected["target"] is None
            np.testing.assert_array_equal(brain.weights,before)
            ready=gate.consume(receipt,receipt.claimed_available+pd.Timedelta(days=1))
            assert ready["status"]=="ready"
            brain.reinforce_return(ready["target"])
            applied+=1; negative+=ready["target"]<0
            duplicate=gate.consume(receipt,receipt.claimed_available+pd.Timedelta(days=2))
            duplicates+=duplicate["status"]=="duplicate"
            assert duplicate["target"] is None
            scores=brain.cue_scores()
            trace.append({"bout":i,"target":ready["target"],"scores":scores.tolist(),
                          "preference":float(scores[0]-scores[1])})
        results["cases"][name]={"full_log_target":math.log(values[-1]/values[0]),
            "first_interval_target":math.log(values[1]/values[0]),"trace":trace}
    results["checks"]["whole_episode_gain_has_positive_preference"] = results["cases"]["delayed_gain"]["trace"][-1]["preference"]>0
    results["checks"]["net_loss_has_negative_preference"] = results["cases"]["sweet_but_net_loss"]["trace"][-1]["preference"]<0
    results["checks"]["large_valid_loss_is_learned"] = results["cases"]["valid_large_loss"]["trace"][-1]["preference"]<0
    # Neutral stimulus adaptation may cause numerical/circuit bias; record it,
    # and require an explicitly declared small band, not exact mathematical zero.
    results["checks"]["neutral_preference_abs_below_0_02"] = abs(results["cases"]["neutral"]["trace"][-1]["preference"])<.02
    brain=CausalMemory(params)
    for block in range(protocol["reversal_blocks"]):
        reward=.005 if block%2==0 else -.005
        for _ in range(protocol["bouts_per_reversal_block"]):brain.reinforce_return(reward)
        scores=brain.cue_scores(); preference=float(scores[0]-scores[1])
        results["reversal_blocks"].append({"block":block,"log_return":reward,"preference":preference,
            "correct_sign":bool(preference*reward>0),"max_abs_weight":float(np.abs(brain.weights).max())})
    results["checks"]["all_reversal_block_end_signs_correct"] = all(b["correct_sign"] for b in results["reversal_blocks"])
    results["checks"]["no_early_learning"] = early==0
    results["gate_counts"]={"applied":applied,"negative_applied":negative,
        "early_applied":early,"invalid_rejected":invalid,"duplicates_rejected":duplicates}
    results["all_checks_passed"]=all(results["checks"].values())
    results["limitations"]=["Forced outcomes, no learned entry/exit policy or market execution",
        "Cue comparison readout is not a calibrated return estimate",
        "Clock reset, financial scaling and signed reward extension are engineering choices",
        "No sample-efficiency or economic advantage demonstrated",
        "reversal assay isolates memory and bypasses the separately tested receipt gate"]
    (args.output/"results.json").write_text(json.dumps(results,indent=2),encoding="utf8")
    print(json.dumps({"all_checks_passed":results["all_checks_passed"],"checks":results["checks"],
                      "gate_counts":results["gate_counts"]}),flush=True)
    if not results["all_checks_passed"]:raise SystemExit(1)


if __name__=="__main__":main()
