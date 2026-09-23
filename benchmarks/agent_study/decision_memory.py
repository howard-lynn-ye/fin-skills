"""Auditable LLM-facing episodic memory with an optional real causal fly circuit.

Correctness reinforcement is a new engineering mapping, not a financial return or
the original biological task. No synthesized NAV receipts are used for this mapping.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.agent_study.decision_tasks import PAIRS, digest

PRIOR = Path('/beacon-projects/radfm/wy891/fin-skills-memory-paper-20260921-mamba')


def load_parameters():
    from scipy.io import loadmat
    from benchmarks.fly_paper.model import UPSTREAM_COMMIT, parameter_matrices, reset_valence
    path = PRIOR / 'upstream/data_and_parameters/Dx_steady_state_nonlinear_3_27-Mar-2023_3modules.mat'
    audit = PRIOR / 'port3/results/results.json'
    receipt = json.loads(audit.read_text())
    if not receipt['all_checks_passed'] or receipt['upstream_commit'] != UPSTREAM_COMMIT:
        raise RuntimeError('native equation reproduction has not passed')
    raw = loadmat(path)
    params = reset_valence(parameter_matrices(raw['para_mu'].T, raw['mat_lu_cell']), 0.)
    return params, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, audit)}


class ExperienceMemory:
    def __init__(self, mode, params=None):
        if mode not in ('none', 'frozen', 'reflection', 'retrieval', 'fly'):
            raise ValueError('unknown memory intervention')
        self.mode, self.params = mode, params
        self.records, self.brains, self.applied = [], {}, set()

    def state(self):
        return dict(records=[r['id'] for r in self.records],
            circuits={k: dict(weights=b.weights.tolist(), clock=b.clock,
                             scores=b.cue_scores().tolist()) for k, b in self.brains.items()})

    def read(self, case):
        before = digest(self.state())
        relevant = [r for r in self.records if r['available_day'] < case['decision_day']
                    and r['client'] == case['client']]
        if self.mode == 'none':
            relevant = []
        if self.mode == 'reflection':
            selected = relevant[-3:]
            items = [dict(id=r['id'], text=r['reflection']) for r in selected]
        else:
            selected = relevant[-3:]
            items = [dict(id=r['id'], text=json.dumps(r, sort_keys=True)) for r in selected]
        scores = None
        if self.mode in ('fly', 'frozen') and case['client'] in self.brains:
            scores = dict(zip(PAIRS[case['family']], self.brains[case['client']].cue_scores().tolist()))
            items.append(dict(id=f"circuit-{case['client']}", text=json.dumps({
                'past_experience_cue_scores': scores, 'meaning': 'Uncalibrated circuit approach scores, not expected returns; use with recorded feedback.'})))
        if before != digest(self.state()):
            raise AssertionError('reading memory mutated it')
        return items, dict(state_sha256=before, experience_ids=[r['id'] for r in selected],
                           scores=scores, actual_items=copy.deepcopy(items))

    def update(self, case, action, score, reflection):
        if case['id'] in self.applied:
            raise ValueError('duplicate experience update')
        before = digest(self.state())
        self.applied.add(case['id'])
        if self.mode == 'none' or (self.mode == 'frozen' and case['phase'] == 'later_evaluation'):
            return dict(updated=False, before=before, after=before)
        record = dict(id=case['id'], client=case['client'], family=case['family'],
            available_day=case['feedback_available_day'], action=action,
            correct=score['correct'], reflection=reflection,
            feedback_contract=case['policy'])
        self.records.append(record)
        if self.mode in ('fly', 'frozen'):
            from benchmarks.fly_paper.online import CausalMemory
            brain = self.brains.setdefault(case['client'], CausalMemory(self.params))
            method = action.get('method')
            if method in PAIRS[case['family']]:
                cue = PAIRS[case['family']].index(method)
                reward = 1. if score['correct'] else -1.
                # Explicit task reward, not a fabricated return or account value.
                brain.step(30., np.eye(2)[cue], -reward, conditioning=True)
                brain.step(135.)
        return dict(updated=True, before=before, after=digest(self.state()),
                    reward_mapping='correct=+1, all incorrect/invalid=-1; chosen valid cue only')
