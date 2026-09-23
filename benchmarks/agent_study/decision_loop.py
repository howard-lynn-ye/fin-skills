"""Shared financial agent loop: tools, episodic memory and physical context pruning."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from benchmarks.agent_study.decision_tasks import (
    PAIRS, PARAMETERS, RULES, cases, clean, digest, execute, grade, reference,
)
from benchmarks.agent_study.decision_memory import ExperienceMemory, load_parameters
from benchmarks.agent_study.histrim import PackedChat
from benchmarks.agent_study.transformers_chat import TransformersChat

SYSTEM = ('You are a financial research agent. Follow the current client contract. '
          'Historical experience is fallible. Return exactly the requested JSON object, '
          'without markdown. Use the available numerical API rather than inventing values.')
GROUPS = {
    'tools': [('generic','none','full'), ('guidance','none','full'), ('finskills','none','full')],
    'memory': [('finskills',m,'full') for m in ('frozen','reflection','retrieval','fly')],
    'context': [('finskills',m,c) for c in ('recency','histrim') for m in ('none','fly')],
}
# Prospective seed replication puts all fixed conditions in one GPU allocation.
# The deployed seed-11 source snapshot remains unchanged.
GROUPS['joint'] = [arm for group in ('tools','memory','context') for arm in GROUPS[group]]


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(clean(value), f, indent=2, allow_nan=False)


def parse(text):
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('response must be an object')
    return value


def catalog(style):
    names = list(PARAMETERS)
    mappings = {f'function_{i}': name for i,name in enumerate(names)}
    if style == 'finskills':
        mappings = {name:name for name in names}
    rows = [dict(id=external, implements=internal, parameters=PARAMETERS[internal])
            for external,internal in mappings.items()]
    # Same numerical implementation and input data in all three conditions.
    # Guidance contains the exact same rule text as the organized FinSkills arm.
    if style == 'finskills':
        for row in rows:
            row['domain_note'] = RULES[row['implements']]
    guidance = '\n'.join(RULES.values()) if style == 'guidance' else ''
    return rows, mappings, guidance


def distractors(case):
    items = []
    for family, methods in PAIRS.items():
        for method in methods:
            items.append(dict(id='manual-'+method, text=f"Reference manual for {family}: {RULES[method]} Default API parameters: {json.dumps(PARAMETERS[method])}. These defaults do not override the client contract."))
    # Deterministic rotation varies evidence position without using method labels.
    shift = case['episode'] % len(items)
    return items[shift:]+items[:shift]


def calibration_examples():
    result = []
    # Only manual-to-task relevance labels, no evaluation answers or returns.
    for i,family in enumerate(PAIRS):
        case = dict(episode=i)
        items = distractors(case)
        result.append(dict(system=SYSTEM, protected=f"Review methods for a {family} task. Reply with an applicable method ID.",
            items=items, labels=[1. if item['id'].removeprefix('manual-') in PAIRS[family] else -1. for item in items]))
    return result


def qualification(chat, params, output):
    checks = []
    for case in cases(101):
        receipt = execute(case, case['method'], case['parameters'])
        checks.append(dict(case=case['id'], passed=bool(np.allclose(receipt['values'], reference(case), rtol=1e-10, atol=1e-12))))
    write(output/'reference-calibration.json', checks)
    if not all(x['passed'] for x in checks):
        raise RuntimeError('numerical reference calibration failed')
    example = calibration_examples()[0]
    equivalence = chat.equivalence(example['system'], example['protected'], example['items'])
    write(output/'decoder-equivalence.json', equivalence)
    if not equivalence['passed']:
        raise RuntimeError('manual decoder disagrees with official full-context logits')
    started = time.monotonic()
    routers = chat.calibrate(calibration_examples())
    write(output/'routers.json', dict(implementation='new paper-equation implementation',
        calibration='four manual-relevance prompts, no task output labels',
        elapsed_seconds=time.monotonic()-started, layers=routers,
        objective='item relevance squared error, ridge .1, 24 paired tangent refinements',
        threshold='separate calibration classification threshold; shared token caps 50% then 25%',
        beta=.1, batch_size=1))
    chat.max_tokens = 8
    traces = {}
    for mode in ('full','recency','histrim'):
        traces[mode] = chat(example['system'], example['protected'], example['items'], mode=mode, seed=97)
    chat.max_tokens = 384
    base_bytes = traces['full']['prefill_kv_bytes']
    compact = all(traces[m]['prefill_kv_bytes'] < base_bytes for m in ('recency','histrim'))
    protected_ok = nested_ok = True
    _, assignments, _ = chat.encode(example['system'],example['protected'],example['items'])
    protected = set(np.flatnonzero(assignments < 0).tolist())
    for mode in ('recency','histrim'):
        for row in traces[mode]['selection']:
            now, before = set(row['retained_original_positions']), set(row['previous_positions'])
            protected_ok &= protected <= now
            nested_ok &= now <= before
    report = dict(compact=compact, protected_ok=bool(protected_ok), nested_ok=bool(nested_ok), traces=traces)
    write(output/'physical-cache-check.json', report)
    if not compact or not protected_ok or not nested_ok:
        raise RuntimeError('physical cache/protection/nesting qualification failed')
    return dict(passed=True, numeric_cases=len(checks), equivalence=equivalence,
                physical_compaction=True, scope='engineering qualification, not a task quality score')


def run_episode(chat, case, arm, memory, folder, seed):
    style, memory_mode, context = arm
    rows, mappings, guidance = catalog(style)
    experiences, memory_read = memory.read(case)
    items = distractors(case)+experiences
    protected = json.dumps(dict(client=case['client'], family=case['family'],
        observed_block=case['observed_block'], decision_day=case['decision_day'],
        contract=case['policy'], input_summary={k: dict(shape=np.asarray(v).shape,
            first=clean(np.asarray(v)[0]), last=clean(np.asarray(v)[-1])) for k,v in case['data'].items()},
        data_sha256=case['data_sha256'], available_tools=rows))
    protected += '\n'+guidance+'\nChoose one API call: {"tool_id": "ID", "parameters": {...}, "reason": "short reason"}.'
    record = dict(case=case['id'], phase=case['phase'], family=case['family'], arm=arm,
        memory_read=memory_read, input_sha256=case['data_sha256'], calls=[], action={}, final={},
        errors=[], receipt=None)
    started = time.monotonic()
    try:
        response = chat(SYSTEM, protected, items, mode=context, seed=seed)
        record['calls'].append(response)
        action = parse(response['text'])
        action['method'] = mappings[action['tool_id']]
        record['action'] = action
        tick = time.monotonic()
        record['receipt'] = execute(case, action['method'], action['parameters'])
        record['tool_seconds'] = time.monotonic()-tick
    except Exception as exc:
        record['errors'].append(dict(stage='selection_or_execution', type=type(exc).__name__, message=str(exc)))
    # Every condition has exactly two action-response opportunities even after
    # an invalid first action. No secret retries or extra repair calls.
    final_prompt = protected+'\nSelected action: '+json.dumps(record['action'])
    final_prompt += '\nAPI result: '+json.dumps(record['receipt'] or record['errors'])
    final_prompt += '\nReturn {"values": [numbers in API order], "receipt_sha256": "exact hash", "reason": "how this answers the contract"}. No additional tool call.'
    try:
        response = chat(SYSTEM, final_prompt, items, mode=context, seed=seed+1)
        record['calls'].append(response)
        record['final'] = parse(response['text'])
    except Exception as exc:
        record['errors'].append(dict(stage='final', type=type(exc).__name__, message=str(exc)))
    record['grade'] = grade(case, record['action'], record['final'], record['receipt'])
    # Feedback and reflection opportunity are identical; storage/readout differs.
    feedback = dict(grade=record['grade'], contract=case['policy'], action=record['action'])
    reflection_prompt = 'Completed episode feedback: '+json.dumps(feedback)+'\nReturn {"lesson": "A short lesson about method, parameters and output use for this client"}.'
    reflection = ''
    try:
        response = chat(SYSTEM, reflection_prompt, [], mode='full', seed=seed+2)
        record['calls'].append(response)
        reflection = str(parse(response['text']).get('lesson',''))
    except Exception as exc:
        record['errors'].append(dict(stage='reflection', type=type(exc).__name__, message=str(exc)))
    record['memory_update'] = memory.update(case, record['action'], record['grade'], reflection)
    record['elapsed_seconds'] = time.monotonic()-started
    record['planned_model_calls'] = 3
    write(folder/(case['id']+'.json'), record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', required=True); p.add_argument('--revision', required=True)
    p.add_argument('--group', choices=['qualification',*GROUPS], required=True)
    p.add_argument('--seed', type=int, default=11)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not str(args.output.resolve()).startswith('/beacon-projects/radfm/'):
        raise RuntimeError('Beacon Slurm and RADFM output required')
    args.output.mkdir(parents=True, exist_ok=False)
    from benchmarks.agent_study.prepare_followup import MODEL, REVISION
    from benchmarks.agent_study.model_transfer import MODEL as M2, REVISION as R2
    if (args.model,args.revision) not in ((MODEL,REVISION),(M2,R2)):
        raise ValueError('model revision not frozen')
    params, provenance = load_parameters()
    source = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    task_cases = cases()
    write(args.output/'protocol.json', dict(model=args.model, revision=args.revision, seed=args.seed,
        group=args.group, arms=GROUPS.get(args.group,[]), tasks=task_cases,
        source_sha256=source, fly_parameters=provenance, response_cap=384, calls_per_episode=3,
        num_task_contracts=8, num_mechanism_families=4, input_seed=101,
        independent_external_holdout=False, trading_returns_measured=False,
        interpretation='Current contract always supplied equally; acquisition 0/1 then new synthetic blocks 2/3. No test-label router fitting.'))
    chat = PackedChat(TransformersChat(args.model,args.revision,max_tokens=384,seed=args.seed))
    result = qualification(chat,params,args.output)
    write(args.output/'qualification.json',result)
    if args.group == 'qualification':
        arm=('finskills','fly','histrim')
        folder=args.output/'smoke'; folder.mkdir()
        memory=ExperienceMemory('fly',params)
        records=[run_episode(chat,c,arm,memory,folder,args.seed+100*c['episode']) for c in task_cases[:2]]
        write(args.output/'smoke-summary.json',dict(attempts=len(records), correct=sum(r['grade']['correct'] for r in records),
            memory_read_on_second=records[1]['memory_read'], note='Quality does not gate expansion; only execution defects do.'))
        return
    arms=list(GROUPS[args.group]); random.Random(args.seed).shuffle(arms)
    summary=[]
    for arm in arms:
        folder=args.output/('-'.join(arm)); folder.mkdir()
        memory=ExperienceMemory(arm[1],params)
        for index,case in enumerate(task_cases):
            record=run_episode(chat,case,arm,memory,folder,args.seed+index*10)
            summary.append(dict(case=case['id'], family=case['family'], phase=case['phase'], arm=arm,
                grade=record['grade'], errors=record['errors'], elapsed_seconds=record['elapsed_seconds'],
                actual_calls=len(record['calls']), prompt_tokens=sum(c['prompt_tokens'] for c in record['calls']),
                completion_tokens=sum(c['completion_tokens'] for c in record['calls'])))
            print(json.dumps(dict(completed=len(summary),planned=len(arms)*len(task_cases),case=case['id'],arm=arm)),flush=True)
    write(args.output/'results.json',dict(planned=len(arms)*len(task_cases), completed=len(summary), rows=summary))


if __name__ == '__main__':
    main()
