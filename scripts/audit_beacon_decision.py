"""Audit the complete frozen decision matrix on Beacon without rerunning inference."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(sys.argv[1])
    assert root in [Path('/beacon-projects/radfm/wy891')/name for name in (
        'fin-skills-campaign-decision-loop-20260923-v1',
        'fin-skills-campaign-decision-replication-20260923-v1')]
    assert root.resolve() == root
    plan = read(root/'campaign.json')
    # Completion is a prerequisite: never summarize only the fastest jobs.
    for job in plan['jobs']:
        completion = read(root/'jobs'/job['id']/'completion.json')
        assert completion['status'] == 'completed', job['id']
        assert all(c['returncode'] == 0 for c in completion['commands'])
    for name, expected in plan['source_sha256'].items():
        assert sha(root/'source'/name) == expected, name
    sys.path.insert(0, str(root/'source'))
    from benchmarks.agent_study.decision_tasks import cases, clean, digest, execute, grade
    from benchmarks.agent_study.decision_loop import GROUPS, distractors
    planned_total = sum(32*len(GROUPS[j['group']]) for j in plan['jobs'])
    planned_groups = sum(2*len(GROUPS[j['group']]) for j in plan['jobs'])
    task_cases = cases(101)
    groups, evidence, transitions, numerical_roundoff = {}, [], [], []

    def check_numbers(saved, recomputed, path, field):
        assert len(saved) == len(recomputed), (path, field)
        assert all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-14)
                   for a,b in zip(saved,recomputed)), (path, field)
        if saved != recomputed:
            numerical_roundoff.append(dict(path=str(path.relative_to(root)),field=field,
                saved=saved,recomputed=recomputed,
                max_absolute_difference=max(abs(a-b) for a,b in zip(saved,recomputed))))
    total = 0
    for job in plan['jobs']:
        work = root/'jobs'/job['id']
        output = work/'results/decision-loop/study'
        receipt, complete = read(work/'submission-receipt.json'), read(work/'completion.json')
        assert receipt['status'] == 'submitted'
        assert receipt['job_id'] == complete['slurm_job_id']
        protocol, result = read(output/'protocol.json'), read(output/'results.json')
        assert protocol['model'] == job['model'] and protocol['revision'] == job['revision']
        assert protocol['seed'] == job['seed'] and protocol['input_seed'] == 101
        assert protocol['tasks'] == task_cases
        assert protocol['arms'] == clean(GROUPS[job['group']])
        assert protocol['response_cap'] == 384 and protocol['calls_per_episode'] == 3
        for name, expected in protocol['source_sha256'].items():
            assert sha(root/'source/benchmarks/agent_study'/name) == expected
        for name, expected in protocol['fly_parameters'].items():
            assert sha(Path(name)) == expected
        qualification = read(output/'qualification.json')
        assert qualification['passed'] and qualification['numeric_cases'] == 32
        assert all(c['passed'] for c in read(output/'reference-calibration.json'))
        expected_count = 32*len(protocol['arms'])
        assert result['planned'] == result['completed'] == len(result['rows']) == expected_count
        summaries = {(tuple(x['arm']), x['case']):x for x in result['rows']}
        assert len(summaries) == expected_count
        record_hashes = {}
        for arm in protocol['arms']:
            style, memory_mode, context = arm
            folder = output/'-'.join(arm)
            assert {p.stem for p in folder.glob('*.json')} == {c['id'] for c in task_cases}
            seen, previous_after, previous_client = [], None, {}
            for case in task_cases:
                path = folder/(case['id']+'.json'); row = read(path)
                record_hashes[str(path.relative_to(output))] = sha(path)
                assert row['case'] == case['id'] and row['arm'] == arm
                assert row['input_sha256'] == digest(case['data']) == case['data_sha256']
                assert row['phase'] == case['phase'] and row['family'] == case['family']
                recomputed_grade = clean(grade(case,row['action'],row['final'],row['receipt']))
                assert {k:v for k,v in row['grade'].items() if k != 'expected'} == {
                    k:v for k,v in recomputed_grade.items() if k != 'expected'}, path
                check_numbers(row['grade']['expected'],recomputed_grade['expected'],path,'expected')
                if row['receipt'] is not None:
                    saved_receipt = row['receipt']
                    assert saved_receipt['sha256'] == digest({k:v for k,v in saved_receipt.items() if k != 'sha256'})
                    recomputed_receipt = clean(execute(case,row['action']['method'],row['action']['parameters']))
                    assert {k:v for k,v in saved_receipt.items() if k not in ('values','sha256')} == {
                        k:v for k,v in recomputed_receipt.items() if k not in ('values','sha256')}, path
                    check_numbers(saved_receipt['values'],recomputed_receipt['values'],path,'tool_values')
                m, update = row['memory_read'], row['memory_update']
                eligible = [c['id'] for c in seen if c['client'] == case['client']
                    and c['feedback_available_day'] < case['decision_day']][-3:]
                assert m['experience_ids'] == eligible
                assert m['state_sha256'] == update['before']
                if previous_after is not None:
                    assert m['state_sha256'] == previous_after
                expected_update = memory_mode != 'none' and not (
                    memory_mode == 'frozen' and case['phase'] == 'later_evaluation')
                assert update['updated'] == expected_update
                if expected_update:
                    seen.append(case)
                else:
                    assert update['after'] == update['before']
                previous_after = update['after']
                item_ids = [x['id'] for x in distractors(case)+m['actual_items']]
                assert set(eligible) <= set(item_ids)
                if memory_mode in ('none','reflection','retrieval'):
                    assert m['scores'] is None
                assert row['planned_model_calls'] == 3 and len(row['calls']) <= 3
                key = '|'.join([job['model'], str(job['seed']), *arm, case['phase']])
                g = groups.setdefault(key, dict(model=job['model'], seed=job['seed'], arm=arm, phase=case['phase'],
                    planned=16, completed=0, outcomes=Counter(), errors=Counter(), costs=defaultdict(float),
                    per_family={}, peaks=defaultdict(int), retention=[]))
                g['completed'] += 1
                for metric in ('correct','method_correct','parameters_correct','numeric_correct','used_output'):
                    g['outcomes'][metric] += int(row['grade'][metric])
                family = g['per_family'].setdefault(case['family'],dict(completed=0,correct=0))
                family['completed'] += 1; family['correct'] += int(row['grade']['correct'])
                for error in row['errors']:
                    g['errors'][error['stage']+':'+error['type']+':'+error['message']] += 1
                g['costs']['episode_wall_seconds'] += row['elapsed_seconds']
                g['costs']['tool_seconds'] += row.get('tool_seconds',0)
                g['costs']['tool_receipts'] += int(row['receipt'] is not None)
                g['costs']['returned_model_calls'] += len(row['calls'])
                g['costs']['missing_model_call_records'] += 3-len(row['calls'])
                for call_index, call in enumerate(row['calls']):
                    assert 0 < call['completion_tokens'] <= 384
                    assert call['prompt_tokens']+384 <= 8192
                    for metric in ('prompt_tokens','completion_tokens','elapsed_seconds','prefill_seconds'):
                        g['costs'][metric] += call[metric]
                    for metric in ('peak_allocated_bytes','peak_reserved_bytes','prefill_kv_bytes'):
                        g['peaks'][metric] = max(g['peaks'][metric],call[metric])
                    for selection in call['selection']:
                        assert selection['history_tokens'] <= selection['token_budget']
                        assert set(selection['retained_original_positions']) <= set(selection['previous_positions'])
                        retained = [item_ids[i] for i in selection['retained_items']]
                        g['retention'].append(dict(case=case['id'], call_index=call_index,
                            layer=selection['layer'], retained=retained,
                            supplied_experiences=eligible, history_tokens=selection['history_tokens'],
                            token_budget=selection['token_budget'],kv_bytes=call['prefill_kv_bytes']))
                prior = previous_client.get(case['client'])
                transitions.append(dict(model=job['model'],seed=job['seed'],arm=arm,case=case['id'],phase=case['phase'],
                    experience_ids=eligible,previous_case=prior['case'] if prior else None,
                    previous_method=prior['action'].get('method') if prior else None,
                    method=row['action'].get('method'),correct=row['grade']['correct'],
                    state_before=m['state_sha256'],state_after=update['after']))
                previous_client[case['client']] = row
                summary = dict(case=case['id'], family=case['family'],phase=case['phase'],arm=arm,
                    grade=row['grade'],errors=row['errors'],elapsed_seconds=row['elapsed_seconds'],
                    actual_calls=len(row['calls']),prompt_tokens=sum(c['prompt_tokens'] for c in row['calls']),
                    completion_tokens=sum(c['completion_tokens'] for c in row['calls']))
                assert summaries[(tuple(arm),case['id'])] == summary
                total += 1
        evidence.append(dict(job=job['id'],seed=job['seed'],job_id=receipt['job_id'],elapsed_seconds=complete['elapsed_seconds'],
            hashes={p.name:sha(p) for p in output.glob('*.json')},record_hashes=record_hashes,
            calibration_seconds=read(output/'routers.json')['elapsed_seconds']))
    assert total == planned_total and len(groups) == planned_groups
    assert all(g['completed'] == g['planned'] for g in groups.values())
    report = dict(complete=True,planned=planned_total,completed=total,groups=list(groups.values()),
        evidence=evidence,transitions=transitions,auditor_sha256=sha(Path(__file__)),
        numerical_roundoff=numerical_roundoff,
        audit_numeric_tolerance=dict(rtol=1e-12,atol=1e-14,
            scope='Recomputed numerical values only; all original grade booleans exact, scoring unchanged.'),
        limits=['Authored synthetic development tasks; repeated inference seeds are not independent tasks; no independent holdout.',
            'Acquisition and later outcomes separate. Four mechanisms are the task units.',
            'Same token caps do not ensure equal retained tokens or actual compute.',
            'Model-call costs omit failed calls without returned traces; count those explicitly.',
            'Episode wall includes tool and model time; summed job times are not calendar duration.',
            'Recorded experience supply and choice changes do not prove causal reasoning.'])
    target = root/'decision-verified-summary.json'
    with target.open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(dict(path=str(target),sha256=sha(target),planned=planned_total,completed=total)))


if __name__ == '__main__':
    main()
