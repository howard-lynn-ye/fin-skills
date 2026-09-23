"""Hash-verify completed matched-repair or boundary campaigns without rerunning models."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.parent != Path('/beacon-projects/radfm/wy891'):
        raise ValueError('canonical RADFM campaign required')
    plan = read(root/'campaign.json')
    for name, expected in plan['source_sha256'].items():
        assert digest(root/'source'/name) == expected, name
    sys.path.insert(0, str(root/'source'))
    from benchmarks.agent_study.open_agent_runner import artifact_digest
    from benchmarks.agent_study.summarize_matrix import summarize
    if plan['suite'] == 'boundary-recheck':
        work = root/'jobs/boundary-recheck'
        completion = read(work/'completion.json')
        assert completion['status'] == 'completed'
        path = work/'results/boundary-recheck/results.json'
        result = read(path)
        assert result['complete'] and len(result['rows']) == 6
        assert {(r['seed'],r['case']) for r in result['rows']} == {
            (s,c) for s in (11,23,37) for c in ('zero_positions','clean_news')}
        rows = []
        for row in result['rows']:
            directory = path.parent/f"s{row['seed']}-{row['case']}"
            assert read(directory/'result.json') == row
            assert artifact_digest(directory/'workspace') == row['artifact_sha256']
            prior = Path('/beacon-projects/radfm/wy891/fin-skills-campaign-validity-20260923-v1')
            prior /= f"jobs/audit-validity-s{row['seed']}/results/audit-validity/{row['case']}/result.json"
            assert digest(prior) == row['prior_result_sha256']
            assert read(prior)['artifact_sha256'] == row['artifact_sha256']
            checks = row['checks']
            if row['case'] == 'zero_positions':
                assert checks['survivorship_audit']['status'] == 'UNASSESSABLE'
                assert checks['survivorship_audit']['passed'] is False
            else:
                assert all(c.get('passed') is True for c in checks.values())
            rows.append(dict(seed=row['seed'],case=row['case'],
                checks={k:dict(status=v.get('status'),passed=v.get('passed'),error=v.get('error'))
                        for k,v in checks.items()}, artifact_sha256=row['artifact_sha256']))
        report = dict(complete=True, planned=6, rows=rows, results_sha256=digest(path),
                      completion=completion, scope='execution boundary verification; not new model performance')
        target = root/'boundary-verified-summary.json'
    else:
        assert plan['suite'] in ('matched-repair','model-transfer')
        roots, evidence, rows = [], [], []
        for seed in (11,23,37):
            work = root/f'jobs/matched-repair-s{seed}'
            complete = read(work/'completion.json')
            assert complete['status'] == 'completed'
            assert all(c['returncode'] == 0 for c in complete['commands'])
            receipt = work/'submission-receipt.json'
            if not read(receipt).get('job_id'):
                attempts = list(work.glob('submission-attempt-*/submission-receipt.json'))
                accepted = [p for p in attempts if read(p).get('job_id')]
                assert len(accepted) == 1
                receipt = accepted[0]
            assert str(read(receipt)['job_id']) == complete['slurm_job_id']
            output = work/'results/matched-repair'; roots.append(output)
            protocol = read(output/'protocol.json')
            assert len(protocol['cells']) == 5
            assert digest(root/'source/benchmarks/agent_study/matched_repair.py') == protocol['source_sha256']
            initial = None
            for cell in protocol['cells']:
                directory = output/f"s{seed}-r0-{cell['condition']}"
                inputs = read(directory/'input_receipt.json')
                if initial is None:
                    initial = inputs
                assert inputs == initial, 'initial files differ across arms within seed'
                assert inputs['submission.py'] == protocol['starter_sha256']
                result = read(directory/'result.json')
                submission = read(directory/'submission_receipt.json')
                assert digest(directory/'result.json') == submission['result_sha256']
                assert result['artifact_sha256'] == submission['artifact_sha256'] == artifact_digest(directory/'workspace')
                assert str(submission['slurm_job_id']) == complete['slurm_job_id']
                assert result['market_seed'] == seed and result['condition'] == cell['condition']
                grade_path = directory/'grade.json'
                error_path = directory/'grading_error.json'
                assert grade_path.exists() or error_path.exists(), 'missing grading denominator'
                usage = result['usage']
                checks = result['receipts']
                rows.append(dict(seed=seed,condition=cell['condition'],
                    result_sha256=digest(directory/'result.json'),
                    grade_sha256=digest(grade_path) if grade_path.exists() else None,
                    grading_error=read(error_path) if error_path.exists() else None,
                    actual_responses=len(result['transcript']), tool_calls=result['tool_call_count'],
                    prompt_tokens=sum(u.get('prompt_tokens',0) for u in usage),
                    completion_tokens=sum(u.get('completion_tokens',0) for u in usage),
                    wall_seconds=result['total_wall_including_initial_checks'],
                    checked_seconds=sum(c.get('elapsed_s',0) for c in checks),
                    check_count=len(checks),
                    coverage=read(grade_path).get('universe_coverage') if grade_path.exists() else None))
            assert read(output/'summary.json') == summarize([output])
            evidence.append(dict(seed=seed,job_id=complete['slurm_job_id'],
                elapsed_seconds=complete['elapsed_seconds'], protocol_sha256=digest(output/'protocol.json'),
                summary_sha256=digest(output/'summary.json'), submission_receipt=str(receipt.relative_to(root))))
        combined = summarize(roots)
        assert combined['all_planned_cells_completed']
        assert sum(g['planned'] for g in combined['groups']) == 15
        costs = defaultdict(lambda: defaultdict(float))
        for row in rows:
            for key in ('actual_responses','tool_calls','prompt_tokens','completion_tokens',
                        'wall_seconds','checked_seconds','check_count'):
                costs[row['condition']][key] += row[key]
        report = dict(summary=combined,evidence=evidence,rows=rows,costs=dict(costs),
            limits='Original correctness predicate omits universe coverage; preserved. Three seeds of one authored repair task, not independent tasks. Check times are included in task wall time and must not be added to it.')
        target = root/'matched-repair-verified-summary.json'
    with target.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    compact = dict(path=str(target),sha256=digest(target))
    if 'summary' in report:
        compact.update(groups=[{k:v for k,v in g.items() if k != 'cells'} for g in report['summary']['groups']], costs=report['costs'])
    else:
        compact.update(planned=report['planned'],rows=report['rows'],results_sha256=report['results_sha256'])
    print(json.dumps(compact,allow_nan=False))


if __name__ == '__main__':
    main()
