"""Describe context costs and retained experience from already audited pilot traces."""
import hashlib
import json
from pathlib import Path
import statistics
import sys


def main():
    root = Path(sys.argv[1])
    assert root == Path('/beacon-projects/radfm/wy891/fin-skills-campaign-decision-loop-20260923-v1')
    summary = json.loads((root/'decision-verified-summary.json').read_text())
    assert summary['complete'] and summary['completed'] == 704
    evidence = {e['job']:e for e in summary['evidence']}
    sys.path.insert(0,str(root/'source'))
    from benchmarks.agent_study.decision_tasks import cases
    from benchmarks.agent_study.decision_loop import distractors
    task_cases = {c['id']:c for c in cases(101)}
    rows = []
    for group in summary['groups']:
        arm = group['arm']
        if group['phase'] != 'later_evaluation' or arm[0] != 'finskills' or arm[1] not in ('none','fly'):
            continue
        model = 'qwen' if 'Qwen' in group['model'] else 'mistral'
        name = 'tools' if arm == ['finskills','none','full'] else 'memory' if arm[2] == 'full' else 'context'
        job = f'decision-{model}-{name}'
        output = root/f'jobs/{job}/results/decision-loop/study'
        calls, histories = [], []
        offered = kept = 0
        for path in (output/'-'.join(arm)).glob('*.json'):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == evidence[job]['record_hashes'][str(path.relative_to(output))]
            row = json.loads(path.read_text())
            if row['phase'] != 'later_evaluation':
                continue
            assert len(row['calls']) == 3
            ids = [i['id'] for i in distractors(task_cases[row['case']])+row['memory_read']['actual_items']]
            experiences = set(row['memory_read']['experience_ids'])
            for call in row['calls'][:2]:
                calls.append(call); offered += len(experiences)
                if call['selection']:
                    last = call['selection'][-1]
                    retained = {ids[i] for i in last['retained_items']}
                    kept += len(experiences & retained)
                    histories.append(last['history_tokens'])
                else:
                    kept += len(experiences)
        assert len(calls) == 32
        rows.append(dict(model=model,arm=arm,calls=len(calls),
            mean_prompt_tokens=statistics.mean(c['prompt_tokens'] for c in calls),
            mean_output_tokens=statistics.mean(c['completion_tokens'] for c in calls),
            mean_model_seconds=statistics.mean(c['elapsed_seconds'] for c in calls),
            mean_prefill_seconds=statistics.mean(c['prefill_seconds'] for c in calls),
            mean_kv_mib=statistics.mean(c['prefill_kv_bytes'] for c in calls)/2**20,
            peak_allocated_gib=max(c['peak_allocated_bytes'] for c in calls)/2**30,
            mean_final_history_tokens=statistics.mean(histories) if histories else None,
            offered_experience_items=offered,retained_experience_items=kept))
    report = dict(rows=rows,source_summary_sha256=hashlib.sha256((root/'decision-verified-summary.json').read_bytes()).hexdigest(),
        scope='Later-episode selection and final-answer calls only; excludes reflection and calibration.',
        limitations=['Different job allocations and generated lengths confound pilot latency comparisons.',
            'Retained experience counts are item occurrences across calls, not independent tasks.',
            'Removed final-depth tokens can still have influenced protected states at earlier layers.'])
    target = root/'decision-context-description.json'
    with target.open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
