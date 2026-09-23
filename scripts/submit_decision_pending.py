"""Submit untouched jobs from the frozen decision matrix within Beacon capacity.

Run this transport-side helper on Beacon, outside the immutable source tree.
Existing accepted receipts are skipped; any other existing job directory stops it.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(sys.argv[1])
    assert root == Path('/beacon-projects/radfm/wy891/fin-skills-campaign-decision-loop-20260923-v1')
    assert root.resolve() == root
    sys.path.insert(0, str(root/'source'))
    from scripts.beacon_campaign import validate_bundle, scheduler_command, write_once
    from scripts.submit_model_followup import _dispatch
    plan = validate_bundle(root)
    prior = root.parent/'fin-skills-campaign-decision-qualification-serial-20260923-v1'
    qualification_plan = read(prior/'campaign.json')
    assert plan['source_sha256'] == qualification_plan['source_sha256']
    for name, expected in plan['source_sha256'].items():
        assert (root/'source'/name).resolve() == root/'source'/name
        assert sha(root/'source'/name) == sha(prior/'source'/name) == expected
    work = prior/'jobs/decision-qualification'
    complete = read(work/'completion.json')
    assert complete['status'] == 'completed' and complete['slurm_job_id'] == '1651173'
    assert all(c['returncode'] == 0 for c in complete['commands'])
    evidence = {}
    for family in ('qwen', 'mistral'):
        output = work/'results/decision-loop'/family
        q = read(output/'qualification.json')
        assert q['passed'] and q['numeric_cases'] == 32
        physical = read(output/'physical-cache-check.json')
        assert all(physical[k] for k in ('compact', 'protected_ok', 'nested_ok'))
        assert read(output/'smoke-summary.json')['attempts'] == 2
        evidence[family] = dict(qualification=q, hashes={
            str(p.relative_to(output)):sha(p) for p in output.rglob('*.json')})
    verification = dict(job_id='1651173', completion_sha256=sha(work/'completion.json'),
        source_files=len(plan['source_sha256']), source_hashes_identical=True, models=evidence,
        scope='Engineering qualification; smoke quality and probe timing are not study outcomes.')
    audit = root/'analysis/qualification-verification.json'
    if audit.exists():
        assert read(audit) == verification
    else:
        write_once(audit, verification)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    batch = root/'submission-batches'/stamp
    batch.mkdir(parents=True, exist_ok=False)
    write_once(batch/'started.json', dict(utc=stamp, dispatcher_sha256=sha(Path(__file__))))
    observations, receipts, pending = [], [], []
    ordered = sorted(plan['jobs'], key=lambda j: (
        ('tools','memory','context').index(j['group']), j['id']))
    for job in ordered:
        destination = root/'jobs'/job['id']
        if destination.exists():
            receipt_path = destination/'submission-receipt.json'
            assert receipt_path.is_file(), 'ambiguous existing job; inspect before continuing'
            receipt = read(receipt_path)
            assert receipt['status'] == 'submitted', 'existing unsuccessful/uncertain attempt requires manual audit'
            receipts.append(dict(id=job['id'], **receipt))
            continue
        queue = subprocess.run(['squeue','-u','wy891','-p','beacon','-r','-h','-o',
            '%i|%a|%j|%T|%R'], capture_output=True, text=True, timeout=30, check=True).stdout
        rows = [line.split('|') for line in queue.splitlines() if line.strip()]
        relevant = [r for r in rows if r[1] == 'angliece']
        assert not any(r[2] == 'fin-'+job['id'] for r in rows), 'scheduler job without receipt'
        history = subprocess.run(['sacct','-X','-n','-P','-S','2026-09-22T00:00:00',
            '-u','wy891','--name=fin-'+job['id'],'-o','JobIDRaw,JobName%100,State'],
            capture_output=True, text=True, timeout=30, check=True).stdout
        assert not history.strip(), 'accounting record without local receipt'
        observations.append(dict(job=job['id'], queue=queue, history=history,
            association_jobs=len(relevant), capacity=11))
        if len(relevant) >= 11:
            pending.append(job['id'])
            continue
        destination.mkdir(parents=True, exist_ok=False)
        (destination/'logs').mkdir()
        runtime = Path(plan['runtime_root'])/'env/bin/python'
        assert runtime.is_file()
        command = [str(runtime), str(root/'source/scripts/beacon_campaign_worker.py'), str(root), job['id']]
        shell = '#!/usr/bin/env bash\nset -euo pipefail\nunset HISTFILE\n'
        shell += 'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1\n'
        shell += 'exec '+shlex.join(command)+'\n'
        (destination/'job.sh').write_text(shell)
        command = scheduler_command(plan, job)
        assert not any(x.startswith(('--nodelist','--exclude')) for x in command)
        receipt = _dispatch(destination, command)
        receipts.append(dict(id=job['id'], **receipt))
        if receipt['status'] != 'submitted':
            break
    result = dict(jobs=receipts, not_submitted=pending, observations=observations,
        qualification_verification_sha256=sha(audit), batch=str(batch.relative_to(root)))
    write_once(batch/'receipt.json', result)
    if len(receipts) == len(plan['jobs']) and all(r['status'] == 'submitted' for r in receipts):
        target = root/'campaign-submission.json'
        if not target.exists():
            write_once(target, dict(jobs=receipts))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
