"""Qualify unchanged upstream Reflexion update flow and freeze disjoint FinQA task IDs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types
import urllib.request

UPSTREAM=Path('/beacon-projects/radfm/wy891/fin-skills-campaign-reuse-bootstrap-20260923-v2')


def write(path,value):
    with path.open('x') as out:
        json.dump(value,out,indent=2,allow_nan=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def order(tag,value):
    return hashlib.sha256((tag+'|'+value).encode()).hexdigest()


def run(root):
    assert os.environ.get('SLURM_JOB_ID') and root.resolve()==root
    assert root.parent==Path('/beacon-projects/radfm/wy891')
    frozen=json.loads((root/'manifest.json').read_text())
    assert all(sha(root/p)==h for p,h in frozen['sha256'].items())
    upstream=json.loads((UPSTREAM/'upstream-manifest.json').read_text())
    ref=upstream['sources']['reflexion']
    assert ref['revision']=='218cf0ef1df84b05ce379dd4a8e47f17766733a0'
    dest=root/'reflexion'; dest.mkdir()
    for name in ('generate_reflections.py','env_history.py'):
        key='alfworld_runs/'+name; source=UPSTREAM/'upstream/reflexion'/key
        assert sha(source)==ref['files'][key]['sha256']
        (dest/name).write_bytes(source.read_bytes())
    url=f"https://raw.githubusercontent.com/noahshinn/reflexion/{ref['revision']}/alfworld_runs/reflexion_few_shot_examples.txt"
    with urllib.request.urlopen(url,timeout=60) as response:
        asset=response.read(200001)
    assert len(asset)<=200000
    (dest/'reflexion_few_shot_examples.txt').write_bytes(asset)
    write(root/'upstream-receipt.json',dict(revision=ref['revision'],asset_url=url,
        files={p.name:sha(p) for p in dest.iterdir()}))

    # Replace only the network completion dependency; execute the author's original update_memory.
    queries=[]
    shim=types.ModuleType('utils')
    def completion(query):
        queries.append(query)
        return 'Plan: Verify input units before computing the difference.'
    shim.get_completion=completion
    original=sys.modules.get('utils'); sys.modules['utils']=shim
    cwd=os.getcwd()
    try:
        os.chdir(dest)
        spec=importlib.util.spec_from_file_location('upstream_reflexion',dest/'generate_reflections.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        log=root/'fixture-trial.log'
        log.write_text('#####\n\n#####'.join('Here is the task:\n'+x for x in
            ['Compute a change. Failed: confused units.','Solved a change.','Skipped task.']))
        states=[dict(is_success=False,skip=False,memory=['old0','old1','old2','old3']),
                dict(is_success=True,skip=False,memory=[]),dict(is_success=False,skip=True,memory=[])]
        result=module.update_memory(str(log),states)
        assert len(queries)==1 and 'old0' not in queries[0] and all(x in queries[0] for x in ['old1','old2','old3'])
        assert len(result[0]['memory'])==5 and not result[1]['memory'] and not result[2]['memory']
        spec=importlib.util.spec_from_file_location('upstream_env_history',dest/'env_history.py')
        history_module=importlib.util.module_from_spec(spec); spec.loader.exec_module(history_module)
        history=history_module.EnvironmentHistory('Solve financial tasks.','Next report.',result[0]['memory'][-1:],history=[])
        separate=history_module.EnvironmentHistory('Base','Other task',[],history=[])
        history.add('action','compute'); history.add('observation','42')
        assert result[0]['memory'][-1] in str(history) and 'compute' not in str(separate)
        write(root/'reflexion-software-qualification.json',dict(passed=True,
            original_update_memory_executed=True,original_environment_history_executed=True,
            scripted_completion_calls=len(queries),prompt_sha256=hashlib.sha256(queries[0].encode()).hexdigest(),
            unsuccessful_only=True,last_three_plans=True,fresh_history_lists=True,
            inference=False,limits='Software fixture; original ALFWorld same-task reflection is not yet a financial transfer result.'))
    finally:
        os.chdir(cwd)
        if original is None: del sys.modules['utils']
        else: sys.modules['utils']=original

    inputs={split:json.loads((UPSTREAM/f'public-inputs/finqa-{split}.json').read_text()) for split in ('train','dev','test')}
    calibration=sorted(inputs['dev'],key=lambda x:order('finqa-dev-calibration-v1',x['id']))[:4]
    excluded={x['id'].split('/')[0] for x in calibration}
    def grouped(rows):
        companies={}
        for row in rows:
            company,year=row['id'].split('/')[:2]
            companies.setdefault(company,{}).setdefault(company+'/'+year,[]).append(row)
        return companies
    train,test=grouped(inputs['train']),grouped(inputs['test'])
    eligible=[c for c,reports in test.items() if c not in excluded and len(reports)>=2]
    evaluation_companies=sorted(eligible,key=lambda c:order('finqa-transfer-eval-company-v1',c))[:16]
    eligible_train=[c for c,reports in train.items() if c not in excluded|set(evaluation_companies) and len(reports)>=2]
    acquisition_companies=sorted(eligible_train,key=lambda c:order('finqa-transfer-acquire-company-v1',c))[:8]
    assert len(evaluation_companies)==16 and len(acquisition_companies)==8, 'Insufficient companies; do not silently shrink frozen design'
    def select(groups,companies,tag):
        chosen=[]
        for c in companies:
            reports=sorted(groups[c],key=lambda r:order(tag+'-report',r))[:2]
            for report in reports:
                chosen.append(min(groups[c][report],key=lambda x:order(tag+'-question',x['id'])))
        return chosen
    acquisition=select(train,acquisition_companies,'finqa-transfer-acquire-v1')
    evaluation=select(test,evaluation_companies,'finqa-transfer-eval-v1')
    assert len(acquisition)==16 and len(evaluation)==32
    assert not set(acquisition_companies)&set(evaluation_companies)
    assert len({tuple(x['id'].split('/')[:2]) for x in acquisition+evaluation})==48
    write(root/'acquisition-inputs.json',acquisition)
    write(root/'evaluation-inputs.json',evaluation)
    write(root/'task-selection.json',dict(acquisition_ids=[x['id'] for x in acquisition],
        evaluation_ids=[x['id'] for x in evaluation],acquisition_companies=acquisition_companies,
        evaluation_companies=evaluation_companies,excluded_calibration_companies=sorted(excluded),
        input_sha256={split:sha(UPSTREAM/f'public-inputs/finqa-{split}.json') for split in inputs},
        output_sha256={p.name:sha(p) for p in [root/'acquisition-inputs.json',root/'evaluation-inputs.json']},
        selection='Prospective SHA order; 16 test companies x2 reports, 8 disjoint train companies x2 reports, 1 question/report',
        gold_read=False,inference=False,memory_arm_protocol_not_yet_frozen=True,
        limitations=['Public historical FinQA; model pretraining exposure unknown.',
            'Company/report separation does not remove similar question templates.',
            'Source task selection only; no model scores or memory effectiveness yet.']))


if __name__=='__main__':
    root=Path(sys.argv[1]); start=time.monotonic()
    try: run(root)
    except Exception as exc:
        write(root/'completion.json',dict(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
        raise
    write(root/'completion.json',dict(status='completed',elapsed_seconds=time.monotonic()-start,inference=False))
