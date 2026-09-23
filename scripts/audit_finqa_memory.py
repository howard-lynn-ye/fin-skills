"""Audit all planned external-memory episodes without changing inference or original grades."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

root=Path(sys.argv[1]); assert root.resolve()==root and root.parent==Path('/beacon-projects/radfm/wy891')
sys.path.insert(0,str(root/'source'))
import finqa_memory_study as study
from fin_skills.rag import RAGIndex


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


manifest=read(root/'manifest.json')
assert all(sha(root/p)==h for p,h in manifest['sha256'].items())
assert read(root/'completion.json')['status']=='completed'
assert read(root/'software-qualification.json')['passed']
selection=read(study.PREP/'task-selection.json')
assert all(sha(study.PREP/p)==h for p,h in selection['output_sha256'].items())
cases={x['id']:x for p in ('acquisition-inputs.json','evaluation-inputs.json') for x in read(study.PREP/p)}
gold={split:{x['id']:x for x in read(study.UPSTREAM/f'upstream/finqa/dataset/{split}.json')} for split in ('train','test')}
official=study.evaluator(study.UPSTREAM)
acq=read(root/'acquisition-receipt.json'); bank=read(root/'memory-bank.json')
assert acq['bank_sha256']==sha(root/'memory-bank.json') and len(acq['rows'])==16
assert [r['id'] for r in bank['records']]==selection['acquisition_ids']
fixed,learned,params=study.load_circuits(); assert params==bank['parameter_provenance']
totals={phase:dict(planned=16 if phase=='acquisition' else 160,model_call_attempts=0,
    generated_responses=0,input_tokens=0,output_tokens=0,tool_receipts=0,wall_seconds=0.)
    for phase in ('acquisition','evaluation')}
reflection_totals=dict(attempts=0,responses=0,input_tokens=0,output_tokens=0,errors=0)


def verify_episode(row,case,payload,phase):
    assert len(row['calls'])<=6
    expected=dict(case,past_experiences=payload,memory_usage_policy=study.POLICY)
    if row['calls']:
        assert json.loads(row['calls'][0]['messages'][1]['content'])==expected
    total=totals[phase]
    total['model_call_attempts']+=len(row['calls'])
    total['tool_receipts']+=len(row['tool_receipts']); total['wall_seconds']+=row['elapsed_seconds']
    for call in row['calls']:
        from finqa_reuse import template_messages
        messages,mapping=template_messages(call['messages'])
        assert messages==call['template_messages'] and mapping==call['message_mapping']
        if 'response' in call:
            usage=call['response']['usage']; assert usage['completion_tokens']<=512 and usage['prompt_tokens']+512<=32768
            total['generated_responses']+=1; total['input_tokens']+=usage['prompt_tokens']; total['output_tokens']+=usage['completion_tokens']
    for receipt in row['tool_receipts']:
        invalid,value=official.eval_program(official.program_tokenization(receipt['program']),case['table'])
        assert invalid==receipt['invalid'] and value==receipt['result']


for i,item in enumerate(acq['rows']):
    path=root/'acquisition'/f'{i:02d}.json'; feedback_path=root/'feedback'/f'{i:02d}.json'
    refl_path=root/'reflections'/f'{i:02d}.json'
    assert sha(path)==item['episode_sha256']==read(root/'acquisition'/f'{i:02d}-receipt.json')['episode_sha256']
    assert sha(feedback_path)==item['feedback_sha256'] and sha(refl_path)==item['reflection_sha256']
    row=read(path); feedback=read(feedback_path); reflection=read(refl_path)
    assert row['id']==item['id']==selection['acquisition_ids'][i]
    grade=study.grade_row(row,gold['train'][row['id']],official)
    assert all(feedback[k]==v for k,v in grade.items())
    verify_episode(row,cases[row['id']],dict(records=[],circuit=None),'acquisition')
    program=study.program_of(row); record=bank['records'][i]
    assert record['program']==program and record['cue']==study.cue_of(program)
    assert record['feedback']=={k:feedback[k] for k in ('execution_correct','format_error','scorer_error')}
    assert record['reflection']=='\n'.join(reflection['plans']) and record['reflection_error']==reflection['error']
    assert record['available_after_acquisition']==i
    study.condition(learned,program,grade['execution_correct'])
    if i: assert acq['rows'][i-1]['transition']['after']==item['transition']['before']
    if reflection['error'] is None:
        assert len(reflection['calls'])==int(not grade['execution_correct'])
    else: reflection_totals['errors']+=1
    assert len(reflection['calls'])<=1
    reflection_totals['attempts']+=len(reflection['calls'])
    for call in reflection['calls']:
        assert call['messages']==[dict(role='user',content=study.REFLECTION_PREFIX+call['original_prompt'])]
        if 'response' in call:
            usage=call['response']['usage']; assert usage['completion_tokens']<=512
            reflection_totals['responses']+=1; reflection_totals['input_tokens']+=usage['prompt_tokens']; reflection_totals['output_tokens']+=usage['completion_tokens']
for brain,key in ((fixed,'frozen'),(learned,'learned')):
    replay=study.state(brain)
    assert replay['clock']==bank[key]['clock']
    for field in ('weights','cue_scores'):
        assert np.allclose(replay[field],bank[key][field],rtol=1e-10,atol=1e-12)

records=bank['records']; by_id={r['id']:r for r in records}
index=RAGIndex([dict(id=r['id'],text=r['question']+'\n'+r['reflection']) for r in records],chunk_size=100000,overlap=0)
state_hash=study.digest(dict(records=records,frozen=bank['frozen'],learned=bank['learned']))
receipt=read(root/'evaluation-receipt.json'); scores=read(root/'evaluation-scores.json')
assert scores['evaluation_receipt_sha256']==sha(root/'evaluation-receipt.json')
assert receipt['bank_sha256']==sha(root/'memory-bank.json') and receipt['planned']==160
expected={(i,a) for i in selection['evaluation_ids'] for a in study.ARMS}
assert len(receipt['rows'])==160 and {(r['id'],r['arm']) for r in receipt['rows']}==expected
saved={(r['id'],r['arm']):r for r in scores['rows']}; assert set(saved)==expected
verified=[]
for item in receipt['rows']:
    p=root/item['file']; assert sha(p)==item['sha256']; row=read(p)
    assert row['id']==item['id'] and row['arm']==item['arm']
    arm=row['arm']; case=cases[row['id']]
    if arm=='none': chosen=[]
    elif arm=='reflexion_recent': chosen=records[-3:]
    else: chosen=[by_id[h['document_id']] for h in index.search(case['question'],top_k=3)]
    payload=dict(records=chosen,circuit=None)
    if arm in ('fly_frozen','fly_learned'):
        state=bank['frozen' if arm=='fly_frozen' else 'learned']
        payload['circuit']=dict(scores=dict(zip(study.CUES,state['cue_scores'])),
            meaning='Uncalibrated past-only method-family scores, not correctness probabilities or expected returns.')
    assert row['memory_payload']==payload
    assert row['memory_read']==dict(state_sha256=state_hash,experience_ids=[r['id'] for r in chosen],payload_sha256=study.digest(payload))
    assert row['bank_sha256']==sha(root/'memory-bank.json')
    verify_episode(row,case,payload,'evaluation')
    grade=study.grade_row(row,gold['test'][row['id']],official)
    assert all(saved[(row['id'],arm)][k]==v for k,v in grade.items())
    verified.append(dict(id=row['id'],arm=arm,**grade))
aggregate={arm:dict(planned=32,execution_correct=sum(r['execution_correct'] for r in verified if r['arm']==arm),
    program_correct=sum(r['program_correct'] for r in verified if r['arm']==arm),
    format_errors=sum(r['format_error'] is not None for r in verified if r['arm']==arm),
    scorer_errors=sum(r['scorer_error'] is not None for r in verified if r['arm']==arm)) for arm in study.ARMS}
assert aggregate==scores['aggregate']
output=root/'memory-verified-summary.json'
with output.open('x') as f:
    json.dump(dict(verified=True,aggregate=aggregate,costs=totals,reflection_costs=reflection_totals,
        evaluation_receipt_sha256=sha(root/'evaluation-receipt.json'),bank_sha256=sha(root/'memory-bank.json'),
        score_sha256=sha(root/'evaluation-scores.json'),source_manifest_sha256=sha(root/'manifest.json'),
        audit_source_sha256=sha(Path(__file__)),
        limits='32 external public evaluation tasks, five arms; shared acquisition once; circuit replay tolerance applies only to read-only floats, original grade booleans exact.'),f,indent=2)
print(json.dumps(dict(sha256=sha(output),aggregate=aggregate)))
