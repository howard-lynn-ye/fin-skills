"""Read-only verification of a completed FinQA qualification; append one exclusive audit."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root=Path(sys.argv[1]); assert root.resolve()==root and root.parent==Path('/beacon-projects/radfm/wy891')
upstream=root.parent/'fin-skills-campaign-reuse-bootstrap-20260923-v2'
manifest=json.loads((root/'manifest.json').read_text())
assert all(sha(root/p)==h for p,h in manifest['sha256'].items())
completion=json.loads((root/'completion.json').read_text())
assert completion['status']=='completed' and all(c['returncode']==0 for c in completion['commands'])
spec=importlib.util.spec_from_file_location('adapter_audit',root/'source/finqa_reuse.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
official=module.evaluator(upstream)
gold={r['id']:r for r in json.loads((upstream/'upstream/finqa/dataset/dev.json').read_text())}
inputs=json.loads((upstream/'public-inputs/finqa-dev.json').read_text())
selected=sorted(inputs,key=lambda x:hashlib.sha256(('finqa-dev-calibration-v1|'+x['id']).encode()).hexdigest())[:4]
families={}
for family in ('qwen','mistral'):
    d=root/'results'/family
    protocol=json.loads((d/'protocol.json').read_text())
    assert protocol['selected_ids']==[r['id'] for r in selected]
    assert protocol['input_sha256']==sha(upstream/'public-inputs/finqa-dev.json')
    assert protocol['adapter_sha256']==sha(root/'source/finqa_reuse.py')
    assert json.loads((d/'software-qualification.json').read_text())['passed']
    receipt=json.loads((d/'inference-receipt.json').read_text())
    scores=json.loads((d/'scores.json').read_text())
    assert scores['inference_receipt_sha256']==sha(d/'inference-receipt.json')
    assert receipt['planned']==receipt['completed']==scores['planned']==scores['completed']==4
    total=dict(planned=4,execution_correct=0,program_correct=0,raw_call_attempts=0,
               generated_responses=0,input_tokens=0,output_tokens=0,tool_receipts=0,episode_seconds=0.)
    outcomes=[]
    for i,item in enumerate(receipt['records']):
        p=d/item['file']; assert sha(p)==item['sha256']
        row=json.loads(p.read_text()); target=gold[row['id']]
        assert row['id']==selected[i]['id']==scores['rows'][i]['id']
        assert len(row['calls'])<=6 and scores['rows'][i]['actual_calls']==len(row['calls'])
        assert scores['rows'][i]['tool_calls']==len(row['tool_receipts'])
        format_error=scorer_error=None; execution_correct=program_correct=False
        try:
            parsed=json.loads(row['final'])
            assert isinstance(parsed,dict) and set(parsed)=={'program'} and isinstance(parsed['program'],str)
            tokens=official.program_tokenization(parsed['program'])
            if len(tokens)>81: raise ValueError('Program exceeds operation budget')
        except Exception as exc: format_error=type(exc).__name__
        else:
            try:
                invalid,value=official.eval_program(tokens,target['table'])
                execution_correct=invalid==0 and value==target['qa']['exe_ans']
                program_correct=bool(official.equal_program(official.program_tokenization(target['qa']['program']),tokens))
            except Exception as exc: scorer_error=dict(type=type(exc).__name__,message=str(exc))
        expected=dict(execution_correct=execution_correct,program_correct=program_correct,
                      format_error=format_error,scorer_error=scorer_error)
        assert all(scores['rows'][i][k]==v for k,v in expected.items())
        for call in row['calls']:
            if 'template_messages' in call:
                messages,mapping=module.template_messages(call['messages'])
                assert call['template_messages']==messages and call['message_mapping']==mapping
            if 'response' in call:
                usage=call['response']['usage']; assert usage['completion_tokens']<=512
                total['generated_responses']+=1
                total['input_tokens']+=usage['prompt_tokens']; total['output_tokens']+=usage['completion_tokens']
        for tool in row['tool_receipts']:
            invalid,value=official.eval_program(official.program_tokenization(tool['program']),target['table'])
            assert tool['invalid']==invalid and tool['result']==value
        total['execution_correct']+=execution_correct; total['program_correct']+=program_correct
        total['raw_call_attempts']+=len(row['calls']); total['tool_receipts']+=len(row['tool_receipts'])
        total['episode_seconds']+=row['elapsed_seconds']
        outcomes.append(dict(id=row['id'],error=row['error'],final_present=row['final'] is not None,
            backend_errors=[dict(type=c.get('error_type'),message=c['error']) for c in row['calls'] if 'error' in c],**expected))
    assert total['execution_correct']==scores['execution_correct'] and total['program_correct']==scores['program_correct']
    families[family]=dict(totals=total,outcomes=outcomes,scores_sha256=sha(d/'scores.json'),
        inference_receipt_sha256=sha(d/'inference-receipt.json'))
result=dict(verified=True,families=families,completion_sha256=sha(root/'completion.json'),
    source_manifest_sha256=sha(root/'manifest.json'),audit_source_sha256=sha(Path(__file__)),
    limits='Eight development attempts, not an independent test score. Tool counts are returned executor receipts, not all attempted tool dispatches.')
target=root/'qualification-verified.json'
with target.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(dict(summary_sha256=sha(target),families={k:v['totals'] for k,v in families.items()})))
