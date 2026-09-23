"""Actual upstream Reflexion + LlamaIndex on frozen, company-disjoint FinQA tasks."""
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import traceback
import types

from finqa_reuse import evaluator,episode,software_qualification,sha,write

BASE=Path('/beacon-projects/radfm/wy891')
UPSTREAM=BASE/'fin-skills-campaign-reuse-bootstrap-20260923-v2'
PREP=BASE/'fin-skills-campaign-memory-reuse-prepare-20260923-v1'
ARMS=('none','reflexion_recent','reflexion_retrieval','fly_frozen','fly_learned')
CUES=('scalar_arithmetic','table_operation')
POLICY='Past experiences are fallible method guidance from other reports. Compute this answer only from the current report; never copy old numeric answers.'
REFLECTION_PREFIX='Financial transfer adaptation: produce reusable procedure advice from this experience; do not treat report-specific numbers as answers to future reports.\n\n'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def program_of(row):
    try:
        value=json.loads(row['final'])['program']
        if isinstance(value,str): return value
    except (TypeError,ValueError,KeyError): pass
    return row['tool_receipts'][-1]['program'] if row['tool_receipts'] else ''


def cue_of(program):
    if re.search(r'\btable_(?:sum|average|max|min)\s*\(',program): return 1
    if re.search(r'\b(?:add|subtract|multiply|divide|exp|greater)\s*\(',program): return 0
    return None


def grade_row(row,target,official):
    result=dict(execution_correct=False,program_correct=False,format_error=None,scorer_error=None)
    try:
        parsed=json.loads(row['final'])
        assert isinstance(parsed,dict) and set(parsed)=={'program'} and isinstance(parsed['program'],str)
        tokens=official.program_tokenization(parsed['program'])
        if len(tokens)>81: raise ValueError('Program exceeds operation budget')
    except Exception as exc: result['format_error']=type(exc).__name__
    else:
        try:
            invalid,value=official.eval_program(tokens,target['table'])
            result['execution_correct']=invalid==0 and value==target['qa']['exe_ans']
            result['program_correct']=bool(official.equal_program(official.program_tokenization(target['qa']['program']),tokens))
        except Exception as exc: result['scorer_error']=dict(type=type(exc).__name__,message=str(exc))
    return result


def load_circuits():
    import numpy as np
    from scipy.io import loadmat
    from fly_model import UPSTREAM_COMMIT,parameter_matrices,reset_valence
    from fly_online import CausalMemory
    prior=BASE/'fin-skills-memory-paper-20260921-mamba'
    audit=prior/'port3/results/results.json'
    result=json.loads(audit.read_text())
    assert result['all_checks_passed'] and result['upstream_commit']==UPSTREAM_COMMIT
    path=prior/'upstream/data_and_parameters/Dx_steady_state_nonlinear_3_27-Mar-2023_3modules.mat'
    raw=loadmat(path)
    parameters=reset_valence(parameter_matrices(raw['para_mu'].T,raw['mat_lu_cell']),0.)
    return CausalMemory(parameters),CausalMemory(parameters),dict(parameters_sha256=sha(path),reference_audit_sha256=sha(audit))


def state(brain):
    return dict(weights=brain.weights.tolist(),clock=brain.clock,cue_scores=brain.cue_scores().tolist())


def condition(brain,program,correct):
    import numpy as np
    cue=cue_of(program)
    before=digest(state(brain))
    if cue is not None:
        reward=1. if correct else -1.
        brain.step(30.,np.eye(2)[cue],-reward,conditioning=True)
        brain.step(135.)
    return dict(cue=cue,before=before,after=digest(state(brain)))


class NativeReflexion:
    def __init__(self,backend):
        self.backend=backend; self.calls=[]
        receipt=json.loads((PREP/'upstream-receipt.json').read_text())
        assert all(sha(PREP/'reflexion'/p)==h for p,h in receipt['files'].items())
        shim=types.ModuleType('utils'); shim.get_completion=self.complete
        old=sys.modules.get('utils'); cwd=os.getcwd()
        try:
            sys.modules['utils']=shim; os.chdir(PREP/'reflexion')
            spec=importlib.util.spec_from_file_location('native_reflexion',PREP/'reflexion/generate_reflections.py')
            self.module=importlib.util.module_from_spec(spec); spec.loader.exec_module(self.module)
        finally:
            os.chdir(cwd)
            if old is None: del sys.modules['utils']
            else: sys.modules['utils']=old

    def complete(self,prompt):
        messages=[dict(role='user',content=REFLECTION_PREFIX+prompt)]
        record=dict(original_prompt=prompt,messages=messages); self.calls.append(record)
        try:
            response=self.backend(messages); record['response']=response
            return response['choices'][0]['message']['content']
        except Exception as exc:
            record['error']=dict(type=type(exc).__name__,message=str(exc)); raise

    def update(self,path,correct):
        configs=[dict(is_success=correct,skip=False,memory=[])]
        before=len(self.calls)
        self.module.update_memory(str(path),configs)
        assert len(self.calls)-before==int(not correct)
        return configs[0]['memory']


def memory_read(arm,case,records,index,frozen,learned):
    before=digest(dict(records=records,frozen=state(frozen),learned=state(learned)))
    if arm=='none': chosen=[]
    elif arm=='reflexion_recent': chosen=records[-3:]
    else:
        hits=index.search(case['question'],top_k=3)
        lookup={r['id']:r for r in records}; chosen=[lookup[h['document_id']] for h in hits]
    payload=dict(records=copy.deepcopy(chosen),circuit=None)
    if arm in ('fly_frozen','fly_learned'):
        brain=frozen if arm=='fly_frozen' else learned
        payload['circuit']=dict(scores=dict(zip(CUES,brain.cue_scores().tolist())),
            meaning='Uncalibrated past-only method-family scores, not correctness probabilities or expected returns.')
    assert before==digest(dict(records=records,frozen=state(frozen),learned=state(learned)))
    return payload,dict(state_sha256=before,experience_ids=[r['id'] for r in chosen],payload_sha256=digest(payload))


def qualify(root):
    from fin_skills.rag import RAGIndex
    frozen=json.loads((root/'manifest.json').read_text())
    assert all(sha(root/p)==h for p,h in frozen['sha256'].items())
    selection=json.loads((PREP/'task-selection.json').read_text())
    assert all(sha(PREP/p)==h for p,h in selection['output_sha256'].items())
    assert len(selection['acquisition_ids'])==16 and len(selection['evaluation_ids'])==32
    assert not set(selection['acquisition_companies'])&set(selection['evaluation_companies'])
    official=evaluator(UPSTREAM)
    runtime=software_qualification(official)
    frozen,learned,provenance=load_circuits()
    initial=digest(state(frozen)); condition(learned,'subtract(12, 10)',False)
    assert digest(state(frozen))==initial and digest(state(learned))!=initial
    records=[dict(id='fixture',question='change in cash',program='subtract(12, 10)',
                  feedback=dict(execution_correct=False),reflection='Check report units.')]
    index=RAGIndex([dict(id=r['id'],text=r['question']) for r in records],chunk_size=100000,overlap=0)
    for arm in ARMS:
        payload,trace=memory_read(arm,dict(question='change in cash'),records,index,frozen,learned)
        responses=iter(['Thought: Compute.\nAction: execute_program\nAction Input: {"program":"subtract(12, 10)"}',
                        'Thought: Done.\nAnswer: {"program":"subtract(12, 10)"}'])
        def scripted(messages):
            assert 'past_experiences' in messages[1]['content']
            assert ('Check report units.' in messages[1]['content'])==(arm!='none')
            return dict(choices=[dict(message=dict(content=next(responses)))])
        case=dict(id='fixture',question='change in cash',table=[],pre_text=[],post_text=[],past_experiences=payload,memory_usage_policy=POLICY)
        row=asyncio.run(episode(case,scripted,official))
        assert row['error'] is None and json.loads(row['final'])['program']=='subtract(12, 10)'
    reflection=NativeReflexion(lambda messages:dict(choices=[dict(message=dict(content='Plan: Check units.'))]))
    path=root/'fixture-reflection.log'; path.write_text('Here is the task:\nCompute a change. Failed: wrong units.')
    assert reflection.update(path,False)==['Plan: Check units.']
    assert reflection.update(path,True)==[]
    write(root/'software-qualification.json',dict(passed=True,runtime=runtime,all_five_arms_visible=True,
        nonmutating_reads=True,frozen_state_unchanged=True,conditioned_state_changed=True,
        original_reflexion_update_executed=True,parameter_provenance=provenance,model_quality_evidence=False))


def grade_command(root,path,out,split):
    assert split=='train', 'Intermediate feedback is limited to acquisition labels'
    receipt=json.loads(path.with_name(path.stem+'-receipt.json').read_text())
    assert receipt['episode_sha256']==sha(path)
    row=json.loads(path.read_text())
    target=next(x for x in json.loads((UPSTREAM/f'upstream/finqa/dataset/{split}.json').read_text()) if x['id']==row['id'])
    result=grade_row(row,target,evaluator(UPSTREAM))
    write(out,dict(id=row['id'],episode_sha256=sha(path),**result))


def run(root):
    import torch
    from llama_index.core import Settings
    from transformers_chat import TransformersChat
    from fin_skills.rag import RAGIndex
    assert os.environ.get('SLURM_JOB_ID') and root.resolve()==root and root.parent==BASE
    frozen_manifest=json.loads((root/'manifest.json').read_text())
    assert all(sha(root/p)==h for p,h in frozen_manifest['sha256'].items())
    prerequisite=Path(frozen_manifest['plan']['qualification_root'])
    assert json.loads((prerequisite/'completion.json').read_text())['status']=='completed'
    assert json.loads((prerequisite/'software-qualification.json').read_text())['passed']
    prior_manifest=json.loads((prerequisite/'manifest.json').read_text())
    assert all(prior_manifest['sha256'][p]==h for p,h in frozen_manifest['sha256'].items() if p.startswith('source/'))
    selection=json.loads((PREP/'task-selection.json').read_text())
    assert all(sha(PREP/p)==h for p,h in selection['output_sha256'].items())
    acquisition=json.loads((PREP/'acquisition-inputs.json').read_text())
    evaluation=json.loads((PREP/'evaluation-inputs.json').read_text())
    assert len(acquisition)==16 and len(evaluation)==32
    assert not {x['id'].split('/')[0] for x in acquisition}&{x['id'].split('/')[0] for x in evaluation}
    qualify(root)
    plan=frozen_manifest['plan']; family=plan['family']
    models=dict(qwen=('Qwen/Qwen2.5-Coder-14B-Instruct','aedcc2d42b622764e023cf882b6652e646b95671',
        str(BASE/'fin-skills-audit-20260921/cache/hf/hub')),
        mistral=('mistralai/Mistral-Nemo-Instruct-2407','04d8a90549d23fc6bd7f642064003592df51e9b3',
        str(BASE/'fin-skills-campaign-model-transfer-20260923-v1/model-cache/hub')))
    model,revision,cache=models[family]
    os.environ.update(HF_HUB_CACHE=cache,HUGGINGFACE_HUB_CACHE=cache)
    write(root/'protocol.json',dict(family=family,model=model,revision=revision,arms=ARMS,
        planned_acquisition=16,planned_evaluation=160,underlying_evaluation_tasks=32,
        selection_sha256=sha(PREP/'task-selection.json'),protocol_markdown_sha256=sha(root/'source/FINQA_MEMORY_PROTOCOL_20260923.md'),
        max_responses=6,max_output_tokens=512,reflection_max_responses=1,reflection_max_output_tokens=512,
        memory_policy=POLICY,reflection_prefix=REFLECTION_PREFIX,evaluation_feedback=False,
        shared_acquisition_charged_once=True,seed_rule='11 + first 8 hex SHA256(task ID) modulo 100000000',
        order_seed=20260923,limits='Public cross-company transfer adaptation, not original Reflexion scores; one inference seed.'))
    backend=TransformersChat(model,revision,max_tokens=512,seed=11)
    assert torch.cuda.get_device_properties(0).total_memory>=40*2**30
    Settings.tokenizer=lambda text:backend.tokenizer.encode(text,add_special_tokens=False)
    official=evaluator(UPSTREAM)
    fixed,learned,provenance=load_circuits()
    initial_fixed=digest(state(fixed))
    reflection=NativeReflexion(backend)
    for name in ('acquisition','feedback','reflections','evaluation'): (root/name).mkdir()
    records=[]; acquisition_receipts=[]

    def infer(case):
        backend.calls=0; backend.seed=11+int(hashlib.sha256(case['id'].encode()).hexdigest()[:8],16)%100000000
        torch.cuda.reset_peak_memory_stats()
        row=asyncio.run(episode(case,backend,official))
        torch.cuda.synchronize()
        row['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        row['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        return row

    for i,case in enumerate(acquisition):
        row=infer(dict(case,past_experiences=dict(records=[],circuit=None),memory_usage_policy=POLICY))
        path=root/'acquisition'/f'{i:02d}.json'; write(path,row)
        receipt=root/'acquisition'/f'{i:02d}-receipt.json'
        write(receipt,dict(id=case['id'],episode_sha256=sha(path)))
        grade=root/'feedback'/f'{i:02d}.json'
        subprocess.run([sys.executable,'-B',str(Path(__file__)),'grade',str(root),str(path),str(grade),'train'],check=True)
        feedback=json.loads(grade.read_text()); assert feedback['episode_sha256']==sha(path)
        trace='Here is the task:\n'+json.dumps(case,ensure_ascii=False)+'\nAttempted assistant messages:\n'
        trace+='\n'.join(c['response']['choices'][0]['message']['content'] for c in row['calls'] if 'response' in c)
        trace+='\nTool observations:\n'+json.dumps(row['tool_receipts'])+'\nFeedback:\n'+json.dumps(feedback)
        log=root/'reflections'/f'{i:02d}.log'; log.write_text(trace)
        backend.calls=0; backend.seed=100000001+i
        before=len(reflection.calls); reflection_error=None; plans=[]
        try: plans=reflection.update(log,feedback['execution_correct'])
        except Exception as exc: reflection_error=dict(type=type(exc).__name__,message=str(exc))
        write(root/'reflections'/f'{i:02d}.json',dict(plans=plans,error=reflection_error,calls=reflection.calls[before:]))
        program=program_of(row)
        transition=condition(learned,program,feedback['execution_correct'])
        record=dict(id=case['id'],question=case['question'],program=program,cue=cue_of(program),
            feedback={k:feedback[k] for k in ('execution_correct','format_error','scorer_error')},
            reflection='\n'.join(plans),reflection_error=reflection_error,available_after_acquisition=i)
        records.append(record)
        acquisition_receipts.append(dict(id=case['id'],episode_sha256=sha(path),feedback_sha256=sha(grade),
            reflection_sha256=sha(root/'reflections'/f'{i:02d}.json'),transition=transition))
        print(json.dumps(dict(phase='acquisition',completed=i+1,planned=16)),flush=True)
    assert digest(state(fixed))==initial_fixed
    bank=dict(records=records,frozen=state(fixed),learned=state(learned),parameter_provenance=provenance)
    write(root/'memory-bank.json',bank)
    write(root/'acquisition-receipt.json',dict(rows=acquisition_receipts,bank_sha256=sha(root/'memory-bank.json')))
    index=RAGIndex([dict(id=r['id'],text=r['question']+'\n'+r['reflection']) for r in records],chunk_size=100000,overlap=0)
    assert len(index.chunks)==16
    order=[(i,arm) for i in range(32) for arm in ARMS]; random.Random(20260923).shuffle(order)
    write(root/'evaluation-order.json',order)
    results=[]
    for n,(i,arm) in enumerate(order):
        case=evaluation[i]
        payload,read=memory_read(arm,case,records,index,fixed,learned)
        row=infer(dict(case,past_experiences=payload,memory_usage_policy=POLICY))
        row.update(arm=arm,memory_read=read,memory_payload=payload,bank_sha256=sha(root/'memory-bank.json'))
        assert digest(bank)==digest(dict(records=records,frozen=state(fixed),learned=state(learned),parameter_provenance=provenance))
        p=root/'evaluation'/f'{i:02d}-{arm}.json'; write(p,row)
        results.append(dict(id=case['id'],arm=arm,file=p.relative_to(root).as_posix(),sha256=sha(p)))
        print(json.dumps(dict(phase='evaluation',completed=n+1,planned=160)),flush=True)
    write(root/'evaluation-receipt.json',dict(planned=160,rows=results,bank_sha256=sha(root/'memory-bank.json')))
    subprocess.run([sys.executable,'-B',str(Path(__file__)),'score',str(root)],check=True)


def score(root):
    receipt=json.loads((root/'evaluation-receipt.json').read_text())
    assert len(receipt['rows'])==receipt['planned']==160
    planned=json.loads((PREP/'task-selection.json').read_text())['evaluation_ids']
    assert {(r['id'],r['arm']) for r in receipt['rows']}=={(i,a) for i in planned for a in ARMS}
    targets={x['id']:x for x in json.loads((UPSTREAM/'upstream/finqa/dataset/test.json').read_text())}
    official=evaluator(UPSTREAM); scores=[]
    assert len({(r['id'],r['arm']) for r in receipt['rows']})==160
    for item in receipt['rows']:
        path=root/item['file']; assert sha(path)==item['sha256']; row=json.loads(path.read_text())
        result=grade_row(row,targets[row['id']],official)
        scores.append(dict(id=row['id'],arm=row['arm'],error=row['error'],cue=cue_of(program_of(row)),
            model_call_attempts=len(row['calls']),generated_responses=sum('response' in c for c in row['calls']),
            input_tokens=sum(c['response']['usage']['prompt_tokens'] for c in row['calls'] if 'response' in c),
            output_tokens=sum(c['response']['usage']['completion_tokens'] for c in row['calls'] if 'response' in c),
            tool_receipts=len(row['tool_receipts']),elapsed_seconds=row['elapsed_seconds'],**result))
    write(root/'evaluation-scores.json',dict(rows=scores,evaluation_receipt_sha256=sha(root/'evaluation-receipt.json'),
        aggregate={arm:dict(planned=32,execution_correct=sum(r['execution_correct'] for r in scores if r['arm']==arm),
            program_correct=sum(r['program_correct'] for r in scores if r['arm']==arm),
            format_errors=sum(r['format_error'] is not None for r in scores if r['arm']==arm),
            scorer_errors=sum(r['scorer_error'] is not None for r in scores if r['arm']==arm)) for arm in ARMS}))


if __name__=='__main__':
    mode=sys.argv[1]; root=Path(sys.argv[2])
    if mode=='grade': grade_command(root,Path(sys.argv[3]),Path(sys.argv[4]),sys.argv[5])
    elif mode=='score': score(root)
    else:
        started=time.monotonic()
        try:
            if mode=='qualify': qualify(root)
            elif mode=='run': run(root)
            else: raise ValueError(mode)
        except Exception as exc:
            write(root/'completion.json',dict(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
            raise
        write(root/'completion.json',dict(status='completed',elapsed_seconds=time.monotonic()-started,mode=mode))
