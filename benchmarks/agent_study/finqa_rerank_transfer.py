"""Pinned BGE/Kev relevance ranking and downstream answers on the same FinQA tasks."""
import asyncio
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

from finqa_rag_ablation import (BASE,TASKS,UPSTREAM,contexts,episode,evaluator,pack,
                               parsed_answer,read,sha,write)


def verify(root):
    manifest=read(root/'manifest.json')
    assert all(sha(root/p)==h for p,h in manifest['sha256'].items())
    selection=read(TASKS/'task-selection.json')
    assert all(sha(TASKS/p)==h for p,h in selection['output_sha256'].items())
    cases=read(TASKS/'evaluation-inputs.json')
    assert len(cases)==32
    return manifest['plan'],cases


def rank(root):
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    from fin_skills.model_zoo import create_model
    plan,cases=verify(root)
    write(root/'protocol.json',dict(task_ids=[c['id'] for c in cases],candidate_k=10,context_k=5,
        context_characters=12000,methods=['bge','kev'],seed=11,
        candidate_source='actual FinSkills BM25 positive top 10, same report, question only',
        input_sha256=sha(TASKS/'evaluation-inputs.json'),
        limits=['Same FinQA task set as other studies, not independent extra tasks.',
            'BGE pair truncation 512 tokens disclosed; Kev strict no silent truncation.',
            'No threshold tuning, online feedback or LLM judge.',
            'First query includes model warmup; report separate load time, no speed-superiority claim.']))
    candidates={c['id']:contexts(c)[1].search(c['question'],top_k=10) for c in cases}
    write(root/'candidates.json',candidates)
    (root/'rankings').mkdir();inventory=[]
    for method in ('bge','kev'):
        started=time.monotonic()
        if method=='bge':
            modelroot=BASE/'fin-skills-campaign-bge-prepare-20260923-v1'
            modelreceipt=read(modelroot/'model-receipt.json')
            snapshot=Path(modelreceipt['snapshot'])
            assert modelreceipt['revision']=='953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
            assert all(sha(snapshot/n)==v['sha256'] for n,v in modelreceipt['files'].items())
            tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
            model=AutoModelForSequenceClassification.from_pretrained(snapshot,local_files_only=True,
                trust_remote_code=False,dtype=torch.float32).to('cuda').eval()
        else:
            config=read(root/'source/kev-config.json')
            model=create_model(config['model_id'],**config['parameters'])
            model._load()
        torch.cuda.synchronize()
        write(root/(method+'-load.json'),dict(seconds=time.monotonic()-started,gpu=torch.cuda.get_device_name(0)))
        for i,case in enumerate(cases):
            hits=candidates[case['id']];error=None;raw=None;prepared=None
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.monotonic()
            try:
                if method=='bge':
                    pairs=[[case['question'],h['text']] for h in hits]
                    lengths=[len(tokenizer(q,p,truncation=False)['input_ids']) for q,p in pairs]
                    scores=[]
                    for first in range(0,len(pairs),8):
                        data=tokenizer(pairs[first:first+8],padding=True,truncation=True,max_length=512,
                                       return_tensors='pt').to('cuda')
                        with torch.inference_mode(): values=model(**data,return_dict=True).logits.view(-1).float()
                        assert torch.isfinite(values).all();scores.extend(values.cpu().tolist())
                    ranked=sorted(range(len(hits)),key=lambda n:(-scores[n],n))
                    reordered=[dict(hits[n],bge_score=scores[n]) for n in ranked]
                    raw=dict(scores=scores,untruncated_pair_tokens=lengths,truncated_pairs=sum(n>512 for n in lengths))
                else:
                    raw=model.rerank(case['question'],hits)
                    reordered=raw['passages']
                assert {h['id'] for h in reordered}=={h['id'] for h in hits}
                prepared=pack(reordered,top_k=5,budget=12000)
            except Exception as exc:
                error=dict(type=type(exc).__name__,message=str(exc))
            torch.cuda.synchronize()
            path=root/'rankings'/f'{i:02d}-{method}.json'
            write(path,dict(id=case['id'],method=method,prepared=prepared,error=error,raw=raw,
                seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            inventory.append(dict(id=case['id'],method=method,file=path.relative_to(root).as_posix(),sha256=sha(path)))
            print(json.dumps(dict(phase='rank',method=method,completed=i+1,planned=32)),flush=True)
        del model
        if method=='bge': del tokenizer
        gc.collect();torch.cuda.empty_cache()
    write(root/'ranking-receipt.json',dict(rows=inventory,planned=64,candidates_sha256=sha(root/'candidates.json'),
        protocol_sha256=sha(root/'protocol.json')))


def answer(root):
    from llama_index.core import Settings
    from transformers_chat import TransformersChat
    plan,cases=verify(root)
    parent=Path(plan['ranking_root']);receipt=read(parent/'ranking-receipt.json')
    assert read(parent/'completion.json')['status']=='completed'
    assert len(receipt['rows'])==receipt['planned']==64
    assert {(r['id'],r['method']) for r in receipt['rows']}=={(c['id'],m) for c in cases for m in ('bge','kev')}
    records={}
    for item in receipt['rows']:
        p=parent/item['file'];assert sha(p)==item['sha256'];records[item['id'],item['method']]=read(p)
    guidance=(root/'source/skill.md').read_text(encoding='utf-8')
    write(root/'protocol.json',dict(plan=plan,ranking_receipt_sha256=sha(parent/'ranking-receipt.json'),
        task_ids=[c['id'] for c in cases],generated_conditions=['bge','kev'],
        max_responses=6,max_response_tokens=512,paired_with='same-seed rag_api from FinQA RAG ablation',
        limits='Same questions/model/guidance/calculator/output schema. Ranking failures remain planned failures.'))
    backend=TransformersChat(plan['model'],plan['revision'],max_tokens=512,seed=11)
    Settings.tokenizer=lambda text:backend.tokenizer.encode(text,add_special_tokens=False)
    official=evaluator(UPSTREAM)
    order=[(i,m) for i in range(32) for m in ('bge','kev')];random.Random(20260923).shuffle(order)
    (root/'episodes').mkdir();inventory=[]
    for n,(i,method) in enumerate(order):
        case=cases[i];ranked=records[case['id'],method]
        backend.calls=0;backend.seed=11+int(hashlib.sha256(case['id'].encode()).hexdigest()[:8],16)%100000000
        if ranked['error'] is None:
            row=asyncio.run(episode(case,ranked['prepared'],backend,official,guidance))
        else:
            row=dict(id=case['id'],final=None,error=dict(type='RankingFailure',details=ranked['error']),
                     calls=[],tool_receipts=[],elapsed_seconds=0)
        row.update(arm=method,ranking=ranked)
        path=root/'episodes'/f'{i:02d}-{method}.json';write(path,row)
        inventory.append(dict(id=case['id'],arm=method,file=path.relative_to(root).as_posix(),sha256=sha(path)))
        print(json.dumps(dict(phase='answer',completed=n+1,planned=len(order))),flush=True)
    write(root/'inference-receipt.json',dict(rows=inventory,planned=64,protocol_sha256=sha(root/'protocol.json')))
    subprocess.run([sys.executable,'-B',str(Path(__file__)),'score',str(root)],check=True)


def score(root):
    receipt=read(root/'inference-receipt.json');protocol=read(root/'protocol.json')
    assert sha(root/'protocol.json')==receipt['protocol_sha256']
    assert len(receipt['rows'])==receipt['planned']==64
    assert {(r['id'],r['arm']) for r in receipt['rows']}=={(i,m) for i in protocol['task_ids'] for m in ('bge','kev')}
    targets={r['id']:r for r in read(UPSTREAM/'upstream/finqa/dataset/test.json')}
    official=evaluator(UPSTREAM);scores=[]
    for item in receipt['rows']:
        p=root/item['file'];assert sha(p)==item['sha256'];row=read(p)
        result=dict(id=row['id'],arm=row['arm'],execution_correct=False,program_correct=False,
            error=None,model_calls=len(row['calls']),tool_calls=len(row['tool_receipts']),
            input_tokens=sum(c['response']['usage']['prompt_tokens'] for c in row['calls'] if 'response' in c),
            output_tokens=sum(c['response']['usage']['completion_tokens'] for c in row['calls'] if 'response' in c),
            seconds=row['elapsed_seconds'])
        try:
            parsed=parsed_answer(row['final']);tokens=official.program_tokenization(parsed['program'])
            if not parsed['program'].strip() or len(tokens)>81: raise ValueError('Invalid program length')
            target=targets[row['id']];invalid,value=official.eval_program(tokens,target['table'])
            result['execution_correct']=invalid==0 and value==target['qa']['exe_ans']
            result['program_correct']=bool(official.equal_program(official.program_tokenization(target['qa']['program']),tokens))
        except Exception as exc: result['error']=dict(type=type(exc).__name__,message=str(exc))
        scores.append(result)
    aggregate={m:dict(planned=32,correct=sum(r['execution_correct'] for r in scores if r['arm']==m),
        errors=sum(r['error'] is not None for r in scores if r['arm']==m)) for m in ('bge','kev')}
    write(root/'scores.json',dict(rows=scores,aggregate=aggregate,inference_receipt_sha256=sha(root/'inference-receipt.json')))


if __name__=='__main__':
    mode,root=sys.argv[1],Path(sys.argv[2])
    assert os.environ.get('SLURM_JOB_ID') and root.resolve()==root and root.parent==BASE
    if mode=='score': score(root)
    else:
        start=time.monotonic()
        try:
            if mode=='rank': rank(root)
            elif mode=='answer': answer(root)
            else: raise ValueError(mode)
        except Exception as exc:
            write(root/'completion.json',dict(status='failed',error=str(exc),traceback=traceback.format_exc()));raise
        write(root/'completion.json',dict(status='completed',seconds=time.monotonic()-start))
