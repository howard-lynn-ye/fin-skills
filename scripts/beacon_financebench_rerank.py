"""Run the official BGE Transformers inference recipe on frozen retrieval candidates."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

BASE=Path('/beacon-projects/radfm/wy891')
CORPUS=BASE/'fin-skills-campaign-financebench-corpus-20260923-v2'
RETRIEVAL=BASE/'fin-skills-campaign-financebench-retrieval-20260923-v2'
MODEL=BASE/'fin-skills-campaign-bge-prepare-20260923-v1'
LABELS=BASE/'fin-skills-campaign-reuse-bootstrap-20260923-v2/upstream/financebench/data/financebench_open_source.jsonl'


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False)


def run(root):
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    from fin_skills.rag.documents import chunk_documents
    assert os.environ.get('SLURM_JOB_ID') and root.resolve()==root and root.parent==BASE
    frozen=json.loads((root/'manifest.json').read_text())
    assert all(sha(root/p)==h for p,h in frozen['sha256'].items())
    for parent in (CORPUS,RETRIEVAL,MODEL):
        assert json.loads((parent/'completion.json').read_text())['status']=='completed'
    retrieval=json.loads((RETRIEVAL/'retrieval-receipt.json').read_text())
    assert len(retrieval['rows'])==150
    corpus_receipt=json.loads((CORPUS/'corpus-receipt.json').read_text())
    assert sha(CORPUS/'extraction-manifest.json')==corpus_receipt['extraction_manifest_sha256']
    protocol=json.loads((RETRIEVAL/'protocol.json').read_text())
    assert sha(RETRIEVAL/'protocol.json')==retrieval['protocol_sha256']
    assert protocol['corpus_receipt_sha256']==sha(CORPUS/'corpus-receipt.json')
    model_receipt=json.loads((MODEL/'model-receipt.json').read_text())
    assert model_receipt['revision']=='953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
    model_path=Path(model_receipt['snapshot'])
    for name,item in model_receipt['files'].items():
        assert sha(model_path/name)==item['sha256']
    documents=[]
    for row in json.loads((CORPUS/'extraction-manifest.json').read_text()):
        if row['status']!='extracted': continue
        path=CORPUS/row['file']; assert sha(path)==row['sha256']
        for page in json.loads(path.read_text()):
            documents.append(dict(id=page['id'],text=page['text'],source=page['source'],
                metadata=dict(doc_name=page['doc_name'],page_num=page['page_num'],pdf_sha256=page['pdf_sha256'])))
    chunks={c['id']:c for c in chunk_documents(documents,chunk_size=1200,overlap=200)}
    assert torch.cuda.is_available()
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    model=AutoModelForSequenceClassification.from_pretrained(model_path,local_files_only=True,
        trust_remote_code=False,torch_dtype=torch.float32).to('cuda').eval()
    write(root/'protocol.json',dict(model=model_receipt['model'],revision=model_receipt['revision'],
        model_receipt_sha256=sha(MODEL/'model-receipt.json'),retrieval_receipt_sha256=sha(RETRIEVAL/'retrieval-receipt.json'),
        planned_questions=150,planned_candidate_sets=300,methods=['finskills','bm25s'],candidate_limit=50,
        candidate_source='frozen native retrieval outputs, no gold page filtering',
        max_pair_tokens=512,truncation='longest_first as official example; all original pair lengths reported',
        dtype='float32',batch_size=8,seed=11,score='sequence-classification raw logit',
        tie_break='original candidate rank',grade_cutoffs=[1,5,10,20,50],
        gpu=torch.cuda.get_device_name(0),gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
        limits=['Independent BGE baseline, not Jev.', 'Evidence page retrieval only, not answer correctness.',
            'Candidates limited by first-stage retrieval; candidate ceiling reported.',
            'Truncated pairs cannot demonstrate full-document support; record truncation explicitly.',
            'Model public-data training contamination unknown; inference offline, no external API.']))
    torch.manual_seed(11)
    warm=tokenizer([['capital expenditure','capital expenditure was 10 million']],padding=True,
        truncation=True,max_length=512,return_tensors='pt').to('cuda')
    with torch.inference_mode():
        out=model(**warm,return_dict=True).logits
    assert out.shape==(1,1) and torch.isfinite(out).all()
    torch.cuda.synchronize()
    write(root/'qualification.json',dict(passed=True,finite_logits=True,financial_labels_used=False))
    (root/'queries').mkdir()
    inventory=[]
    for entry in retrieval['rows']:
        path=RETRIEVAL/entry['file']; assert sha(path)==entry['sha256']
        original=json.loads(path.read_text())
        record=dict(id=entry['id'],retrieval_row_sha256=entry['sha256'],methods={})
        for method in ('finskills','bm25s'):
            row=dict(source_errors=original['methods'][method]['errors'],hits=[],error=None)
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started=time.monotonic()
            try:
                candidates=original['methods'][method].get('hits',[])
                assert len(candidates)<=50 and len({h['id'] for h in candidates})==len(candidates)
                pairs=[[original['query'],chunks[h['id']]['text']] for h in candidates]
                assert all(h['metadata']==chunks[h['id']]['metadata'] for h in candidates)
                lengths=[len(tokenizer(q,p,add_special_tokens=True,truncation=False)['input_ids']) for q,p in pairs]
                logits=[]; actual_tokens=0; padded_tokens=0
                for start in range(0,len(pairs),8):
                    inputs=tokenizer(pairs[start:start+8],padding=True,truncation=True,
                                     max_length=512,return_tensors='pt').to('cuda')
                    actual_tokens+=int(inputs['attention_mask'].sum().item())
                    padded_tokens+=int(inputs['input_ids'].numel())
                    with torch.inference_mode():
                        scores=model(**inputs,return_dict=True).logits.view(-1).float()
                    assert torch.isfinite(scores).all()
                    logits.extend(scores.cpu().tolist())
                ranked=sorted(range(len(candidates)),key=lambda i:(-logits[i],i))
                row.update(hits=[dict(candidates[i],bge_score=logits[i],original_rank=i+1) for i in ranked],
                    original_pair_tokens=lengths,truncated_pairs=sum(x>512 for x in lengths),
                    actual_tokens=actual_tokens,padded_tokens=padded_tokens)
            except Exception as exc:
                row['error']=dict(type=type(exc).__name__,message=str(exc))
            torch.cuda.synchronize()
            row.update(elapsed_seconds=time.monotonic()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            record['methods'][method]=row
        output=root/'queries'/(entry['id']+'.json'); write(output,record)
        inventory.append(dict(id=entry['id'],file=output.relative_to(root).as_posix(),sha256=sha(output)))
        print(json.dumps(dict(completed=len(inventory),planned=150)),flush=True)
    write(root/'rerank-receipt.json',dict(rows=inventory,planned_candidate_sets=300,protocol_sha256=sha(root/'protocol.json')))
    assert sha(LABELS)=='a5a2aa673e573e55675fc3c0f9aa38c1cf59d2abc91edb077534f71f10a71877'
    gold={x['financebench_id']:x for x in (json.loads(l) for l in LABELS.read_text().splitlines() if l)}
    scored=[]
    for entry in inventory:
        p=root/entry['file']; assert sha(p)==entry['sha256']; record=json.loads(p.read_text())
        expected={(x['doc_name'],x['evidence_page_num']) for x in gold[entry['id']]['evidence']}
        for method,item in record['methods'].items():
            metrics={}
            for k in (1,5,10,20,50):
                pages={(h['metadata']['doc_name'],h['metadata']['page_num']) for h in item['hits'][:k]}
                metrics[str(k)]=dict(any_page_hit=bool(pages&expected),all_pages_hit=bool(expected) and expected<=pages,
                    page_recall=len(pages&expected)/len(expected) if expected else None)
            scored.append(dict(id=entry['id'],method=method,metrics=metrics,error=item['error'],source_errors=item['source_errors']))
    write(root/'rerank-scores.json',dict(rows=scored,planned=300,rerank_receipt_sha256=sha(root/'rerank-receipt.json'),
        aggregate={method:{str(k):dict(any_page_hits=sum(x['metrics'][str(k)]['any_page_hit'] for x in scored if x['method']==method),
            all_pages_hits=sum(x['metrics'][str(k)]['all_pages_hit'] for x in scored if x['method']==method),denominator=150)
            for k in (1,5,10,20,50)} for method in ('finskills','bm25s')}))


if __name__=='__main__':
    root=Path(sys.argv[1]); start=time.monotonic()
    try: run(root)
    except Exception as exc:
        write(root/'completion.json',dict(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
        raise
    write(root/'completion.json',dict(status='completed',elapsed_seconds=time.monotonic()-start))
