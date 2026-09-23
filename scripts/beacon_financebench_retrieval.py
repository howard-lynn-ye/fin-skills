"""Paired full-corpus FinanceBench retrieval using actual FinSkills and BM25S APIs."""
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

CORPUS = Path('/beacon-projects/radfm/wy891/fin-skills-campaign-financebench-corpus-20260923-v2')
LABELS = Path('/beacon-projects/radfm/wy891/fin-skills-campaign-reuse-bootstrap-20260923-v2/upstream/financebench/data/financebench_open_source.jsonl')


def write(path, value):
    with path.open('x') as out:
        json.dump(value, out, indent=2, allow_nan=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root):
    import bm25s
    import numpy as np
    from fin_skills.rag import RAGIndex
    from fin_skills.rag.index import _tokens
    assert os.environ.get('SLURM_JOB_ID') and root.resolve()==root
    assert root.parent == Path('/beacon-projects/radfm/wy891')
    manifest=json.loads((root/'manifest.json').read_text())
    assert all(sha(root/p)==h for p,h in manifest['sha256'].items())
    assert json.loads((CORPUS/'completion.json').read_text())['status']=='completed'
    receipt=json.loads((CORPUS/'corpus-receipt.json').read_text())
    assert sha(CORPUS/'extraction-manifest.json')==receipt['extraction_manifest_sha256']
    assert sha(CORPUS/'public-inputs.json')==receipt['inputs_sha256']
    assert importlib.metadata.version('bm25s')=='0.3.11'

    # Same tokenizer, unique query terms, k1/b and corpus. Lucene omits a common 2.5 scale.
    toy=RAGIndex([dict(id=str(i),text=text) for i,text in enumerate([
        'cash cash growth dividends', 'cash debt', 'sales growth'])],chunk_size=1200,overlap=200)
    toy_chunks=toy.chunks
    mature=bm25s.BM25(k1=1.5,b=.75,method='lucene',dtype='float64')
    mature.index([_tokens(c['text']) for c in toy_chunks],show_progress=False)
    checks=[]
    for query in ['cash', 'growth dividends', 'cash cash debt']:
        expected={h['id']:h['score'] for h in toy.search(query,top_k=3)}
        observed=mature.get_scores(sorted(set(_tokens(query))))*2.5
        reference=np.array([expected.get(c['id'],0.) for c in toy_chunks])
        assert np.allclose(observed,reference,rtol=1e-6,atol=1e-8), (observed,reference)
        checks.append(dict(query=query,max_abs_error=float(np.max(np.abs(observed-reference)))))
    write(root/'qualification.json',dict(passed=True,matched_bm25_scores=checks,
        bm25s_version=importlib.metadata.version('bm25s'),
        bm25s_source_sha256=sha(Path(inspect.getfile(bm25s.BM25))),
        finskills_source_sha256=sha(Path(inspect.getfile(RAGIndex)))))

    documents=[]
    extraction=json.loads((CORPUS/'extraction-manifest.json').read_text())
    for row in extraction:
        if row['status']!='extracted':
            continue
        path=CORPUS/row['file']; assert sha(path)==row['sha256']
        for page in json.loads(path.read_text()):
            documents.append(dict(id=page['id'],text=page['text'],source=page['source'],
                metadata=dict(doc_name=page['doc_name'],page_num=page['page_num'],pdf_sha256=page['pdf_sha256'])))
    questions=json.loads((CORPUS/'public-inputs.json').read_text())
    assert len(questions)==150 and len({x['financebench_id'] for x in questions})==150
    write(root/'protocol.json',dict(questions=[x['financebench_id'] for x in questions],
        corpus_receipt_sha256=sha(CORPUS/'corpus-receipt.json'),
        corpus='all extracted pages of every PDF at the pinned upstream commit; global retrieval',
        query_fields=['question'],doc_name_filter=False,chunk_characters=1200,overlap_characters=200,
        k1=1.5,b=.75,bm25s_method='lucene',bm25s_dtype='float64',query_terms='sorted unique FinSkills tokens',
        max_returned_chunks=50,grade_cutoffs=[1,5,10,20,50],timed_repeats=3,warmups_per_question_per_method=1,
        order_seed=20260923,calls_sequential_same_allocation=True,
        ties='native APIs retained; BM25S top-k tie selection may differ from FinSkills ID sort',
        positive_scores_only=True,labels_used_after_retrieval_receipt=True,
        limitations=['External public historical task set, not training-contamination-free.',
            'Page evidence retrieval is not answer correctness or complete claim support.',
            'Same lexical algorithm; this isolates component/API behavior, not whole-library superiority.',
            'Shared chunking/tokenization; their preprocessing is outside query timing.',
            'Index construction work differs; costs reported separately. No historical as-of filtering.']))
    construction={}
    start=time.monotonic(); index=RAGIndex(documents,chunk_size=1200,overlap=200)
    construction['finskills_seconds']=time.monotonic()-start
    chunks=index.chunks
    assert len(chunks)>=50
    start=time.monotonic()
    tokens=[_tokens(c['text']) for c in chunks]
    construction['shared_bm25s_token_materialization_seconds']=time.monotonic()-start
    start=time.monotonic(); mature=bm25s.BM25(k1=1.5,b=.75,method='lucene',dtype='float64')
    mature.index(tokens,show_progress=False)
    construction['bm25s_index_seconds']=time.monotonic()-start
    del tokens
    write(root/'index-inventory.json',dict(pages=len(documents),chunks=len(chunks),
        construction=construction,corpus_receipt_sha256=sha(CORPUS/'corpus-receipt.json')))
    (root/'queries').mkdir()

    def query(method,text):
        if method=='finskills':
            hits=index.search(text,top_k=50)
        else:
            ids,scores=mature.retrieve([sorted(set(_tokens(text)))],k=50,
                show_progress=False,n_threads=0,backend_selection='numpy')
            hits=[dict(chunks[int(i)],score=float(score)) for i,score in zip(ids[0],scores[0]) if score>0]
        return [dict(id=h['id'],document_id=h['document_id'],metadata=h['metadata'],score=h['score']) for h in hits]

    rng=random.Random(20260923)
    inventory=[]
    for n,question in enumerate(questions):
        record=dict(id=question['financebench_id'],query=question['question'],methods={})
        order=['finskills','bm25s']; rng.shuffle(order)
        for method in order:
            try:
                query(method,question['question'])
                record['methods'][method]=dict(warmup_ok=True,timings=[],errors=[])
            except Exception as exc:
                record['methods'][method]=dict(warmup_ok=False,timings=[],errors=[dict(stage='warmup',type=type(exc).__name__,message=str(exc))])
        record['repeat_order']=[]
        for repeat in range(3):
            order=['finskills','bm25s']; rng.shuffle(order); record['repeat_order'].append(order[:])
            for method in order:
                item=record['methods'][method]; start=time.monotonic()
                try:
                    hits=query(method,question['question'])
                    item['timings'].append(time.monotonic()-start)
                    if 'hits' in item:
                        assert item['hits']==hits, 'Within-method repeated ranking changed'
                    else:
                        item['hits']=hits
                except Exception as exc:
                    item['errors'].append(dict(stage='repeat',repeat=repeat,type=type(exc).__name__,message=str(exc)))
        path=root/'queries'/(question['financebench_id']+'.json')
        write(path,record); inventory.append(dict(id=record['id'],file=path.relative_to(root).as_posix(),sha256=sha(path)))
        print(json.dumps(dict(completed=n+1,planned=len(questions))),flush=True)
    write(root/'retrieval-receipt.json',dict(planned_questions=150,planned_method_queries=300,
        rows=inventory,protocol_sha256=sha(root/'protocol.json')))
    # Labels cannot affect index, prompts, queries or ranking; only read after frozen receipt.
    assert sha(LABELS)=='a5a2aa673e573e55675fc3c0f9aa38c1cf59d2abc91edb077534f71f10a71877'
    labels={x['financebench_id']:x for x in (json.loads(l) for l in LABELS.read_text().splitlines() if l)}
    available={(d['metadata']['doc_name'],d['metadata']['page_num']) for d in documents if d['text'].strip()}
    scores=[]
    for entry in inventory:
        path=root/entry['file']; assert sha(path)==entry['sha256']; record=json.loads(path.read_text())
        expected={(e['doc_name'],e['evidence_page_num']) for e in labels[entry['id']]['evidence']}
        for method,item in record['methods'].items():
            metrics={}
            for k in (1,5,10,20,50):
                got={(h['metadata']['doc_name'],h['metadata']['page_num']) for h in item.get('hits',[])[:k]}
                metrics[str(k)]=dict(any_page_hit=bool(got & expected),
                    all_pages_hit=bool(expected) and expected<=got,
                    page_recall=len(got & expected)/len(expected) if expected else None)
            scores.append(dict(id=entry['id'],method=method,metrics=metrics,errors=item['errors'],
                expected_pages=len(expected),missing_or_empty_pages=sorted(expected-available),
                query_seconds_median=statistics.median(item['timings']) if item['timings'] else None))
    aggregate={}
    for method in ('finskills','bm25s'):
        group=[r for r in scores if r['method']==method]; assert len(group)==150
        aggregate[method]=dict(planned=150,with_errors=sum(bool(x['errors']) for x in group),
            cutoffs={str(k):dict(any_page_hits=sum(x['metrics'][str(k)]['any_page_hit'] for x in group),
                all_pages_hits=sum(x['metrics'][str(k)]['all_pages_hit'] for x in group),
                macro_page_recall=sum(x['metrics'][str(k)]['page_recall'] or 0 for x in group)/150)
                for k in (1,5,10,20,50)},
            query_seconds_median=statistics.median([x['query_seconds_median'] for x in group if x['query_seconds_median'] is not None]))
    write(root/'retrieval-scores.json',dict(rows=scores,aggregate=aggregate,
        retrieval_receipt_sha256=sha(root/'retrieval-receipt.json'),
        labels_sha256=sha(LABELS),limits='Evidence page retrieval only; missing/error queries kept in all 150 denominators.'))


if __name__=='__main__':
    root=Path(sys.argv[1]); start=time.monotonic()
    try:
        run(root)
    except Exception as exc:
        write(root/'completion.json',dict(status='failed',error_type=type(exc).__name__,error=str(exc),
            traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start))
        raise
    write(root/'completion.json',dict(status='completed',elapsed_seconds=time.monotonic()-start))
