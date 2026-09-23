"""Read-only replay of a complete FinQA RAG or reranked-answer run."""
import hashlib
import json
from pathlib import Path
import re
import sys


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    base=Path('/beacon-projects/radfm/wy891')
    assert root.resolve()==root and root.parent==base
    sys.path.insert(0,str(root/'source'))
    library=root/'source/library'
    if library.exists(): sys.path.insert(0,str(library))
    from finqa_rag_ablation import evaluator,parsed_answer,UPSTREAM,TASKS,GENERATED,REPORTED
    manifest=read(root/'manifest.json')
    assert all(sha(root/p)==h for p,h in manifest['sha256'].items())
    assert read(root/'completion.json')['status']=='completed'
    receipt=read(root/'inference-receipt.json');scores=read(root/'scores.json');protocol=read(root/'protocol.json')
    assert scores['inference_receipt_sha256']==sha(root/'inference-receipt.json')
    assert receipt['protocol_sha256']==sha(root/'protocol.json')
    reranked='ranking_root' in manifest['plan']
    methods=('bge','kev') if reranked else GENERATED
    reported=methods if reranked else REPORTED
    cases={c['id']:c for c in read(TASKS/'evaluation-inputs.json')}
    assert len(cases)==32 and set(protocol['task_ids'])==set(cases)
    upstream=read(UPSTREAM/'upstream-manifest.json')
    label_path=UPSTREAM/'upstream/finqa/dataset/test.json'
    assert sha(label_path)==upstream['sources']['finqa']['files']['dataset/test.json']['sha256']
    gold={r['id']:r for r in read(label_path)};official=evaluator(UPSTREAM)
    expected={(i,a) for i in cases for a in methods}
    assert len(receipt['rows'])==receipt['planned']==len(expected)
    assert {(r['id'],r['arm']) for r in receipt['rows']}==expected
    saved={(r['id'],r['arm']):r for r in scores['rows']}
    assert len(saved)==len(scores['rows'])==len(cases)*len(reported)
    assert set(saved)=={(i,a) for i in cases for a in reported}
    replay=[]
    for item in receipt['rows']:
        path=root/item['file'];assert sha(path)==item['sha256'];row=read(path)
        assert row['id']==item['id'] and row['arm']==item['arm']
        assert len(row['calls'])<=6
        for call in row['calls']:
            if 'response' in call:
                usage=call['response']['usage']
                assert usage['completion_tokens']<=512 and usage['prompt_tokens']+512<=32768
        # Replay actual arithmetic calls independently; do not accept logged result claims.
        for call in row['tool_receipts']:
            invalid,value=official.eval_program(official.program_tokenization(call['program']),cases[row['id']]['table'])
            assert invalid==call['invalid'] and value==call['result']
        correct=program_correct=accepted=False
        try:
            parsed=parsed_answer(row['final']);tokens=official.program_tokenization(parsed['program'])
            assert parsed['program'].strip() and len(tokens)<=81
            accepted=True
            invalid,value=official.eval_program(tokens,gold[row['id']]['table'])
            correct=invalid==0 and value==gold[row['id']]['qa']['exe_ans']
            program_correct=bool(official.equal_program(official.program_tokenization(gold[row['id']]['qa']['program']),tokens))
        except (ValueError,TypeError,KeyError,AssertionError,IndexError,ZeroDivisionError): pass
        saved_row=saved[row['id'],row['arm']]
        assert saved_row['execution_correct']==correct and saved_row['program_correct']==program_correct
        if not reranked:
            assert saved_row['accepted']==accepted
            if row['arm']=='rag_api':
                known={p['citation_id'] for p in row['prepared']['passages']}
                cited=set(re.findall(r'\[(S\d+)\]',row['final'] or ''))
                gate_valid=bool(cited) and cited<=known and bool(row['prepared']['passages'])
                assert saved[row['id'],'rag_gate']['accepted']==(accepted and gate_valid)
                assert saved[row['id'],'rag_gate']['execution_correct']==correct
        replay.append(dict(id=row['id'],arm=row['arm'],correct=correct,program_correct=program_correct))
    output=root/'rag-verified-summary.json'
    result=dict(verified=True,generated_rows=len(replay),reported_rows=len(saved),
        unique_tasks=len(cases),unique_companies=len({i.split('/')[0] for i in cases}),
        aggregate=scores['aggregate'],inference_receipt_sha256=sha(root/'inference-receipt.json'),
        labels_sha256=sha(label_path),score_sha256=sha(root/'scores.json'),
        audit_source_sha256=sha(Path(__file__)),
        limits='Replay verifies original numerical/program grades and receipts, not citation semantic support or contamination.')
    with output.open('x') as out: json.dump(result,out,indent=2)
    print(json.dumps(result))


if __name__=='__main__': audit(Path(sys.argv[1]))
