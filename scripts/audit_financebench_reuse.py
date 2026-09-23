"""Read-only corpus/retrieval/rerank checks; exclusive summaries and complete denominators."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

BASE = Path('/beacon-projects/radfm/wy891')
CORPUS = BASE / 'fin-skills-campaign-financebench-corpus-20260923-v2'
RETRIEVAL = BASE / 'fin-skills-campaign-financebench-retrieval-20260923-v2'
LABELS = BASE / 'fin-skills-campaign-reuse-bootstrap-20260923-v2/upstream/financebench/data/financebench_open_source.jsonl'
METHODS = ('finskills', 'bm25s')
CUTOFFS = (1, 5, 10, 20, 50)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_check(root):
    assert root.resolve() == root and root.parent == BASE
    assert read(root / 'completion.json')['status'] == 'completed'
    assert all(sha(root / p) == h for p, h in read(root / 'manifest.json')['sha256'].items())


def corpus(root):
    receipt = read(root / 'corpus-receipt.json')
    protocol = read(root / 'acquisition-protocol.json')
    assert protocol['revision'] == 'cc39aeb4afdf33909ee1412188bf89035950c2eb'
    assert sha(root / 'upstream-tree.json') == protocol['tree_sha256']
    for name, key in [('download-manifest.json', 'download_manifest_sha256'),
                      ('extraction-manifest.json', 'extraction_manifest_sha256'),
                      ('public-inputs.json', 'inputs_sha256')]:
        assert sha(root / name) == receipt[key]
    assert read(root / 'software-qualification.json')['passed']
    downloads = read(root / 'download-manifest.json')
    extraction = read(root / 'extraction-manifest.json')
    planned = {x['path']: x for x in protocol['planned_pdfs']}
    assert len(planned) == len(downloads) == len(extraction) == receipt['planned_pdfs'] == 368
    assert {x['upstream_path'] for x in downloads} == set(planned)
    assert {x['upstream_path'] for x in extraction} == set(planned)
    down = {x['upstream_path']: x for x in downloads}
    for item in downloads:
        if item['status'] != 'downloaded':
            continue
        path = root / item['local_path']
        assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
        blob = hashlib.sha1(b'blob ' + str(item['bytes']).encode() + b'\0' + path.read_bytes()).hexdigest()
        assert blob == item['git_blob_sha1'] == planned[item['upstream_path']]['sha']
    counts = collections.Counter()
    for item in extraction:
        counts[item['status']] += 1
        if item['status'] != 'extracted':
            continue
        path = root / item['file']
        assert sha(path) == item['sha256']
        pages = read(path)
        assert len(pages) == item['pages']
        assert sum(bool(x['text'].strip()) for x in pages) == item['nonempty_pages']
        for n, page in enumerate(pages):
            assert page['page_num'] == n
            assert page['doc_name'] == Path(item['upstream_path']).stem
            assert page['id'] == page['doc_name'] + ':page:' + str(n)
            assert page['pdf_sha256'] == down[item['upstream_path']]['sha256']
        if item.get('reused_from'):
            assert sha(Path(item['reused_from'])) == item['sha256']
        counts['pages'] += len(pages)
        counts['nonempty_pages'] += item['nonempty_pages']
    assert counts['extracted'] == receipt['extracted']
    assert counts['pages'] == receipt['pages'] and counts['nonempty_pages'] == receipt['nonempty_pages']
    prior = Path(protocol['prior_corpus']) if protocol.get('prior_corpus') else None
    if prior:
        old = {x['upstream_path']: x for x in read(prior / 'extraction-manifest.json')}
        for item in extraction:
            before = old[item['upstream_path']]
            if before['status'] == 'extracted':
                assert item['sha256'] == before['sha256'] and item.get('reused_from')
            elif item.get('repair_evidence'):
                evidence = item['repair_evidence']; log = Path(evidence['log'])
                assert sha(log) == evidence['sha256']
                assert 'DependencyError: cryptography>=3.1 is required for AES algorithm' in log.read_text()
            else:
                assert item['status'] == before['status'] and item['retained_failure']
    inputs = read(root / 'public-inputs.json')
    assert len(inputs) == len({x['financebench_id'] for x in inputs}) == 150
    assert all(set(x) == {'financebench_id', 'question', 'company', 'doc_name'} for x in inputs)
    return dict(kind='corpus', planned_pdfs=368, counts=dict(counts), question_denominator=150,
        receipt_sha256=sha(root / 'corpus-receipt.json'), failures=[x for x in extraction if x['status'] != 'extracted'],
        prior_corpus=str(prior) if prior else None, label_generation=False)


def metrics(hits, expected):
    result = {}
    for k in CUTOFFS:
        got = {(h['metadata']['doc_name'], h['metadata']['page_num']) for h in hits[:k]}
        result[str(k)] = dict(any_page_hit=bool(got & expected),
            all_pages_hit=bool(expected) and expected <= got,
            page_recall=len(got & expected) / len(expected) if expected else None)
    return result


def rankings(root, rerank):
    source_check(CORPUS)
    assert sha(LABELS) == 'a5a2aa673e573e55675fc3c0f9aa38c1cf59d2abc91edb077534f71f10a71877'
    gold = {x['financebench_id']: x for x in map(json.loads, LABELS.read_text().splitlines())}
    questions = {x['financebench_id']: x for x in read(CORPUS / 'public-inputs.json')}
    assert set(questions) == set(gold) and len(questions) == 150
    prefix = 'rerank' if rerank else 'retrieval'
    receipt = read(root / (prefix + '-receipt.json'))
    scores = read(root / (prefix + '-scores.json'))
    protocol = read(root / 'protocol.json')
    assert sha(root / 'protocol.json') == receipt['protocol_sha256']
    assert read(root / 'qualification.json')['passed']
    assert sha(root / (prefix + '-receipt.json')) == scores[prefix + '_receipt_sha256']
    assert len(receipt['rows']) == 150 and {x['id'] for x in receipt['rows']} == set(questions)
    assert len(scores['rows']) == 300
    saved = {(x['id'], x['method']): x for x in scores['rows']}
    assert len(saved) == 300 and set(saved) == {(qid, m) for qid in questions for m in METHODS}
    if rerank:
        source_check(RETRIEVAL)
        assert protocol['retrieval_receipt_sha256'] == sha(RETRIEVAL / 'retrieval-receipt.json')
        assert protocol['revision'] == '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
        original = {x['id']: x for x in read(RETRIEVAL / 'retrieval-receipt.json')['rows']}
    else:
        assert protocol['corpus_receipt_sha256'] == sha(CORPUS / 'corpus-receipt.json')
        assert protocol['doc_name_filter'] is False and protocol['query_fields'] == ['question']
    result = {m: dict(planned=150, errors=0, hits={str(k): 0 for k in CUTOFFS},
        all_pages_hits={str(k): 0 for k in CUTOFFS}, latencies=[], actual_tokens=0, padded_tokens=0,
        truncated_pairs=0, pairs=0, peak_allocated_bytes=0, peak_reserved_bytes=0,
        candidate_ceiling_any=0) for m in METHODS}
    for entry in receipt['rows']:
        path = root / entry['file']; assert sha(path) == entry['sha256']
        row = read(path); assert row['id'] == entry['id'] and set(row['methods']) == set(METHODS)
        expected = {(x['doc_name'], x['evidence_page_num']) for x in gold[row['id']]['evidence']}
        if rerank:
            old_path = RETRIEVAL / original[row['id']]['file']
            assert sha(old_path) == original[row['id']]['sha256'] == row['retrieval_row_sha256']
            old = read(old_path)
        else:
            assert row['query'] == questions[row['id']]['question']
            assert len(row['repeat_order']) == 3 and all(sorted(x) == sorted(METHODS) for x in row['repeat_order'])
        for method, item in row['methods'].items():
            out = result[method]; hits = item.get('hits', [])
            assert len(hits) <= 50 and len({h['id'] for h in hits}) == len(hits)
            calculated = metrics(hits, expected)
            grade = saved[(row['id'], method)]
            assert calculated == grade['metrics']
            for k in CUTOFFS:
                out['hits'][str(k)] += calculated[str(k)]['any_page_hit']
                out['all_pages_hits'][str(k)] += calculated[str(k)]['all_pages_hit']
            if rerank:
                candidates = old['methods'][method].get('hits', [])
                out['candidate_ceiling_any'] += metrics(candidates, expected)['50']['any_page_hit']
                assert item['source_errors'] == old['methods'][method]['errors'] == grade['source_errors']
                assert item['error'] == grade['error']
                out['errors'] += bool(item['error'] or item['source_errors'])
                if item['error'] is None:
                    assert len(hits) == len(candidates)
                    assert {x['original_rank'] for x in hits} == set(range(1, len(candidates) + 1))
                    assert hits == sorted(hits, key=lambda x: (-x['bge_score'], x['original_rank']))
                    for hit in hits:
                        assert math.isfinite(hit['bge_score'])
                        assert {k: v for k, v in hit.items() if k not in ('bge_score', 'original_rank')} == candidates[hit['original_rank'] - 1]
                    assert len(item['original_pair_tokens']) == len(candidates)
                    assert item['truncated_pairs'] == sum(x > 512 for x in item['original_pair_tokens'])
                    assert item['actual_tokens'] <= item['padded_tokens'] <= 512 * len(candidates)
                    out['pairs'] += len(candidates)
                    for key in ('actual_tokens', 'padded_tokens', 'truncated_pairs'):
                        out[key] += item[key]
                out['latencies'].append(item['elapsed_seconds'])
                for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
                    out[key] = max(out[key], item[key])
            else:
                assert item['errors'] == grade['errors']
                out['errors'] += bool(item['errors'])
                assert all(math.isfinite(h['score']) and h['score'] > 0 for h in hits)
                assert all(hits[i]['score'] >= hits[i+1]['score'] for i in range(len(hits)-1))
                if not item['errors']:
                    assert item['warmup_ok'] and len(item['timings']) == 3
                observed = statistics.median(item['timings']) if item['timings'] else None
                assert observed == grade['query_seconds_median']
                if observed is not None:
                    out['latencies'].append(observed)
    for method, item in result.items():
        for k in CUTOFFS:
            aggregate = scores['aggregate'][method][str(k)] if rerank else scores['aggregate'][method]['cutoffs'][str(k)]
            assert aggregate['any_page_hits'] == item['hits'][str(k)]
            assert aggregate['all_pages_hits'] == item['all_pages_hits'][str(k)]
        item['latency_median_seconds'] = statistics.median(item['latencies']) if item['latencies'] else None
        item['measured_latency_units'] = len(item.pop('latencies'))
    return dict(kind=prefix, methods=result, denominator=300, receipt_sha256=sha(root / (prefix + '-receipt.json')),
        score_sha256=sha(root / (prefix + '-scores.json')),
        limits='Page retrieval only. Recomputes scores and validates frozen candidate artifacts; does not rerun/tune rankings or measure answer correctness.')


if __name__ == '__main__':
    root = Path(sys.argv[1]); source_check(root)
    if (root / 'corpus-receipt.json').exists():
        result = corpus(root)
    else:
        result = rankings(root, (root / 'rerank-receipt.json').exists())
    result['audit_sha256'] = sha(Path(__file__))
    with (root / 'reuse-verified-summary.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result))
