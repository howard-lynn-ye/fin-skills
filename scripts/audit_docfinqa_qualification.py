"""Verify the two completed interface qualifications before submitting the study."""
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = Path(sys.argv[1])
assert root.resolve() == root and root.parent == Path('/beacon-projects/radfm/wy891')
manifest = read(root / 'manifest.json')
assert all(sha(root / p) == h for p, h in manifest['sha256'].items())
assert read(root / 'runner-software-qualification.json')['passed']
assert read(root / 'runner-software-qualification.json')['router_restore_roundtrip']
pack = root.parent / 'fin-skills-campaign-docfinqa-packs-20260923-v1'
assert read(pack / 'completion.json')['status'] == 'completed'
receipt = read(pack / 'pack-receipt.json')
assert all(sha(pack / e['file']) == e['sha256'] for e in receipt['rows'])
assert sha(pack / 'retrieved-packs.json') == receipt['retrieved_packs_sha256']
assert sha(root / 'source/DOCFINQA_CONTEXT_PROTOCOL_20260923.md') == receipt['protocol_sha256']
families = {}
for family in ('qwen', 'mistral'):
    assert read(root / (family + '-completion.json'))['status'] == 'completed'
    out = root / family
    assert read(out / 'qualification.json')['passed']
    manual, real = read(out / 'decoder-equivalence.json'), read(out / 'payload-equivalence.json')
    assert manual['passed'] and real['passed'] and manual['top_token_equal'] and real['top_token_equal']
    trace = read(out / 'physical-cache-check.json')
    for mode in ('recency', 'histrim'):
        assert trace[mode]['prefill_kv_bytes'] < trace['full']['prefill_kv_bytes']
        assert len(trace[mode]['selection']) == 2
        for step in trace[mode]['selection']:
            assert set(step['retained_original_positions']) <= set(step['previous_positions'])
            assert step['history_tokens'] <= step['token_budget']
    families[family] = dict(manual_equivalence=manual, payload_equivalence=real,
        routers_sha256=sha(out / 'routers.json'), fixture_kv_bytes={k: v['prefill_kv_bytes'] for k, v in trace.items()},
        artifacts_sha256={p.name: sha(p) for p in out.glob('*.json')})
summary = dict(passed=True, families=families, source_sha256=manifest['sha256'],
    pack_receipt_sha256=sha(pack / 'pack-receipt.json'), pack_inputs=receipt['rows'],
    planned_units=384, model_quality_evidence=False, audit_sha256=sha(Path(__file__)))
with (root / 'qualification-verified.json').open('x') as stream:
    json.dump(summary, stream, indent=2, allow_nan=False)
print(json.dumps(dict(passed=True, families=families, summary_sha256=sha(root / 'qualification-verified.json'))))
