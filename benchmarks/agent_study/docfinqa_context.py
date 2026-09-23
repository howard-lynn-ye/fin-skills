"""Frozen retrieved-report context experiment; reuse FinQA scoring and existing HiSTrim."""
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback

import numpy as np
from finqa_reuse import evaluator, sha, write

BASE = Path('/beacon-projects/radfm/wy891')
DOC = BASE / 'fin-skills-campaign-docfinqa-prepare-20260923-v1'
TASKS = BASE / 'fin-skills-campaign-memory-reuse-prepare-20260923-v1'
BOOT = BASE / 'fin-skills-campaign-reuse-bootstrap-20260923-v2'
PACKS = BASE / 'fin-skills-campaign-docfinqa-packs-20260923-v1'
MODELS = dict(
    qwen=('Qwen/Qwen2.5-Coder-14B-Instruct', 'aedcc2d42b622764e023cf882b6652e646b95671',
          BASE / 'fin-skills-audit-20260921/cache/hf/hub'),
    mistral=('mistralai/Mistral-Nemo-Instruct-2407', '04d8a90549d23fc6bd7f642064003592df51e9b3',
             BASE / 'fin-skills-campaign-model-transfer-20260923-v1/model-cache/hub'))
SYSTEM = ('Answer financial questions using only the supplied report excerpts for current numbers. '
          'Past experience is fallible procedural guidance, not evidence for current numbers. '
          'Return exactly one JSON object with the single key program, without markdown. '
          'Use the FinQA scalar DSL: add(a, b), subtract(a, b), multiply(a, b), divide(a, b), '
          'exp(a, b), greater(a, b). Separate operations with comma-space and refer to earlier '
          'results as #0, #1. Numeric constants may use const_100 or literal 100. '
          'Percentages must be expressed in percentage points when the question asks a percent. '
          'No table operations or outside data. Example: {"program":"subtract(12, 10)"}.')
ARMS = [(m, c) for m in ('none', 'fly_learned') for c in ('full', 'recency', 'histrim')]


def read(path):
    return json.loads(path.read_text())


def source_check(root):
    assert root.resolve() == root and root.parent == BASE and os.environ.get('SLURM_JOB_ID')
    assert all(sha(root / p) == h for p, h in read(root / 'manifest.json')['sha256'].items())


def scalar_result(text, official):
    parsed = json.loads(text)
    assert isinstance(parsed, dict) and set(parsed) == {'program'} and isinstance(parsed['program'], str)
    tokens = official.program_tokenization(parsed['program'])
    assert len(tokens) <= 81
    operators = re.findall(r'([A-Za-z_]+)\s*\(', parsed['program'])
    assert operators and set(operators) <= {'add', 'subtract', 'multiply', 'divide', 'exp', 'greater'}
    invalid, value = official.eval_program(tokens, [])
    return dict(program=parsed['program'], invalid=invalid, value=value)


def calibration_examples():
    # Same four non-benchmark manual-relevance prompts used in the original qualification.
    from decision_tasks import PAIRS, PARAMETERS, RULES
    system = ('You are a financial research agent. Follow the current client contract. '
              'Historical experience is fallible. Return exactly the requested JSON object, '
              'without markdown. Use the available numerical API rather than inventing values.')
    result = []
    for i, family in enumerate(PAIRS):
        items = [dict(id='manual-' + method, text=f'Reference manual for {f}: {RULES[method]} Default API parameters: {json.dumps(PARAMETERS[method])}. These defaults do not override the client contract.')
                 for f, methods in PAIRS.items() for method in methods]
        shift = i % len(items); items = items[shift:] + items[:shift]
        result.append(dict(system=system, protected=f'Review methods for a {family} task. Reply with an applicable method ID.',
            items=items, labels=[1. if x['id'].removeprefix('manual-') in PAIRS[family] else -1. for x in items]))
    return result


def render(tokenizer, protected, items):
    blocks = [f"<history_{i}>\n{item['text']}\n</history_{i}>" for i, item in enumerate(items)]
    content = '\n'.join(blocks) + '\nCURRENT TASK (protected):\n' + protected
    prompt = tokenizer.apply_chat_template([dict(role='system', content=SYSTEM),
        dict(role='user', content=content)], tokenize=False, add_generation_prompt=True)
    return prompt, len(tokenizer(prompt, add_special_tokens=False)['input_ids'])


def prepare(root):
    from fin_skills.rag import RAGIndex
    from transformers import AutoTokenizer
    import finqa_memory_study as memory
    source_check(root)
    assert read(DOC / 'completion.json')['status'] == 'completed'
    doc_receipt = read(DOC / 'corpus-receipt.json')
    assert all(sha(DOC / n) == h for n, h in doc_receipt['artifacts'].items())
    selection = read(TASKS / 'task-selection.json')
    assert all(sha(TASKS / n) == h for n, h in selection['output_sha256'].items())
    cases = read(TASKS / 'evaluation-inputs.json')
    assert [x['id'] for x in cases] == selection['evaluation_ids'] and len(cases) == 32
    overlap = read(DOC / 'overlap.json')
    by_doc = {x['id']: x for x in overlap['rows']}
    docs = {}
    for split in ('train', 'dev', 'test'):
        p = DOC / f'public-inputs/{split}.jsonl'
        assert sha(p) == doc_receipt['inventory'][split]['public_sha256']
        docs.update((x['id'], x) for x in map(json.loads, p.read_text().splitlines()))
    training_contexts = {by_doc[did]['context_sha256'] for fid in selection['acquisition_ids']
                         for did in overlap['fixed_matches'][fid]}
    records = []
    for case in cases:
        match = overlap['fixed_matches'][case['id']]
        row = dict(id=case['id'], question=case['question'], candidates=match, error=None, chunks=[])
        if len(match) != 1:
            row['error'] = 'missing_or_multiple_docfinqa_candidates'
        elif by_doc[match[0]]['finqa_candidates'] != [dict(split='test', id=case['id'])] or not match[0].startswith('docfinqa/test/'):
            row['error'] = 'ambiguous_or_cross_split_question_mapping'
        elif by_doc[match[0]]['context_sha256'] in training_contexts:
            row['error'] = 'exact_report_overlap_with_acquisition'
        else:
            d = docs[match[0]]; p = DOC / d['context_path']; assert sha(p) == d['context_sha256']
            text = p.read_text(encoding='utf-8'); start = time.monotonic()
            index = RAGIndex([dict(id=d['id'], text=text)], chunk_size=1200, overlap=200)
            built = time.monotonic(); hits = index.search(case['question'], top_k=8)
            row.update(docfinqa_id=d['id'], context_sha256=d['context_sha256'],
                index_seconds=built-start, query_seconds=time.monotonic()-built,
                total_report_chunks=len(index.chunks), mapping_scope='unique question match; not an independently verified report identifier')
            row['chunks'] = sorted([dict(h, retrieval_rank=i+1) for i, h in enumerate(hits)], key=lambda h: h['start'])
            assert all(text[h['start']:h['end']] == h['text'] for h in row['chunks'])
        records.append(row)
    write(root / 'retrieved-packs.json', records)
    official = evaluator(BOOT)
    for program, value in [('subtract(12, 10)', 2.), ('divide(1, 3)', .33333), ('greater(3, 2)', 'yes')]:
        got = scalar_result(json.dumps(dict(program=program)), official)
        assert got['invalid'] == 0 and got['value'] == value, got
    for text in ('```json\n{"program":"add(1, 1)"}\n```', '{"program":"table_sum(revenue, none)"}'):
        try:
            scalar_result(text, official)
        except Exception:
            pass
        else:
            raise AssertionError('Invalid output fixture accepted')
    write(root / 'scorer-qualification.json', dict(passed=True, original_executor=True,
        empty_table=True, strict_json=True, scalar_only=True, financial_labels_read=False))
    manifest = []
    for family, (model, revision, cache) in MODELS.items():
        snapshot = cache / ('models--' + model.replace('/', '--')) / 'snapshots' / revision
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        bank_root = BASE / f'fin-skills-campaign-finqa-memory-{family}-20260923-v2'
        acquisition = read(bank_root / 'acquisition-receipt.json'); bank = read(bank_root / 'memory-bank.json')
        assert acquisition['bank_sha256'] == sha(bank_root / 'memory-bank.json')
        assert [r['id'] for r in bank['records']] == selection['acquisition_ids'] and len(acquisition['rows']) == 16
        fixed, learned, provenance = memory.load_circuits(); assert provenance == bank['parameter_provenance']
        for r in bank['records']:
            memory.condition(learned, r['program'], r['feedback']['execution_correct'])
        for brain, name in [(fixed, 'frozen'), (learned, 'learned')]:
            for key, value in memory.state(brain).items():
                assert np.allclose(value, bank[name][key], rtol=1e-10, atol=1e-12)
        idx = RAGIndex([dict(id=r['id'], text=r['question']+'\n'+r['reflection']) for r in bank['records']], chunk_size=100000, overlap=0)
        lookup = {r['id']: r for r in bank['records']}; rows = []
        for case in records:
            chosen = [lookup[h['document_id']] for h in idx.search(case['question'], top_k=3)]
            for mem in ('none', 'fly_learned'):
                items = []
                if mem != 'none':
                    items += [dict(id='memory/' + x['id'], kind='memory', text=json.dumps(x, sort_keys=True)) for x in chosen]
                    items.append(dict(id='memory/circuit', kind='circuit', text=json.dumps(dict(
                        scores=dict(zip(memory.CUES, bank['learned']['cue_scores'])), meaning='Uncalibrated acquisition-only method-family guidance, not correctness probabilities.'))))
                items += [dict(id='document/' + h['id'], kind='document', text=h['text']) for h in case['chunks']]
                protected = json.dumps(dict(question=case['question'], instruction='Return the scalar program computing the answer from these excerpts; follow the system output contract.'))
                prompt, tokens = render(tokenizer, protected, items)
                rows.append(dict(id=case['id'], memory=mem, items=items, protected=protected,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), input_tokens=tokens,
                    error=case['error'] or ('capacity_exceeded' if tokens+512 > 8192 else None),
                    experience_ids=[x['id'] for x in chosen] if mem != 'none' else [],
                    bank_sha256=sha(bank_root / 'memory-bank.json')))
        p = root / (family + '-inputs.json'); write(p, rows)
        manifest.append(dict(family=family, file=p.name, sha256=sha(p), payloads=len(rows),
            planned_inference_units=192, input_failures=sum(x['error'] is not None for x in rows)*3,
            max_input_tokens=max(x['input_tokens'] for x in rows), bank_sha256=sha(bank_root / 'memory-bank.json')))
    write(root / 'pack-receipt.json', dict(rows=manifest, retrieved_packs_sha256=sha(root / 'retrieved-packs.json'),
        protocol_sha256=sha(root / 'source/DOCFINQA_CONTEXT_PROTOCOL_20260923.md'),
        planned_tasks=32, planned_units=384, labels_read=False, inference=False))


def qualify(root, family):
    from transformers_chat import TransformersChat
    from histrim import PackedChat
    source_check(root)
    assert read(PACKS / 'completion.json')['status'] == 'completed'
    receipt = read(PACKS / 'pack-receipt.json')
    p = next(x for x in receipt['rows'] if x['family'] == family)
    assert sha(PACKS / p['file']) == p['sha256']
    # Use a manual fixture, never test answer labels, to qualify the unchanged pruning code.
    model, revision, cache = MODELS[family]
    base = TransformersChat(model, revision, max_tokens=512, seed=11)
    chat = PackedChat(base, max_tokens=8)
    out = root / family; out.mkdir()
    examples = calibration_examples()
    write(out / 'routers.json', chat.calibrate(examples))
    example = examples[0]
    eq = chat.equivalence(example['system'], example['protected'], example['items'])
    assert eq['passed']; write(out / 'decoder-equivalence.json', eq)
    traces = {mode: chat(example['system'], example['protected'], example['items'], mode=mode, seed=97)
              for mode in ('full', 'recency', 'histrim')}
    ids, assignments, _ = chat.encode(example['system'], example['protected'], example['items'])
    protected = set(np.flatnonzero(assignments < 0).tolist())
    for mode in ('recency', 'histrim'):
        assert traces[mode]['prefill_kv_bytes'] < traces['full']['prefill_kv_bytes']
        for step in traces[mode]['selection']:
            assert protected <= set(step['retained_original_positions']) <= set(step['previous_positions'])
    write(out / 'physical-cache-check.json', traces)
    # Test exact production payload rendering and full-decoder logits, without generation or grading.
    rows = read(PACKS / p['file'])
    case = next(x for x in rows if x['memory'] == 'fly_learned' and x['error'] is None)
    ids, _, prompt = chat.encode(SYSTEM, case['protected'], case['items'])
    assert ids.shape[1] == case['input_tokens'] and hashlib.sha256(prompt.encode()).hexdigest() == case['prompt_sha256']
    eq_real = chat.equivalence(SYSTEM, case['protected'], case['items'])
    assert eq_real['passed']
    write(out / 'payload-equivalence.json', dict(id=case['id'], **eq_real, input_tokens=case['input_tokens'],
        labels_read=False, scope='Unlabeled interface/logit check, no generated answer or task score'))
    write(out / 'qualification.json', dict(passed=True, family=family, source_and_packet_hashes=True,
        scalar_scorer=read(PACKS / 'scorer-qualification.json'), financial_quality_evidence=False))


if __name__ == '__main__':
    action, root = sys.argv[1], Path(sys.argv[2]); start = time.monotonic()
    target = root / ('completion.json' if action == 'prepare' else sys.argv[3] + '-completion.json')
    try:
        if action == 'prepare':
            prepare(root)
        elif action == 'qualify':
            qualify(root, sys.argv[3])
        else:
            raise ValueError(action)
    except Exception as exc:
        write(target, dict(status='failed', error_type=type(exc).__name__, error=str(exc),
            traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-start))
        raise
    write(target, dict(status='completed', elapsed_seconds=time.monotonic()-start))
