"""One-pass inference, separate official numerical scoring, and full-denominator audit."""
import collections
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback

import docfinqa_context as ctx
from finqa_reuse import evaluator, sha, write


def inputs(family):
    assert ctx.read(ctx.PACKS / 'completion.json')['status'] == 'completed'
    receipt = ctx.read(ctx.PACKS / 'pack-receipt.json')
    entry = next(x for x in receipt['rows'] if x['family'] == family)
    assert sha(ctx.PACKS / entry['file']) == entry['sha256']
    rows = ctx.read(ctx.PACKS / entry['file'])
    selected = ctx.read(ctx.TASKS / 'task-selection.json')['evaluation_ids']
    assert len(rows) == 64 and {(x['id'], x['memory']) for x in rows} == {
        (i, m) for i in selected for m in ('none', 'fly_learned')}
    return {(x['id'], x['memory']): x for x in rows}, entry


def seed(task):
    return 11 + int(hashlib.sha256(task.encode()).hexdigest()[:8], 16) % 100000000


def grade(row, target, official):
    result = dict(correct=False, input_error=row['input_error'], generation_error=row['generation_error'],
                  format_error=None, execution_invalid=False, scorer_error=None, value=None)
    if result['input_error'] or result['generation_error']:
        return result
    try:
        parsed = ctx.scalar_result(row['response']['text'], official)
    except (ValueError, TypeError, KeyError, AssertionError) as exc:
        result['format_error'] = dict(type=type(exc).__name__, message=str(exc))
    except Exception as exc:
        result['scorer_error'] = dict(type=type(exc).__name__, message=str(exc))
    else:
        result.update(execution_invalid=bool(parsed['invalid']), value=parsed['value'])
        result['correct'] = parsed['invalid'] == 0 and parsed['value'] == target
    return result


def software(root):
    from histrim import DualRouter
    ctx.source_check(root)
    official = evaluator(ctx.BOOT)
    def row(text, error=None):
        return dict(input_error=error, generation_error=None, response=dict(text=text))
    assert grade(row('{"program":"subtract(12, 10)"}'), 2., official)['correct']
    assert not grade(row('{"program":"subtract(12, 10)"}'), 3., official)['correct']
    assert grade(row('{"program":"divide(1, 3)"}'), .33333, official)['correct']
    assert grade(row('{"program":"greater(3, 2)"}'), 'yes', official)['correct']
    assert grade(row('```json\n{"program":"add(1, 1)"}\n```'), 2., official)['format_error']
    assert grade(row('{"program":"table_sum(revenue, none)"}'), 2., official)['format_error']
    assert grade(row('{"program":"divide(1, 0)"}'), 0., official)['execution_invalid']
    assert grade(row('', 'missing_report'), 0., official)['input_error'] == 'missing_report'
    assert len(set(ctx.ARMS)) == 6 and seed('fixture') == seed('fixture')
    stored = DualRouter([1., 0.], [0., 1.], threshold=.4).as_dict()
    restored = DualRouter(**stored)
    assert restored.as_dict() == stored
    write(root / 'runner-software-qualification.json', dict(passed=True,
        correct_and_incorrect_fixtures=True, original_rounding_and_greater=True,
        no_parser_relaxation=True, missing_denominator_preserved=True, router_restore_roundtrip=True, labels_read=False))


def infer(root):
    import torch
    from histrim import PackedChat, DualRouter
    from transformers_chat import TransformersChat
    ctx.source_check(root)
    plan = ctx.read(root / 'manifest.json')['plan']; family = plan['family']
    qual = Path(plan['qualification_root'])
    assert ctx.read(qual / (family + '-completion.json'))['status'] == 'completed'
    assert ctx.read(qual / family / 'qualification.json')['passed']
    assert ctx.read(qual / 'runner-software-qualification.json')['passed']
    frozen = ctx.read(qual / 'manifest.json')['sha256']
    for name in ('docfinqa_context.py', 'docfinqa_run.py', 'histrim.py', 'transformers_chat.py'):
        assert sha(root / 'source' / name) == frozen['source/' + name]
    cases, entry = inputs(family)
    model, revision, _ = ctx.MODELS[family]
    base = TransformersChat(model, revision, max_tokens=512, seed=11)
    chat = PackedChat(base, max_tokens=512)
    routers = ctx.read(qual / family / 'routers.json')
    chat.routers = {int(k): DualRouter(**v) for k, v in routers.items()}
    write(root / 'routers.json', routers)
    order = [(i, m, c) for i in ctx.read(ctx.TASKS / 'task-selection.json')['evaluation_ids'] for m, c in ctx.ARMS]
    random.Random(20260923).shuffle(order)
    write(root / 'evaluation-order.json', order)
    write(root / 'protocol.json', dict(family=family, model=model, revision=revision, planned=192,
        arms=ctx.ARMS, source_protocol_sha256=sha(root / 'source/DOCFINQA_CONTEXT_PROTOCOL_20260923.md'),
        pack_receipt_sha256=sha(ctx.PACKS / 'pack-receipt.json'), inputs_sha256=entry['sha256'],
        router_source_sha256=sha(qual / family / 'routers.json'), responses_per_unit=1,
        output_token_cap=512, total_capacity=8192, seed_rule='11 + first 8 hex task SHA256 modulo 100000000',
        memory_updates=False, gold_read=False, device=torch.cuda.get_device_name(0),
        limits='Common retrieved report excerpts, not full reports; single response, not ReAct; same prior tasks.'))
    (root / 'episodes').mkdir(); inventory = []
    for n, (task, memory, mode) in enumerate(order):
        packet = cases[(task, memory)]
        row = dict(id=task, memory=memory, mode=mode, seed=seed(task), input_error=packet['error'],
            generation_error=None, response=None, offered_item_ids=[x['id'] for x in packet['items']],
            experience_ids=packet['experience_ids'], bank_sha256=packet['bank_sha256'],
            expected_prompt_sha256=packet['prompt_sha256'])
        if packet['error'] is None:
            ids, _, prompt = chat.encode(ctx.SYSTEM, packet['protected'], packet['items'])
            assert ids.shape[1] == packet['input_tokens']
            assert hashlib.sha256(prompt.encode()).hexdigest() == packet['prompt_sha256']
            try:
                row['response'] = chat(ctx.SYSTEM, packet['protected'], packet['items'], mode=mode, seed=seed(task))
            except torch.cuda.OutOfMemoryError:
                raise  # Infrastructure failure retains unfinished units; do not call them model errors.
            except Exception as exc:
                row['generation_error'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        path = root / 'episodes' / f'{n:03d}.json'; write(path, row)
        inventory.append(dict(id=task, memory=memory, mode=mode, file=path.relative_to(root).as_posix(), sha256=sha(path)))
        print(json.dumps(dict(completed=n+1, planned=192)), flush=True)
    write(root / 'inference-receipt.json', dict(planned=192, rows=inventory,
        protocol_sha256=sha(root / 'protocol.json'), order_sha256=sha(root / 'evaluation-order.json')))


def score(root):
    ctx.source_check(root)
    receipt = ctx.read(root / 'inference-receipt.json')
    assert len(receipt['rows']) == 192
    label_path = ctx.BOOT / 'upstream/finqa/dataset/test.json'
    provenance = ctx.read(ctx.BOOT / 'upstream-manifest.json')['sources']['finqa']['files']['dataset/test.json']
    assert sha(label_path) == provenance['sha256']
    labels = {x['id']: x for x in ctx.read(label_path)}
    official = evaluator(ctx.BOOT); rows = []
    for entry in receipt['rows']:
        path = root / entry['file']; assert sha(path) == entry['sha256']
        row = ctx.read(path)
        rows.append(dict(id=row['id'], memory=row['memory'], mode=row['mode'],
            **grade(row, labels[row['id']]['qa']['exe_ans'], official)))
    write(root / 'scores.json', dict(rows=rows, inference_receipt_sha256=sha(root / 'inference-receipt.json'),
        labels_sha256=sha(label_path), scope='Official scalar execution with empty table, exact original execution answer; no symbolic-accuracy claim.'))


def audit(root):
    from transformers import AutoTokenizer
    assert root.resolve() == root and root.parent == ctx.BASE
    assert all(sha(root / p) == h for p, h in ctx.read(root / 'manifest.json')['sha256'].items())
    assert ctx.read(root / 'completion.json')['status'] == 'completed'
    family = ctx.read(root / 'manifest.json')['plan']['family']
    model, revision, cache = ctx.MODELS[family]
    tokenizer = AutoTokenizer.from_pretrained(cache / ('models--' + model.replace('/', '--')) / 'snapshots' / revision,
        local_files_only=True, trust_remote_code=False)
    cases, _ = inputs(family); receipt = ctx.read(root / 'inference-receipt.json')
    assert sha(root / 'protocol.json') == receipt['protocol_sha256']
    assert sha(root / 'evaluation-order.json') == receipt['order_sha256']
    expected = {(i, m, c) for i, m in cases for c in ('full', 'recency', 'histrim')}
    assert len(receipt['rows']) == len(expected) == 192
    assert {(x['id'], x['memory'], x['mode']) for x in receipt['rows']} == expected
    saved = ctx.read(root / 'scores.json'); assert saved['inference_receipt_sha256'] == sha(root / 'inference-receipt.json')
    scores = {(r['id'], r['memory'], r['mode']): r for r in saved['rows']}
    assert len(saved['rows']) == 192 and set(scores) == expected
    assert saved['labels_sha256'] == sha(ctx.BOOT / 'upstream/finqa/dataset/test.json')
    labels = {x['id']: x for x in ctx.read(ctx.BOOT / 'upstream/finqa/dataset/test.json')}
    official = evaluator(ctx.BOOT)
    totals = {m+'|'+c: dict(planned=32, correct=0, input_errors=0, generation_errors=0,
        format_errors=0, execution_invalid=0, scorer_errors=0, responses=0, input_tokens=0,
        output_tokens=0, wall_seconds=0., prefill_seconds=0., kv_bytes=[], peak_allocated_bytes=[],
        peak_reserved_bytes=[], final_memory_items_retained=0, final_document_items_retained=0) for m, c in ctx.ARMS}
    for entry in receipt['rows']:
        path = root / entry['file']; assert sha(path) == entry['sha256']; row = ctx.read(path)
        key = (row['id'], row['memory'], row['mode']); packet = cases[key[:2]]
        assert row['input_error'] == packet['error'] and row['seed'] == seed(row['id'])
        assert row['offered_item_ids'] == [x['id'] for x in packet['items']]
        assert row['experience_ids'] == packet['experience_ids'] and row['bank_sha256'] == packet['bank_sha256']
        g = grade(row, labels[row['id']]['qa']['exe_ans'], official)
        assert all(scores[key][k] == v for k, v in g.items())
        total = totals[row['memory']+'|'+row['mode']]; total['correct'] += g['correct']
        for out, name in [('input_errors', 'input_error'), ('generation_errors', 'generation_error'),
                          ('format_errors', 'format_error'), ('execution_invalid', 'execution_invalid'), ('scorer_errors', 'scorer_error')]:
            total[out] += bool(g[name])
        response = row['response']
        if response is not None:
            assert response['prompt_sha256'] == packet['prompt_sha256'] and response['prompt_tokens'] == packet['input_tokens']
            assert response['mode'] == row['mode'] and response['completion_tokens'] <= 512
            assert response['prompt_tokens'] + 512 <= 8192
            prompt, count = ctx.render(tokenizer, packet['protected'], packet['items'])
            assert count == packet['input_tokens'] and hashlib.sha256(prompt.encode()).hexdigest() == packet['prompt_sha256']
            offsets = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
            protected = set(range(len(offsets)))
            for i, item in enumerate(packet['items']):
                block = f"<history_{i}>\n{item['text']}\n</history_{i}>"
                begin = prompt.index(block); end = begin + len(block)
                protected -= {j for j, (a, b) in enumerate(offsets) if a >= begin and b <= end and b > a}
            total['responses'] += 1; total['input_tokens'] += response['prompt_tokens']; total['output_tokens'] += response['completion_tokens']
            total['wall_seconds'] += response['elapsed_seconds']; total['prefill_seconds'] += response['prefill_seconds']
            for k, saved_key in [('kv_bytes', 'prefill_kv_bytes'), ('peak_allocated_bytes', 'peak_allocated_bytes'), ('peak_reserved_bytes', 'peak_reserved_bytes')]:
                total[k].append(response[saved_key])
            retained = list(range(len(packet['items']))) if not response['selection'] else response['selection'][-1]['retained_items']
            for i in retained:
                kind = packet['items'][i]['kind']
                total['final_document_items_retained' if kind == 'document' else 'final_memory_items_retained'] += 1
            for step in response['selection']:
                assert protected <= set(step['retained_original_positions']) <= set(step['previous_positions'])
                assert step['history_tokens'] <= step['token_budget']
        else:
            assert row['input_error'] or row['generation_error']
    write(root / 'context-verified-summary.json', dict(family=family, denominator=192, arms=totals,
        receipt_sha256=sha(root / 'inference-receipt.json'), scores_sha256=sha(root / 'scores.json'),
        audit_source_sha256=sha(Path(__file__)), limits='Same 32 tasks across six arms; all failures retained. Latency includes output-length differences; reserved memory depends on allocator history.'))


if __name__ == '__main__':
    action, root = sys.argv[1], Path(sys.argv[2]); start = time.monotonic()
    if action in ('software', 'audit'):
        globals()[action](root)
    else:
        try:
            globals()[action](root)
        except Exception as exc:
            write(root / (action+'-completion.json'), dict(status='failed', error_type=type(exc).__name__,
                error=str(exc), traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-start))
            raise
        write(root / (action+'-completion.json'), dict(status='completed', elapsed_seconds=time.monotonic()-start))
        if action == 'score':
            assert ctx.read(root / 'infer-completion.json')['status'] == 'completed'
            write(root / 'completion.json', dict(status='completed', elapsed_seconds=
                ctx.read(root / 'infer-completion.json')['elapsed_seconds'] + time.monotonic()-start))
