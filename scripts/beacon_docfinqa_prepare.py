"""Stage the author's DocFinQA, isolate labels, and inspect overlap/capacity without inference."""
import collections
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
import unicodedata

BASE = Path('/beacon-projects/radfm/wy891')
BOOT = BASE / 'fin-skills-campaign-reuse-bootstrap-20260923-v2'
MEM = BASE / 'fin-skills-campaign-memory-reuse-prepare-20260923-v1'
REVISION = '64ebaff62f692495bcc182f45cf9a9606251b19b'
FILES = {
    'dev.json': (496285887, 'f8b69c455c6772320caf42544eaf802fbabdc716fecd12c938356fbe0635deb9'),
    'test.json': (581393499, '0ebf95ccc719e72f45333e256928c90e8fa7a9a81f226dcb38ad7b3a48a3ff42'),
    'train.json': (3624998065, 'ab59345df17638ebd59402adeab07b0768563a670cfb3538aea97cfdefb93f31'),
}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def norm(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().casefold()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def run(root):
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer
    assert os.environ.get('SLURM_JOB_ID') and root.resolve() == root and root.parent == BASE
    manifest = json.loads((root / 'manifest.json').read_text())
    assert all(sha(root / p) == h for p, h in manifest['sha256'].items())
    assert norm(' A\n B ') == norm('a b')
    assert norm('Profit?') != norm('Profit')  # No fuzzy matching or answer-based disambiguation.
    for folder in ('contexts', 'public-inputs', 'private-labels'):
        (root / folder).mkdir(mode=0o700)
    info = HfApi().dataset_info('kensho/DocFinQA', revision=REVISION, token=False,
                              files_metadata=True)
    assert info.sha == REVISION
    upstream = {x.rfilename: x for x in info.siblings}
    for name, (size, expected) in FILES.items():
        assert upstream[name].size == size and upstream[name].lfs['sha256'] == expected
    write(root / 'upstream-plan.json', dict(dataset='kensho/DocFinQA', revision=REVISION,
        files={k: dict(bytes=v[0], sha256=v[1]) for k, v in FILES.items()},
        source='https://huggingface.co/datasets/kensho/DocFinQA',
        license_metadata=info.card_data.to_dict() if info.card_data else {},
        inference=False, execute_reference_programs=False))
    finqa = collections.defaultdict(list)
    input_hashes = {}
    for split in ('train', 'dev', 'test'):
        source = BOOT / f'public-inputs/finqa-{split}.json'
        input_hashes[split] = sha(source)
        for row in json.loads(source.read_text()):
            finqa[norm(row['question'])].append(dict(split=split, id=row['id']))
    fixed = json.loads((MEM / 'task-selection.json').read_text())
    target_ids = set(fixed['acquisition_ids'] + fixed['evaluation_ids'])
    selected = []
    overlap = []
    inventory = {}
    context_splits = collections.defaultdict(set)
    question_splits = collections.defaultdict(set)
    fixed_matches = collections.defaultdict(list)
    for split in ('dev', 'test', 'train'):
        name = split + '.json'
        source = Path(hf_hub_download('kensho/DocFinQA', name, repo_type='dataset',
            revision=REVISION, cache_dir=str(root / 'cache/hf/hub'), token=False))
        assert source.stat().st_size == FILES[name][0] and sha(source) == FILES[name][1]
        print(json.dumps(dict(event='download_verified', split=split)), flush=True)
        with source.open(encoding='utf-8') as stream:
            rows = json.load(stream)
        assert isinstance(rows, list) and rows
        stats = collections.Counter()
        candidates = []
        with (root / f'public-inputs/{split}.jsonl').open('x', encoding='utf-8') as public, \
             (root / f'private-labels/{split}.jsonl').open('x', encoding='utf-8') as private:
            for i, row in enumerate(rows):
                assert set(row) == {'Context', 'Question', 'Program', 'Answer'}, sorted(row)
                assert all(isinstance(row[k], str) for k in ('Context', 'Question'))
                assert row['Context'].strip() and row['Question'].strip()
                task_id = f'docfinqa/{split}/{i:05d}'
                ctx = digest(row['Context'])
                qhash = digest(norm(row['Question']))
                ctx_path = root / 'contexts' / (ctx + '.txt')
                if not ctx_path.exists():
                    with ctx_path.open('x', encoding='utf-8', newline='') as out:
                        out.write(row['Context'])
                    assert sha(ctx_path) == ctx
                context_splits[ctx].add(split)
                question_splits[qhash].add(split)
                matches = finqa.get(norm(row['Question']), [])
                item = dict(id=task_id, question=row['Question'], context_sha256=ctx,
                    context_path=f'contexts/{ctx}.txt', context_chars=len(row['Context']))
                line(public, item)
                line(private, dict(id=task_id, program=row['Program'], answer=row['Answer']))
                match = dict(id=task_id, context_sha256=ctx, question_sha256=qhash,
                             finqa_candidates=matches)
                overlap.append(match)
                stats['rows'] += 1
                stats['question_unique_match' if len(matches) == 1 else
                      'question_ambiguous_match' if matches else 'question_unmatched'] += 1
                stats['empty_reference_program'] += int(not row['Program'])
                stats['program_type_' + type(row['Program']).__name__] += 1
                stats['answer_type_' + type(row['Answer']).__name__] += 1
                matched_fixed = sorted(target_ids & {x['id'] for x in matches})
                if matched_fixed:
                    item = dict(item, matched_fixed_ids=matched_fixed,
                                match_ambiguous=len(matches) != 1, role='fixed_finqa_overlap')
                    selected.append(item)
                    for fid in matched_fixed:
                        fixed_matches[fid].append(task_id)
                elif split == 'dev':
                    candidates.append(item)
        if split == 'dev':
            # Predetermined order using only public question/context IDs, never lengths or labels.
            dev = sorted(candidates, key=lambda x: digest('docfinqa-capacity-dev-v1|' + x['id']))[:8]
            selected += [dict(x, role='development_capacity_probe') for x in dev]
        inventory[split] = dict(stats, source=str(source), source_sha256=FILES[name][1],
            public_sha256=sha(root / f'public-inputs/{split}.jsonl'),
            labels_sha256=sha(root / f'private-labels/{split}.jsonl'))
        print(json.dumps(dict(event='split_isolated', split=split, rows=len(rows))), flush=True)
        del rows, candidates
        gc.collect()
    write(root / 'overlap.json', dict(normalization='NFKC, whitespace collapse, casefold; no fuzzy/label matching',
        finqa_input_sha256=input_hashes, rows=overlap,
        fixed_matches={fid: fixed_matches.get(fid, []) for fid in sorted(target_ids)},
        exact_context_cross_split={h: sorted(s) for h, s in context_splits.items() if len(s) > 1},
        question_cross_split={h: sorted(s) for h, s in question_splits.items() if len(s) > 1},
        note='Candidate matches are not proven report identity. Ambiguity is retained. Same questions are not independent tasks.'))
    write(root / 'capacity-selection.json', dict(items=selected, gold_used_for_selection=False,
        policy='All normalized-question candidates for the prior fixed 48 FinQA IDs plus eight SHA-selected dev rows. No length filtering.'))
    # Tokenizer-only lower bound: no model/weights loaded, no truncation, no answer labels.
    models = [
        ('qwen', BASE / 'fin-skills-audit-20260921/cache/hf/hub/models--Qwen--Qwen2.5-Coder-14B-Instruct/snapshots/aedcc2d42b622764e023cf882b6652e646b95671'),
        ('mistral', BASE / 'fin-skills-campaign-model-transfer-20260923-v1/model-cache/hub/models--mistralai--Mistral-Nemo-Instruct-2407/snapshots/04d8a90549d23fc6bd7f642064003592df51e9b3'),
    ]
    capacity = []
    tokenizer_hashes = {}
    for family, snapshot in models:
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        tokenizer_hashes[family] = {p.name: sha(p) for p in snapshot.iterdir()
            if p.name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                          'added_tokens.json', 'vocab.json', 'merges.txt', 'tokenizer.model')}
        for item in selected:
            context = (root / item['context_path']).read_text(encoding='utf-8')
            messages = [dict(role='user', content='Read the financial report and answer the question.\n\n'
                        + context + '\n\nQuestion: ' + item['question'])]
            tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            capacity.append(dict(id=item['id'], family=family, input_tokens=len(tokens),
                reserve_tokens=3072, capacity=32768, fits_probe_capacity=len(tokens) + 3072 <= 32768,
                prompt_sha256=digest(messages[0]['content']), truncated=False))
        print(json.dumps(dict(event='capacity_probe_finished', family=family, rows=len(selected))), flush=True)
    write(root / 'capacity.json', dict(rows=capacity, tokenizer_hashes=tokenizer_hashes,
        selection_sha256=sha(root / 'capacity-selection.json'), model_loaded=False,
        scope='Tokenizer-only minimal user-prompt lower bound, not the full ReAct/tool/memory prompt and not a quality result. No truncation.'))
    write(root / 'corpus-receipt.json', dict(dataset='kensho/DocFinQA', revision=REVISION,
        inventory=inventory, unique_exact_contexts=len(context_splits),
        fixed_task_ids=len(target_ids), fixed_ids_with_candidates=len(fixed_matches),
        capacity_probe_rows=len(capacity), source_sha256=manifest['sha256'],
        artifacts={name: sha(root / name) for name in ('upstream-plan.json', 'overlap.json',
                     'capacity-selection.json', 'capacity.json')},
        inference=False, program_executed=False,
        limitations=['Public historical data; model pretraining exposure unknown.',
                    'Original Program/Answer fields preserved as labels; no reference Python executed.',
                    'Question matching is not a verified company/report mapping.',
                    'Corpus preparation is not a long-context agent experiment.']))


if __name__ == '__main__':
    root = Path(sys.argv[1]); start = time.monotonic()
    try:
        run(root)
    except Exception as exc:
        write(root / 'completion.json', dict(status='failed', error_type=type(exc).__name__,
            error=str(exc), traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-start))
        raise
    write(root / 'completion.json', dict(status='completed', elapsed_seconds=time.monotonic()-start,
                                         inference=False))
