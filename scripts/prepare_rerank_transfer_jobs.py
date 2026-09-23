"""Freeze pinned downstream reranking jobs without changing existing runs."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile

from prepare_rag_ablation_jobs import BASE,MODELS,LAUNCHER


def main():
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--family',choices=MODELS)
    p.add_argument('--afterok',required=True);a=p.parse_args()
    bundle=Path('runs')/('beacon-campaign-'+a.name);bundle.mkdir(exist_ok=False)
    source=bundle/'source';source.mkdir()
    for f in ('finqa_reuse.py','finqa_rag_ablation.py','finqa_rerank_transfer.py','transformers_chat.py'):
        shutil.copyfile(Path('benchmarks/agent_study')/f,source/f)
    shutil.copyfile('plugins/fin-market-data/skills/fundamental-and-macro-data/SKILL.md',source/'skill.md')
    shutil.copyfile('paper/FINQA_RERANK_TRANSFER_PROTOCOL_20260923.md',source/'protocol.md')
    with tarfile.open(source/'library.tar.gz','w:gz') as archive:
        for f in sorted(Path('fin_skills').rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc': archive.add(f,arcname=f.as_posix())
    plan=dict(remote_root=BASE+'/fin-skills-campaign-'+a.name,name='fin-rerank-'+(a.family or 'rank'),
              gpu=True,memory='64G',time='04:00:00',afterok=a.afterok)
    if a.family:
        model,revision,cache=MODELS[a.family]
        plan.update(family=a.family,model=model,revision=revision,
            ranking_root=BASE+'/fin-skills-campaign-finqa-rerank-rank-20260923-v1')
    else:
        configs=[{'name': 'kev-4b', 'model_id': 'kev', 'parameters': {'checkpoint': '/beacon-projects/radfm/wy891/fin-skills-campaign-kev-preparation-20260923-v1/cache/hf/hub/models--jaredpalmer--kev-4b/snapshots/485ace8703592fcf405488b262449990824cfed1', 'revision': '485ace8703592fcf405488b262449990824cfed1', 'base_checkpoint': '/beacon-projects/radfm/wy891/fin-skills-campaign-kev-preparation-20260923-v1/cache/hf/hub/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b', 'base_revision': '1001bb4d826a52d1f399e183466143f4da7b741b', 'device': 'cuda', 'dtype': 'bfloat16'}}]
        (source/'kev-config.json').write_text(json.dumps(next(c for c in configs if c['name']=='kev-4b'),indent=2))
    template=LAUNCHER
    lines=[l for l in template.splitlines() if 'finqa_memory_study.py' not in l and not l.startswith('export HF_HUB_CACHE=')]
    lines+=['mkdir "$r/source/library"','tar -xzf "$r/source/library.tar.gz" -C "$r/source/library"',
            f'export PYTHONPATH="$r/source/library:{BASE}/fin-skills-campaign-finqa-rag-qualification-20260923-v1/deps"']
    if a.family:
        lines.append(f'export HF_HUB_CACHE={cache} HUGGINGFACE_HUB_CACHE={cache}')
        env='fin-skills-campaign-reuse-bootstrap-20260923-v2'
    else: env='fin-skills-campaign-kev-preparation-20260923-v1'
    lines.append(f'{BASE}/{env}/env/bin/python -B "$r/source/finqa_rerank_transfer.py" '
                 f'{"answer" if a.family else "rank"} "$r"')
    (bundle/'job.sh').write_text('\n'.join(lines)+'\n',newline='\n')
    (bundle/'plan.json').write_text(json.dumps(plan,indent=2))
    print(bundle)


if __name__=='__main__': main()
