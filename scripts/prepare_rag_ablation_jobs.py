"""Freeze new Beacon jobs locally; submission uses the existing exclusive dispatcher."""
import argparse
import json
from pathlib import Path
import shutil

BASE = '/beacon-projects/radfm/wy891'
LAUNCHER = r'''
#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?}"
r="${1:?}"
cd "$r"
export TMPDIR="$r/tmp" TMP="$r/tmp" TEMP="$r/tmp"
export XDG_CACHE_HOME="$r/cache/xdg" XDG_CONFIG_HOME="$r/cache/config" XDG_DATA_HOME="$r/cache/data" XDG_STATE_HOME="$r/cache/state"
export HF_HOME="$r/cache/hf" HF_DATASETS_CACHE="$r/cache/datasets" HF_MODULES_CACHE="$r/cache/modules"
export PIP_CACHE_DIR="$r/cache/pip" MPLCONFIGDIR="$r/cache/matplotlib" TORCH_HOME="$r/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$r/cache/inductor" TRITON_CACHE_DIR="$r/cache/triton" CUDA_CACHE_PATH="$r/cache/cuda" NUMBA_CACHE_DIR="$r/cache/numba"
export LLAMA_INDEX_CACHE_DIR="$r/cache/llamaindex" TIKTOKEN_CACHE_DIR="$r/cache/tiktoken" NLTK_DATA="$r/cache/nltk"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
'''

MODELS = {
    'qwen': ('Qwen/Qwen2.5-Coder-14B-Instruct','aedcc2d42b622764e023cf882b6652e646b95671',
             BASE+'/fin-skills-audit-20260921/cache/hf/hub'),
    'mistral': ('mistralai/Mistral-Nemo-Instruct-2407','04d8a90549d23fc6bd7f642064003592df51e9b3',
                BASE+'/fin-skills-campaign-model-transfer-20260923-v1/model-cache/hub')}


def prepare(name, family=None, afterok=None):
    bundle = Path('runs')/('beacon-campaign-'+name)
    bundle.mkdir(exist_ok=False)
    source = bundle/'source'; source.mkdir()
    for file in ('finqa_reuse.py','finqa_rag_ablation.py','transformers_chat.py'):
        shutil.copyfile(Path('benchmarks/agent_study')/file,source/file)
    shutil.copyfile('paper/FINQA_RAG_ABLATION_PROTOCOL_20260923.md',source/'protocol.md')
    shutil.copyfile('plugins/fin-market-data/skills/fundamental-and-macro-data/SKILL.md',source/'skill.md')
    # Minimal exact source snapshot, no generated substitutions or runtime dependency mutation.
    (source/'fin_skills').mkdir()
    shutil.copyfile('fin_skills/__init__.py',source/'fin_skills/__init__.py')
    shutil.copytree('fin_skills/rag',source/'fin_skills/rag',ignore=shutil.ignore_patterns('__pycache__'))
    root=BASE+'/fin-skills-campaign-'+name
    plan=dict(remote_root=root,name='fin-rag-'+(family or 'qualify'),gpu=bool(family),
              memory='64G' if family else '8G',time='04:00:00' if family else '00:10:00')
    if afterok: plan['afterok']=afterok
    if family:
        model,revision,cache=MODELS[family]
        plan.update(family=family,model=model,revision=revision,
                    qualification_root=BASE+'/fin-skills-campaign-finqa-rag-qualification-20260923-v1')
    template=LAUNCHER
    lines=[line for line in template.splitlines() if 'finqa_memory_study.py' not in line
           and not line.startswith('export HF_HUB_CACHE=')]
    if family:
        lines.append(f'export HF_HUB_CACHE={cache} HUGGINGFACE_HUB_CACHE={cache}')
        lines.append(f'export PYTHONPATH={plan["qualification_root"]}/deps')
    else:
        lines.append(f'{BASE}/fin-skills-campaign-reuse-bootstrap-20260923-v2/env/bin/python -B -m pip '
                     'install --target "$r/deps" --no-deps bm25s==0.3.11')
        lines.append('export PYTHONPATH="$r/deps"')
    lines.append(f'{BASE}/fin-skills-campaign-reuse-bootstrap-20260923-v2/env/bin/python -B '
                 f'"$r/source/finqa_rag_ablation.py" {"run" if family else "qualify"} "$r"')
    (bundle/'job.sh').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    (bundle/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(bundle)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--family',choices=MODELS)
    p.add_argument('--afterok');a=p.parse_args();prepare(a.name,a.family,a.afterok)
