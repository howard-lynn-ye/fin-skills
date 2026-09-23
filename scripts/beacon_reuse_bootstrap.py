"""Beacon-only upstream acquisition and isolated dependency qualification, no LLM inference."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import traceback
import urllib.request


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as out:
        json.dump(value, out, indent=2, allow_nan=False)


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'fin-skills-research/1.0'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def run(root):
    assert os.environ.get('SLURM_JOB_ID') and root.parent == Path('/beacon-projects/radfm/wy891')
    assert root.resolve() == root
    started = time.monotonic()
    sources = {
        'finqa': ('czyssrs/FinQA', ['README.md', 'LICENSE', 'dataset/train.json',
            'dataset/dev.json', 'dataset/test.json', 'code/evaluate/evaluate.py']),
        'financebench': ('patronus-ai/financebench', ['README.md',
            'data/financebench_open_source.jsonl', 'data/financebench_document_information.jsonl']),
        'reflexion': ('noahshinn/reflexion', ['README.md', 'LICENSE']),
    }
    provenance = {}
    prior = Path(os.environ['FIN_REUSE_PRIOR']) if os.environ.get('FIN_REUSE_PRIOR') else None
    if prior:
        assert prior.parent == root.parent and prior.resolve() == prior
        frozen = json.loads((prior/'upstream-manifest.json').read_text())
        for name, item in frozen['sources'].items():
            for relative, detail in item['files'].items():
                source = prior/'upstream'/name/relative
                assert hashlib.sha256(source.read_bytes()).hexdigest() == detail['sha256']
                dest = root/'upstream'/name/relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
        provenance = frozen['sources']
        sources = {}
    for name, (repo, names) in sources.items():
        metadata = json.loads(fetch('https://api.github.com/repos/'+repo+'/commits/HEAD'))
        revision = metadata['sha']
        if name == 'reflexion':
            tree = json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1'))
            assert not tree.get('truncated')
            names += [p['path'] for p in tree['tree'] if p['type'] == 'blob'
                      and p['path'].endswith('.py') and p.get('size', 0) < 200000]
        item = dict(repository=repo, revision=revision, files={})
        write(root/'upstream'/name/'revision.json', item)
        for name_in_repo in names:
            data = fetch(f'https://raw.githubusercontent.com/{repo}/{revision}/{name_in_repo}')
            path = root/'upstream'/name/name_in_repo
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as out:
                out.write(data)
            item['files'][name_in_repo] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        provenance[name] = item
        print(json.dumps(dict(acquired=name, revision=revision, files=len(names))), flush=True)
    version = (frozen['package_requirements'][0].split('==')[1] if prior else
               json.loads(fetch('https://pypi.org/pypi/llama-index-core/json'))['info']['version'])
    write(root/'upstream-manifest.json', dict(sources=provenance,
        package_requirements=['llama-index-core=='+version], inference=False))
    runtime = Path(sys.executable).parent.parent
    subprocess.run([sys.executable, '-m', 'venv', str(root/'env')], check=True)
    py = root/'env/bin/python'
    purelib = subprocess.check_output([str(py), '-c',
        'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip()
    # Reuse the existing torch/transformers stack without mutating its installation.
    runtime_libs = [p for p in sys.path if p.startswith('/beacon-projects/radfm/')
                    and p.endswith('/site-packages') and Path(p).is_dir()]
    assert any('rfj-gpu' in p for p in runtime_libs)
    (Path(purelib)/'beacon_shared_runtime.pth').write_text('\n'.join(runtime_libs)+'\n')
    with (root/'logs/install.log').open('x') as log:
        subprocess.run([str(py), '-m', 'pip', 'install', '--disable-pip-version-check',
            '--report', str(root/'dependency-install.json'), 'llama-index-core=='+version],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    (root/'requirements-resolved.txt').write_bytes(subprocess.check_output([str(py), '-m', 'pip', 'freeze']))
    subprocess.run([str(py), '-B', str(Path(__file__)), str(root), '--qualify'], check=True)
    return dict(status='completed', elapsed_seconds=time.monotonic()-started, inference=False)


def qualify(root):
    from llama_index.core.agent.workflow import ReActAgent
    from llama_index.core.tools import FunctionTool
    import inspect
    spec = importlib.util.spec_from_file_location('finqa_official', root/'upstream/finqa/code/evaluate/evaluate.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    assert official.eval_program(['subtract(', '5829', '5735', ')', 'EOF'], []) == (0, 94.0)
    assert official.eval_program(['divide(', '1', '0', ')', 'EOF'], [])[0] == 1
    def add(a: float, b: float) -> float:
        """Add two supplied numbers."""
        return a+b
    tool = FunctionTool.from_defaults(fn=add)
    assert tool.call(a=2, b=3).raw_output == 5
    inventory = {}
    for split in ('train', 'dev', 'test'):
        rows = json.loads((root/f'upstream/finqa/dataset/{split}.json').read_text())
        assert len({r['id'] for r in rows}) == len(rows)
        inputs = [dict(id=r['id'], question=r['qa']['question'], pre_text=r['pre_text'],
                       post_text=r['post_text'], table=r['table']) for r in rows]
        write(root/f'public-inputs/finqa-{split}.json', inputs)
        inventory[split] = dict(count=len(rows), input_fields=sorted(inputs[0]),
            input_sha256=hashlib.sha256((root/f'public-inputs/finqa-{split}.json').read_bytes()).hexdigest())
    problems = []
    dev = json.loads((root/'upstream/finqa/dataset/dev.json').read_text())
    for r in dev:
        error, value = official.eval_program(official.program_tokenization(r['qa']['program']), r['table'])
        if error or value != r['qa']['exe_ans']:
            problems.append(dict(id=r['id'], execution_error=error, agrees=False))
    finance = [json.loads(line) for line in
               (root/'upstream/financebench/data/financebench_open_source.jsonl').read_text().splitlines() if line]
    write(root/'qualification.json', dict(status='passed', numerical_reference_checks=2,
        functiontool_executed=True, framework=inspect.getfile(ReActAgent),
        reactagent_signature=str(inspect.signature(ReActAgent)),
        finqa=inventory, dev_reference_disagreements=problems,
        financebench_count=len(finance), financebench_fields=sorted(finance[0]),
        limits=['No LLM calls: this is dependency/data qualification, not agent quality.',
                'Public labels are present in upstream storage but excluded from public-inputs.',
                'Same-user file separation is not adversarial process isolation.']))
    print(json.dumps(dict(qualification='passed', finqa=inventory,
                         financebench_count=len(finance), dev_reference_disagreements=len(problems))), flush=True)


if __name__ == '__main__':
    root = Path(sys.argv[1])
    if '--qualify' in sys.argv:
        qualify(root)
    else:
        try:
            result = run(root)
        except Exception as exc:
            write(root/'completion.json', dict(status='failed', error_type=type(exc).__name__,
                error=str(exc), traceback=traceback.format_exc()))
            raise
        write(root/'completion.json', result)
