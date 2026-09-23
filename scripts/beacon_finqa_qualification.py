"""Run the two-model upstream agent qualification in one dynamic GPU allocation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

root=Path(sys.argv[1])
assert root.resolve()==root and root.parent==Path('/beacon-projects/radfm/wy891')
assert os.environ.get('SLURM_JOB_ID')
upstream=Path('/beacon-projects/radfm/wy891/fin-skills-campaign-reuse-bootstrap-20260923-v2')
assert json.loads((upstream/'completion.json').read_text())['status']=='completed'
assert json.loads((upstream/'qualification.json').read_text())['status']=='passed'
manifest=json.loads((root/'manifest.json').read_text())
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in manifest['sha256'].items())
started=time.monotonic()
commands=[]
status='failed'
error=None
try:
    import torch
    assert torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 40*2**30
    with (root/'preflight.json').open('x') as out:
        json.dump(dict(device=torch.cuda.get_device_name(0), gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
            torch_version=torch.__version__,slurm_job_id=os.environ['SLURM_JOB_ID']),out,indent=2)
    for family,model,revision,cache in (
        ('qwen','Qwen/Qwen2.5-Coder-14B-Instruct','aedcc2d42b622764e023cf882b6652e646b95671',
         '/beacon-projects/radfm/wy891/fin-skills-audit-20260921/cache/hf/hub'),
        ('mistral','mistralai/Mistral-Nemo-Instruct-2407','04d8a90549d23fc6bd7f642064003592df51e9b3',
         '/beacon-projects/radfm/wy891/fin-skills-campaign-model-transfer-20260923-v1/model-cache/hub')):
        os.environ.update(HF_HUB_CACHE=cache,HUGGINGFACE_HUB_CACHE=cache)
        for mode in ('infer','score'):
            command=[sys.executable,'-B',str(root/'source/finqa_reuse.py'),mode,
                '--upstream',str(upstream),'--output',str(root/'results'/family),
                '--model',model,'--revision',revision]
            result=subprocess.run(command,check=False)
            commands.append(dict(family=family,mode=mode,returncode=result.returncode))
            if result.returncode: raise RuntimeError('Qualification command failed: '+family+'/'+mode)
    status='completed'
except Exception as exc:
    error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
finally:
    with (root/'completion.json').open('x') as out:
        json.dump(dict(status=status,commands=commands,error=error,elapsed_seconds=time.monotonic()-started),out,indent=2)
if status!='completed': sys.exit(1)
