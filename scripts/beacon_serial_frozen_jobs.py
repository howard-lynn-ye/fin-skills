"""Execute two frozen, never-started answer jobs in one allocation after queue-limit rejection."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(sys.argv[1])
base=Path('/beacon-projects/radfm/wy891')
assert os.environ.get('SLURM_JOB_ID') and root.resolve()==root and root.parent==base
manifest=json.loads((root/'manifest.json').read_text())
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in manifest['sha256'].items())
results=[]
for item in manifest['plan']['targets']:
    child=Path(item['root'])
    assert child.resolve()==child and child.parent==base and not (child/'completion.json').exists()
    assert hashlib.sha256((child/'manifest.json').read_bytes()).hexdigest()==item['manifest_sha256']
    frozen=json.loads((child/'manifest.json').read_text())
    assert all(hashlib.sha256((child/p).read_bytes()).hexdigest()==h for p,h in frozen['sha256'].items())
    with (child/'shared-allocation.json').open('x') as f:
        json.dump(dict(parent=str(root),job_id=os.environ['SLURM_JOB_ID'],reason='Association submission limit; no scientific source changes'),f,indent=2)
    start=time.monotonic()
    with (child/'logs'/('shared-'+os.environ['SLURM_JOB_ID']+'.out')).open('x') as out:
        with (child/'logs'/('shared-'+os.environ['SLURM_JOB_ID']+'.err')).open('x') as err:
            result=subprocess.run(['bash',str(child/'job.sh'),str(child)],stdout=out,stderr=err,check=False)
    results.append(dict(root=str(child),returncode=result.returncode,seconds=time.monotonic()-start))
    print(json.dumps(results[-1]),flush=True)
with (root/'completion.json').open('x') as f:
    json.dump(dict(status='completed' if all(r['returncode']==0 for r in results) else 'failed',rows=results),f,indent=2)
if any(r['returncode'] for r in results): sys.exit(1)
