"""Transport a frozen single-job source directory; persist before a single submission."""
import hashlib
import json
from pathlib import Path
import shlex
import sys

import paramiko
from scripts.beacon_campaign import connect, validate_root
from scripts.deploy_beacon_followup import remote_command


def main():
    bundle = Path(sys.argv[1])
    plan = json.loads((bundle/'plan.json').read_text())
    root = validate_root(plan['remote_root'])
    receipt = bundle/'deployment-started.json'
    with receipt.open('x') as f:
        json.dump(dict(remote_root=root, status='transport_started'), f)
    files = {p.relative_to(bundle).as_posix(): p.read_bytes() for p in (bundle/'source').rglob('*') if p.is_file()}
    files['job.sh'] = (bundle/'job.sh').read_bytes()
    files['source/submit_model_followup.py'] = Path('scripts/submit_model_followup.py').read_bytes()
    manifest = dict(plan=plan, sha256={n:hashlib.sha256(b).hexdigest() for n,b in files.items()})
    with (bundle/'manifest.json').open('x') as f:
        json.dump(manifest, f, indent=2)
    with connect('D:/ct_agent_vqa/.secrets/beacon_password.txt') as transport:
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            assert sftp.normalize(str(Path(root).parent).replace('\\','/')) == '/beacon-projects/radfm/wy891'
            sftp.mkdir(root, mode=0o700)
            assert sftp.normalize(root) == root
            directories = {'logs','tmp','cache','source'}
            for name in files:
                parts = name.split('/')[:-1]
                directories.update('/'.join(parts[:i]) for i in range(1,len(parts)+1))
            for name in sorted(directories, key=lambda x:(x.count('/'),x)):
                sftp.mkdir(root+'/'+name, mode=0o700)
            files['manifest.json'] = json.dumps(manifest,indent=2).encode()
            for name,data in files.items():
                with sftp.open(root+'/'+name,'wx') as out:
                    out.write(data)
            sftp.chmod(root+'/job.sh',0o700)
        command = ['sbatch','--parsable','--account=angliece','--partition=beacon','--qos=medium',
            '--nodes=1','--ntasks=1','--cpus-per-task=4','--mem='+plan.get('memory','16G'),
            '--time='+plan.get('time','00:30:00'),'--job-name='+plan['name'],'--chdir='+root,
            '--output='+root+'/logs/slurm-%j.out','--error='+root+'/logs/slurm-%j.err']
        if plan.get('gpu'):
            command.append('--gres=gpu:1')
        if plan.get('afterok'):
            dependency = str(plan['afterok'])
            assert dependency.isdigit(), 'One existing accepted prerequisite job ID is required'
            command.append('--dependency=afterok:'+dependency)
        command += [root+'/job.sh',root]
        script = ('import json,hashlib,sys; from pathlib import Path; r=Path('+repr(root)+'); '
            'm=json.loads((r/"manifest.json").read_text()); '
            'assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in m["sha256"].items()); '
            'sys.path.insert(0,str(r/"source")); from submit_model_followup import _dispatch; '
            'print(json.dumps(_dispatch(r,'+repr(command)+')))')
        result = remote_command(transport,shlex.join([
            '/beacon-projects/radfm/wy891/fin-skills-audit-20260921/env/bin/python','-B','-c',script]))
        with (bundle/'deployment-receipt.json').open('x') as f:
            f.write(result+'\n')
        print(result)


if __name__ == '__main__':
    main()
