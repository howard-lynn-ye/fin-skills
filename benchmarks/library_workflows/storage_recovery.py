"""Real subprocess termination around Store.save transactions; synthetic reliability evidence."""
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

from fin_skills.collect.model import Batch, Event, Watch
from fin_skills.collect.store import Store

WATCH = Watch('recovery-study','fixture','synthetic:recovery')


def event(i, revision=0, observed='2026-01-02T00:00:00Z'):
    return Event(str(i),'fixture','report','cash','https://example.invalid/report',
                 observed,data=dict(value=i,revision=revision))


def snapshot(path):
    start=time.monotonic()
    with Store(path) as store:
        result=dict(integrity=store.db.execute('PRAGMA integrity_check').fetchone()[0],
            counts={table:store.db.execute('SELECT count(*) FROM '+table).fetchone()[0]
                    for table in ('events','latest','alerts')},state=store.state(WATCH.id),
            pending=store.events(watch_id=WATCH.id,pending=True,limit=1000),
            latest=store.events(watch_id=WATCH.id,latest=True,limit=1000))
    result['open_and_check_seconds']=time.monotonic()-start
    return result


def child(path, phase):
    with Store(path) as store:
        def stop():
            print('BARRIER:'+phase,flush=True)
            os.kill(os.getpid(),signal.SIGSTOP)
            raise RuntimeError('Parent must terminate the stopped child, never resume')
        def trace(sql):
            if phase=='before_commit' and sql.strip().upper()=='COMMIT': stop()
            if phase=='during_batch' and sql.startswith('INSERT INTO events'):
                trace.count+=1
                if trace.count==2: stop()
        trace.count=0
        store.db.set_trace_callback(trace)
        if phase=='before_save': stop()
        store.save(WATCH,Batch(events=[event(1),event(2)],state={'cursor':2}),next_due=2)
        if phase=='after_commit': stop()
    raise RuntimeError('Requested barrier not encountered')


def write(path,data):
    with path.open('x') as out: json.dump(data,out,indent=2,allow_nan=False)


def run(root):
    assert os.environ.get('SLURM_JOB_ID') and sys.platform=='linux'
    assert root.resolve()==root and root.parent==Path('/beacon-projects/radfm/wy891')
    manifest=json.loads((root/'manifest.json').read_text())
    assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in manifest['sha256'].items())
    write(root/'protocol.json',dict(phases=['before_save','during_batch','before_commit','after_commit'],
        repeats=3,scales=[100,1000,10000],warm_query_repeats=7,
        termination='SIGSTOP at exact child barrier followed by parent SIGKILL',
        transaction='actual Store.save, SQLite trace callback observes but does not alter SQL',
        latest_semantics='last observed arrival, not maximum source publication timestamp',
        limits='Synthetic reliability and timing only; SIGKILL is not power-loss or network-filesystem failure.'))
    (root/'cases').mkdir();results=[]
    for repeat in range(3):
        for phase in ('before_save','during_batch','before_commit','after_commit'):
            path=root/'cases'/f'{repeat}-{phase}.sqlite'
            with Store(path) as store:
                store.put_watch(WATCH)
                store.save(WATCH,Batch(events=[event(0)],state={'cursor':0}),next_due=0)
            proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'child',str(path),phase],
                                  stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            # communicate cannot complete while SIGSTOP is active; read barrier with a timeout.
            import selectors
            selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
            try:
                ready=selector.select(timeout=30)
                barrier=proc.stdout.readline().strip() if ready else ''
                if barrier!='BARRIER:'+phase: raise RuntimeError('Missing barrier: '+barrier)
            finally:
                selector.close();proc.kill()
                stdout,stderr=proc.communicate(timeout=30)
            after=snapshot(path)
            committed=phase=='after_commit'
            expected=3 if committed else 1
            passed=(proc.returncode==-signal.SIGKILL and after['integrity']=='ok'
                and all(x==expected for x in after['counts'].values())
                and after['state']=={'cursor':2 if committed else 0}
                and {e['id'] for e in after['pending']}==({str(i) for i in range(3)} if committed else {'0'})
                and {e['id'] for e in after['latest']}==({str(i) for i in range(3)} if committed else {'0'}))
            result=dict(phase=phase,repeat=repeat,passed=passed,returncode=proc.returncode,
                        barrier=barrier,stderr=stderr,after=after)
            write(root/'cases'/f'{repeat}-{phase}.json',result);results.append(result)
    path=root/'revisions.sqlite'
    with Store(path) as store:
        store.put_watch(WATCH)
        a=event(0);b=event(0,1);older=event(0,0,'2026-01-01T00:00:00Z')
        added=[]
        for i,e in enumerate([a,a,b,b,older]):
            added.append(len(store.save(WATCH,Batch(events=[e],state={'cursor':i}),next_due=i)))
        latest=store.events(latest=True)
        revision=dict(inserted_per_batch=added,latest=latest,all=store.events(),
            passed=added==[1,0,1,0,1] and latest[0]['data']['revision']==0,
            interpretation='Adjacent identical content is deduplicated. An older revision arriving last becomes latest; callers need separate chronological selection.')
    write(root/'revisions.json',revision)
    scales=[]
    for size in (100,1000,10000):
        path=root/f'scale-{size}.sqlite';start=time.monotonic()
        with Store(path) as store:
            store.put_watch(WATCH)
            for first in range(0,size,100):
                store.save(WATCH,Batch(events=[event(i) for i in range(first,min(first+100,size))],
                                      state={'cursor':min(first+100,size)}),next_due=0)
            ingest=time.monotonic()-start
        start=time.monotonic()
        with Store(path) as store:
            reopened=time.monotonic()-start
            store.events(latest=True,limit=100)
            timings=[]
            for _ in range(7):
                start=time.perf_counter();rows=store.events(latest=True,limit=100)
                timings.append(time.perf_counter()-start)
            counts={t:store.db.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in ('events','latest','alerts')}
            ok=all(n==size for n in counts.values()) and len(rows)==100 and store.state(WATCH.id)=={'cursor':size}
        scales.append(dict(records=size,counts=counts,passed=ok,ingest_seconds=ingest,
            reopen_seconds=reopened,query_seconds=timings,query_median_seconds=statistics.median(timings),
            database_bytes=path.stat().st_size))
    write(root/'scales.json',scales)
    passed=all(r['passed'] for r in results+scales+[revision])
    write(root/'results.json',dict(passed=passed,crash_cases=len(results),crash_passed=sum(r['passed'] for r in results),
        revisions=revision,scales=scales,source_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()))
    if not passed: raise RuntimeError('Storage benchmark found a failing contract; preserve all results')


if __name__=='__main__':
    mode,path=sys.argv[1],Path(sys.argv[2])
    if mode=='child': child(path,sys.argv[3])
    else:
        start=time.monotonic()
        try: run(path)
        except Exception as exc:
            write(path/'completion.json',dict(status='failed',error=str(exc),traceback=traceback.format_exc()))
            raise
        write(path/'completion.json',dict(status='completed',seconds=time.monotonic()-start))
