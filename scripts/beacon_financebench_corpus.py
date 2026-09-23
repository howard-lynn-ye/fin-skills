"""Stage pinned full FinanceBench PDFs and qualify mature retrieval dependencies on Beacon."""
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request

REVISION = 'cc39aeb4afdf33909ee1412188bf89035950c2eb'
UPSTREAM = Path('/beacon-projects/radfm/wy891/fin-skills-campaign-reuse-bootstrap-20260923-v2')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as out:
        json.dump(value, out, indent=2, allow_nan=False)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url, maximum=64*2**20):
    req = urllib.request.Request(url, headers={'User-Agent': 'fin-skills-public-research/1.0'})
    with urllib.request.urlopen(req, timeout=90) as response:
        data = response.read(maximum+1)
    if len(data) > maximum:
        raise ValueError('Download exceeds prospective byte limit')
    return data


def extract(root, item):
    from pypdf import PdfReader
    path = root/item['local_path']
    assert digest(path) == item['sha256']
    reader = PdfReader(path)
    rows = []
    for number, page in enumerate(reader.pages):
        rows.append(dict(id=path.stem+':page:'+str(number), doc_name=path.stem,
                         page_num=number, text=page.extract_text() or '',
                         source=item['source_url'], pdf_sha256=item['sha256']))
    target = root/'pages'/(path.stem+'.json')
    write(target, rows)
    print(json.dumps(dict(file=target.name, pages=len(rows),
                          nonempty_pages=sum(bool(x['text'].strip()) for x in rows))))


def prepare(root):
    assert os.environ.get('SLURM_JOB_ID')
    assert root.resolve() == root and root.parent == Path('/beacon-projects/radfm/wy891')
    manifest = json.loads((root/'manifest.json').read_text())
    assert all(digest(root/p) == h for p, h in manifest['sha256'].items())
    # Corpus membership is every PDF in the pinned upstream repository, never gold pages.
    prior = Path(os.environ['FIN_CORPUS_PRIOR']) if os.environ.get('FIN_CORPUS_PRIOR') else None
    prior_downloads, prior_extractions = {}, {}
    if prior:
        assert prior.resolve()==prior and prior.parent==root.parent
        assert json.loads((prior/'completion.json').read_text())['status']=='completed'
        prior_protocol=json.loads((prior/'acquisition-protocol.json').read_text())
        prior_receipt=json.loads((prior/'corpus-receipt.json').read_text())
        assert digest(prior/'download-manifest.json')==prior_receipt['download_manifest_sha256']
        assert digest(prior/'extraction-manifest.json')==prior_receipt['extraction_manifest_sha256']
        prior_downloads={x['upstream_path']:x for x in json.loads((prior/'download-manifest.json').read_text())}
        prior_extractions={x['upstream_path']:x for x in json.loads((prior/'extraction-manifest.json').read_text())}
        tree_bytes=(prior/'upstream-tree.json').read_bytes()
        assert hashlib.sha256(tree_bytes).hexdigest()==prior_protocol['tree_sha256']
        assert prior_protocol['revision']==REVISION
    else:
        tree_bytes = fetch(f'https://api.github.com/repos/patronus-ai/financebench/git/trees/{REVISION}?recursive=1')
    tree = json.loads(tree_bytes)
    assert not tree.get('truncated')
    pdfs = sorted((x for x in tree['tree'] if x['type'] == 'blob'
                   and x['path'].lower().endswith('.pdf')), key=lambda x:x['path'])
    assert pdfs and len({Path(x['path']).name for x in pdfs}) == len(pdfs)
    assert sum(x['size'] for x in pdfs) <= 2*2**30
    assert all(x['size'] <= 64*2**20 for x in pdfs)
    versions = (dict(prior_protocol['package_versions']) if prior else
        {name: json.loads(fetch('https://pypi.org/pypi/'+name+'/json'))['info']['version']
         for name in ('pypdf', 'bm25s')})
    if prior:
        versions['cryptography']=json.loads(fetch('https://pypi.org/pypi/cryptography/json'))['info']['version']
    write(root/'acquisition-protocol.json', dict(revision=REVISION, planned_pdfs=pdfs,
        tree_sha256=hashlib.sha256(tree_bytes).hexdigest(), package_versions=versions,
        prior_corpus=str(prior) if prior else None,
        repair_policy='Reuse verified successful pages; retry only evidenced AES dependency failures' if prior else None,
        corpus_selection='all PDFs in pinned repository, independent of answer/evidence fields',
        page_numbering='zero based PDF physical page', extraction='pypdf default, no OCR',
        download_workers=4, max_pdf_bytes=64*2**20, max_total_pdf_bytes=2*2**30,
        extraction_timeout_per_pdf_seconds=120, framework_test='software only; no LLM inference'))
    (root/'upstream-tree.json').write_bytes(tree_bytes)
    subprocess.run([sys.executable, '-m', 'venv', str(root/'env')], check=True)
    py = root/'env/bin/python'
    purelib = subprocess.check_output([str(py), '-c',
        'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip()
    paths = [p for p in sys.path if p.startswith('/beacon-projects/radfm/')
             and p.endswith('/site-packages') and Path(p).is_dir()]
    assert paths
    (Path(purelib)/'beacon_shared_runtime.pth').write_text('\n'.join(paths)+'\n')
    with (root/'logs/install.log').open('x') as out:
        subprocess.run([str(py), '-m', 'pip', 'install', '--disable-pip-version-check',
            '--report', str(root/'dependency-install.json')]+
            [name+'=='+version for name,version in versions.items()],
            stdout=out, stderr=subprocess.STDOUT, check=True)
    (root/'requirements-resolved.txt').write_bytes(subprocess.check_output([str(py), '-m', 'pip', 'freeze']))
    subprocess.run([str(py), '-B', str(Path(__file__)), str(root), '--software'], check=True)
    (root/'pdfs').mkdir()
    (root/'pages').mkdir()

    def acquire(item):
        started = time.monotonic()
        url = f'https://raw.githubusercontent.com/patronus-ai/financebench/{REVISION}/'+urllib.parse.quote(item['path'])
        result = dict(upstream_path=item['path'], source_url=url, git_blob_sha1=item['sha'])
        try:
            if prior:
                old=prior_downloads[item['path']]
                assert old['status']=='downloaded', 'No new download attempts in repair job'
                data=(prior/old['local_path']).read_bytes()
                assert hashlib.sha256(data).hexdigest()==old['sha256']
                result['reused_from']=str(prior/old['local_path'])
            else:
                data = fetch(url)
            assert len(data) == item['size'] and data.startswith(b'%PDF-')
            assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == item['sha']
            path = root/'pdfs'/Path(item['path']).name
            with path.open('xb') as out:
                out.write(data)
            result.update(status='downloaded', local_path=path.relative_to(root).as_posix(),
                          bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        except Exception as exc:
            result.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        result['elapsed_seconds'] = time.monotonic()-started
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        downloads = []
        for row in pool.map(acquire, pdfs):
            downloads.append(row)
            print(json.dumps(dict(downloaded_or_failed=len(downloads),planned=len(pdfs),
                                  file=row['upstream_path'],status=row['status'])), flush=True)
    write(root/'download-manifest.json', downloads)
    # One bounded subprocess per PDF keeps malformed/enormous PDF failures explicit.
    extracted = []
    for item in downloads:
        row = dict(upstream_path=item['upstream_path'], status='missing_pdf')
        if prior:
            old=prior_extractions[item['upstream_path']]
            if old['status']=='extracted':
                assert digest(prior/old['file'])==old['sha256']
                shutil.copyfile(prior/old['file'],root/old['file'])
                extracted.append(dict(old,reused_from=str(prior/old['file'])))
                continue
            log_path=prior/'logs'/('extract-'+Path(item['upstream_path']).stem+'.log')
            log_text=log_path.read_text() if log_path.exists() else ''
            if 'DependencyError: cryptography>=3.1 is required for AES algorithm' not in log_text:
                extracted.append(dict(old,retained_failure=True,not_retried='No evidenced AES dependency failure'))
                continue
            row['repair_evidence']=dict(log=str(log_path),sha256=digest(log_path))
        if item['status'] == 'downloaded':
            stem = Path(item['local_path']).stem
            with (root/'logs'/('extract-'+stem+'.log')).open('x') as log:
                try:
                    proc = subprocess.run([str(py), '-B', str(Path(__file__)), str(root),
                        '--extract', item['local_path']], stdout=log, stderr=subprocess.STDOUT, timeout=120)
                    row.update(status='extracted' if proc.returncode == 0 else 'failed', returncode=proc.returncode)
                    if proc.returncode == 0:
                        pagefile = root/'pages'/(stem+'.json')
                        pages = json.loads(pagefile.read_text())
                        row.update(file=pagefile.relative_to(root).as_posix(),sha256=digest(pagefile),
                            pages=len(pages),nonempty_pages=sum(bool(p['text'].strip()) for p in pages))
                except subprocess.TimeoutExpired:
                    row.update(status='timeout', timeout_seconds=120)
        extracted.append(row)
        print(json.dumps(dict(extracted_or_failed=len(extracted),planned=len(pdfs),status=row['status'])), flush=True)
    write(root/'extraction-manifest.json', extracted)
    label_path = UPSTREAM/'upstream/financebench/data/financebench_open_source.jsonl'
    assert digest(label_path) == 'a5a2aa673e573e55675fc3c0f9aa38c1cf59d2abc91edb077534f71f10a71877'
    labels = [json.loads(line) for line in label_path.read_text().splitlines() if line]
    inputs = [{k:row[k] for k in ('financebench_id','question','company','doc_name')} for row in labels]
    write(root/'public-inputs.json', inputs)
    # Upstream Reflexion's non-Python dependency is acquired, not presented as an executed baseline.
    ref_revision = '218cf0ef1df84b05ce379dd4a8e47f17766733a0'
    ref_url = f'https://raw.githubusercontent.com/noahshinn/reflexion/{ref_revision}/alfworld_runs/reflexion_few_shot_examples.txt'
    ref_data = fetch(ref_url)
    (root/'reflexion_few_shot_examples.txt').write_bytes(ref_data)
    write(root/'corpus-receipt.json', dict(planned_pdfs=len(pdfs),
        downloaded=sum(x['status']=='downloaded' for x in downloads),
        extracted=sum(x['status']=='extracted' for x in extracted),
        pages=sum(x.get('pages',0) for x in extracted),
        nonempty_pages=sum(x.get('nonempty_pages',0) for x in extracted),
        questions=len(inputs),inputs_sha256=digest(root/'public-inputs.json'),
        download_manifest_sha256=digest(root/'download-manifest.json'),
        extraction_manifest_sha256=digest(root/'extraction-manifest.json'),
        reflexion_asset=dict(url=ref_url,sha256=hashlib.sha256(ref_data).hexdigest()),
        limitations=['PDF extraction can lose tables; no OCR or gold-evidence replacement.',
                    'Partial corpus failures remain missing, not silently dropped from question denominators.',
                    'Public historical labels are external authorship, not proof of unseen training data.',
                    'No answer generation, retrieval quality, or Reflexion effectiveness evaluated here.']))


def software(root):
    import bm25s
    from pypdf import PdfReader
    corpus = ['alpha capital expenditure', 'beta dividends paid', 'gamma cash flow']
    tokenized = bm25s.tokenize(corpus, stopwords=None)
    retriever = bm25s.BM25(k1=1.5, b=.75, method='lucene')
    retriever.index(tokenized)
    ids, scores = retriever.retrieve(bm25s.tokenize(['dividends'],stopwords=None), k=1)
    assert int(ids[0][0]) == 1 and float(scores[0][0]) > 0
    if os.environ.get('FIN_CORPUS_PRIOR'):
        from pypdf._crypt_providers import aes_cbc_encrypt
        assert len(aes_cbc_encrypt(b'0'*16,b'1'*16,b'2'*16))==16
    write(root/'software-qualification.json', dict(passed=True, actual_bm25s_index_and_retrieve=True,
        bm25s_version=importlib.metadata.version('bm25s'),pypdf_version=importlib.metadata.version('pypdf'),
        no_model_or_benchmark_quality_evidence=True))


if __name__ == '__main__':
    root = Path(sys.argv[1])
    if '--software' in sys.argv:
        software(root)
    elif '--extract' in sys.argv:
        path = sys.argv[sys.argv.index('--extract')+1]
        item = next(x for x in json.loads((root/'download-manifest.json').read_text()) if x.get('local_path')==path)
        extract(root, item)
    else:
        start = time.monotonic()
        try:
            prepare(root)
        except Exception as exc:
            write(root/'completion.json', dict(status='failed',error_type=type(exc).__name__,
                error=str(exc),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start))
            raise
        write(root/'completion.json', dict(status='completed',elapsed_seconds=time.monotonic()-start))
