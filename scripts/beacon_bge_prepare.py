"""Pin and stage the public BGE reranker; no inference and no Jev substitution claim."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def write(path,value):
    with path.open('x') as out:
        json.dump(value,out,indent=2,allow_nan=False)


def run(root):
    from huggingface_hub import HfApi,snapshot_download
    from transformers import AutoConfig,AutoTokenizer
    assert os.environ.get('SLURM_JOB_ID') and root.resolve()==root
    assert root.parent==Path('/beacon-projects/radfm/wy891')
    manifest=json.loads((root/'manifest.json').read_text())
    assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in manifest['sha256'].items())
    model='BAAI/bge-reranker-v2-m3'
    info=HfApi().model_info(model,token=False,files_metadata=True)
    revision=info.sha
    allowed={'config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json',
             'sentencepiece.bpe.model','model.safetensors','README.md'}
    files=[x for x in info.siblings if x.rfilename in allowed]
    assert {'config.json','model.safetensors'}<={x.rfilename for x in files}
    sizes=[dict(name=x.rfilename,size=x.size) for x in files]
    assert sum(x['size'] or 0 for x in sizes)<5*2**30
    write(root/'model-plan.json',dict(model=model,revision=revision,files=sizes,
        source='https://huggingface.co/BAAI/bge-reranker-v2-m3',token=False,
        role='Independent open reranker baseline, never called a Jev result',inference=False))
    path=Path(snapshot_download(model,revision=revision,allow_patterns=sorted(allowed),
        cache_dir=str(root/'cache/hf/hub'),token=False,max_workers=4))
    hashes={}
    for item in files:
        f=path/item.rfilename
        with f.open('rb') as stream:
            h=hashlib.file_digest(stream,'sha256').hexdigest()
        hashes[item.rfilename]=dict(bytes=f.stat().st_size,sha256=h)
    config=AutoConfig.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    tokens=tokenizer([['What is capital expenditure?','Capital expenditure was 10 million.']],
                     truncation=True,max_length=512,padding=True,return_tensors='pt')
    assert tokens['input_ids'].shape[0]==1 and 0<tokens['input_ids'].shape[1]<=512
    write(root/'model-receipt.json',dict(model=model,revision=revision,snapshot=str(path),
        files=hashes,config_architectures=config.architectures,tokenizer_fixture_tokens=int(tokens['input_ids'].shape[1]),
        tokenizer_only=True,model_loaded=False,inference=False))


if __name__=='__main__':
    root=Path(sys.argv[1]); start=time.monotonic()
    try:
        run(root)
    except Exception as exc:
        write(root/'completion.json',dict(status='failed',error_type=type(exc).__name__,error=str(exc),
            traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start))
        raise
    write(root/'completion.json',dict(status='completed',elapsed_seconds=time.monotonic()-start))
