"""Bounded LlamaIndex tool-organization ablation on existing authored contracts.

Uses the existing algorithm dispatcher, numerical reference and HF adapter unchanged.
This is a framework/module comparison, not a FinRobot trading-system reproduction.
"""
import asyncio
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np

from decision_tasks import PARAMETERS, RULES, cases, digest, execute, grade, reference
from finqa_reuse import llm_class, sha, template_messages, write

ARMS = ('generic', 'guidance', 'organized')
SYSTEM = '''Follow the supplied financial contract using the numerical tools. Each tool
uses the current case data; you supply all and only its advertised parameters, overriding
example defaults with the contract values. Historical experience is unavailable.
Follow the framework's ReAct syntax. Your final Answer must be exactly one JSON object:
{"values": [numbers in API order], "receipt_sha256": "exact hash from the used tool"}.
Do not use Markdown fences. Numerical tools use identical algorithms across conditions.
'''
MODELS = {
    'qwen': ('Qwen/Qwen2.5-Coder-14B-Instruct', 'aedcc2d42b622764e023cf882b6652e646b95671'),
    'mistral': ('mistralai/Mistral-Nemo-Instruct-2407', '04d8a90549d23fc6bd7f642064003592df51e9b3'),
}


def public_case(case, arm):
    # All arms receive the same rules once. Guidance repeats the flat rules;
    # organized repeats those same rules adjacent to descriptive tool names.
    return dict(id=case['id'], contract=case['policy'], data=case['data'],
                data_sha256=case['data_sha256'], observed_block=case['observed_block'],
                manual=[dict(method=k, rule=v) for k, v in RULES.items()],
                additional_guidance='\n'.join(RULES.values()) if arm == 'guidance' else '')


def tool_name(method, arm):
    return method if arm == 'organized' else 'function_'+str(list(PARAMETERS).index(method))


async def episode(case, arm, backend):
    from llama_index.core.agent.workflow import ReActAgent
    from llama_index.core.tools import FunctionTool
    llm = llm_class()(backend)
    dispatches, tools = [], []

    def bind(method):
        def call(parameters: dict) -> dict:
            started = time.monotonic()
            record = dict(method=method, parameters=parameters, tool_name=tool_name(method, arm))
            dispatches.append(record)
            try:
                receipt = execute(case, method, parameters)
                record['receipt'] = receipt
                return receipt
            except Exception as exc:
                record['error'] = dict(type=type(exc).__name__, message=str(exc))
                raise
            finally:
                record['seconds'] = time.monotonic()-started
        return call

    descriptions = {}
    for method, params in PARAMETERS.items():
        name = tool_name(method, arm)
        text = json.dumps(dict(implements=method, parameters=params))
        if arm == 'organized':
            text += '\n'+RULES[method]
        descriptions[name] = text
        tools.append(FunctionTool.from_defaults(fn=bind(method), name=name, description=text))
    agent = ReActAgent(tools=tools, llm=llm, system_prompt=SYSTEM,
                      streaming=False, timeout=600)
    public = public_case(case, arm)
    start = time.monotonic()
    final, error = None, None
    try:
        result = await agent.run(user_msg=json.dumps(public), max_iterations=8)
        final = str(result)
    except Exception as exc:
        if 'out of memory' in str(exc).lower():
            raise  # retain unfinished denominator on infrastructure failure
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    return dict(id=case['id'], arm=arm, phase=case['phase'], public=public,
                descriptions=descriptions, calls=llm._calls, dispatches=dispatches,
                final=final, error=error, elapsed_seconds=time.monotonic()-start)


def evaluate(case, row):
    final, format_error = {}, None
    try:
        final = json.loads(row['final'])
        assert isinstance(final, dict) and set(final) == {'values', 'receipt_sha256'}
        assert isinstance(final['values'], list) and isinstance(final['receipt_sha256'], str)
    except Exception as exc:
        format_error = type(exc).__name__
        final = {}
    used = [d for d in row['dispatches'] if d.get('receipt', {}).get('sha256') == final.get('receipt_sha256') and 'receipt' in d]
    chosen = used[-1] if used else {}
    result = grade(case, chosen, final, chosen.get('receipt'))
    first = row['dispatches'][0] if row['dispatches'] else {}
    return dict(**result, format_error=format_error,
                first_method_correct=first.get('method') == case['method'],
                first_parameters_correct=first.get('parameters') == case['parameters'])


def qualify(root):
    from llama_index.core import Settings
    Settings.tokenizer = lambda text: list(text.encode())
    data = cases(101)
    for case in data:
        receipt = execute(case, case['method'], case['parameters'])
        assert np.allclose(receipt['values'], reference(case), rtol=1e-12, atol=1e-14)
    fixture = data[0]
    logs = []
    for arm in ARMS:
        receipt = execute(fixture, fixture['method'], fixture['parameters'])
        scripted = iter([
            'Thought: Execute contract.\nAction: '+tool_name(fixture['method'], arm)+
            '\nAction Input: '+json.dumps(dict(parameters=fixture['parameters'])),
            'Thought: Use receipt.\nAnswer: '+json.dumps(dict(values=receipt['values'],receipt_sha256=receipt['sha256']))])
        def backend(messages):
            return dict(choices=[dict(message=dict(content=next(scripted)))])
        row = asyncio.run(episode(fixture, arm, backend))
        assert row['error'] is None, row['error']
        assert evaluate(fixture, row)['correct']
        assert len(row['dispatches']) == 1 and len(row['calls']) == 2
        wrong = dict(row, final='```json\n'+row['final']+'\n```')
        assert not evaluate(fixture, wrong)['correct']
        wrong = dict(row, final=json.dumps(dict(values=receipt['values'],receipt_sha256='wrong')))
        assert not evaluate(fixture, wrong)['correct']
        logs.append(row)
    write(root/'software-qualification.json', dict(passed=True, numerical_cases=32,
        actual_framework_fixtures=logs, strict_final_and_receipt_checks=True,
        scripted_only=True, quality_evidence=False))


def run(root):
    assert os.environ.get('SLURM_JOB_ID')
    manifest = json.loads((root/'manifest.json').read_text())
    assert all(sha(root/p) == h for p,h in manifest['sha256'].items())
    qual = Path(manifest['plan']['qualification_root'])
    assert json.loads((qual/'completion.json').read_text())['status'] == 'completed'
    qm = json.loads((qual/'manifest.json').read_text())
    assert all(qm['sha256'][p] == h for p,h in manifest['sha256'].items() if p.startswith('source/'))
    assert json.loads((qual/'software-qualification.json').read_text())['passed']
    family = manifest['plan']['family']; model, revision = MODELS[family]
    from transformers_chat import TransformersChat
    from llama_index.core import Settings
    from llama_index.core.agent.workflow import ReActAgent
    backend = TransformersChat(model, revision, max_tokens=512, seed=11)
    Settings.tokenizer = lambda text: backend.tokenizer.encode(text, add_special_tokens=False)
    data = cases(101)
    order = [(i,a) for i in range(32) for a in ARMS]
    random.Random(20260923).shuffle(order)
    write(root/'protocol.json', dict(family=family, model=model, revision=revision,
        case_hash=digest(data), ids=[c['id'] for c in data], system=SYSTEM, order=order,
        max_responses=6,max_output_tokens=512,max_framework_iterations=8,context_limit=32768,
        framework_version=importlib.metadata.version('llama-index-core'),
        framework_source_sha256=sha(Path(inspect.getfile(ReActAgent))),
        temperature=.1,seed=11,arms=ARMS,scoring='frozen decision_tasks.grade; strict final JSON',
        exposure='Same 32 authored development contracts; not independent external tasks.'))
    (root/'episodes').mkdir()
    rows = []
    for n,(i,arm) in enumerate(order):
        case = data[i]
        backend.calls=0; backend.seed=11+int(digest(case['id'])[:8],16)%100000000
        row = asyncio.run(episode(case,arm,backend))
        path=root/'episodes'/f'{i:02d}-{arm}.json'; write(path,row)
        rows.append(dict(id=case['id'],arm=arm,file=str(path.relative_to(root)),sha256=sha(path)))
        print(json.dumps(dict(completed=n+1,planned=96)),flush=True)
    write(root/'inference-receipt.json',dict(planned=96,rows=rows))
    subprocess.run([sys.executable,'-B',__file__,'score',str(root)],check=True)


def scored_rows(root, audit=False):
    data={c['id']:c for c in cases(101)}
    manifest=json.loads((root/'manifest.json').read_text())
    assert all(sha(root/p)==h for p,h in manifest['sha256'].items())
    receipt=json.loads((root/'inference-receipt.json').read_text())
    assert len(receipt['rows'])==receipt['planned']==96
    assert {(r['id'],r['arm']) for r in receipt['rows']}=={(c,a) for c in data for a in ARMS}
    output=[]
    for item in receipt['rows']:
        p=root/item['file']; assert sha(p)==item['sha256']; row=json.loads(p.read_text())
        case=data[row['id']]; assert row['id']==item['id'] and row['arm']==item['arm']
        assert row['public']==public_case(case,row['arm'])
        assert len(row['calls'])<=6
        if audit:
            for call in row['calls']:
                normalized,mapping=template_messages(call['messages'])
                assert normalized==call['template_messages'] and mapping==call['message_mapping']
                if 'response' in call:
                    u=call['response']['usage']; assert u['completion_tokens']<=512 and u['prompt_tokens']+512<=32768
            if row['calls']:
                assert json.loads(row['calls'][0]['messages'][-1]['content'])==row['public']
            for d in row['dispatches']:
                assert d['tool_name']==tool_name(d['method'],row['arm'])
                if 'receipt' in d:
                    assert execute(case,d['method'],d['parameters'])==d['receipt']
        usage=[c['response']['usage'] for c in row['calls'] if 'response' in c]
        output.append(dict(id=row['id'],arm=row['arm'],phase=case['phase'],**evaluate(case,row),
            error=row['error'],responses=len(usage),call_attempts=len(row['calls']),
            input_tokens=sum(u['prompt_tokens'] for u in usage),output_tokens=sum(u['completion_tokens'] for u in usage),
            tool_dispatches=len(row['dispatches']),tool_receipts=sum('receipt' in d for d in row['dispatches']),
            wall_seconds=row['elapsed_seconds']))
    return output


def score(root):
    write(root/'scores.json',dict(rows=scored_rows(root),receipt_sha256=sha(root/'inference-receipt.json')))


def audit(root):
    assert json.loads((root/'completion.json').read_text())['status']=='completed'
    rows=scored_rows(root,audit=True); scores=json.loads((root/'scores.json').read_text())
    assert rows==scores['rows'] and scores['receipt_sha256']==sha(root/'inference-receipt.json')
    groups={}
    for arm in ARMS:
        groups[arm]={}
        for phase in ('acquisition','later_evaluation'):
            selected=[r for r in rows if r['arm']==arm and r['phase']==phase]
            groups[arm][phase]=dict(planned=16,**{k:sum(r[k] for r in selected) for k in
                ('correct','method_correct','parameters_correct','numeric_correct','used_output',
                 'first_method_correct','first_parameters_correct','responses','call_attempts',
                 'input_tokens','output_tokens','tool_dispatches','tool_receipts','wall_seconds')},
                format_errors=sum(r['format_error'] is not None for r in selected),
                framework_errors=sum(r['error'] is not None for r in selected))
    write(root/'tools-verified-summary.json',dict(denominator=96,groups=groups,
        receipt_sha256=sha(root/'inference-receipt.json'),scores_sha256=sha(root/'scores.json'),
        manifest_sha256=sha(root/'manifest.json'),audit_source_sha256=sha(Path(__file__))))


if __name__=='__main__':
    mode,root=sys.argv[1],Path(sys.argv[2])
    if mode in ('score','audit'):
        globals()[mode](root)
    else:
        start=time.monotonic()
        try:
            {'qualify':qualify,'run':run}[mode](root)
        except Exception as exc:
            write(root/'completion.json',dict(status='failed',error=str(exc),traceback=traceback.format_exc()))
            raise
        write(root/'completion.json',dict(status='completed',elapsed_seconds=time.monotonic()-start))
        if mode=='run': audit(root)
