"""Thin FinQA adapter around the installed LlamaIndex ReActAgent and official executor.

Inference only opens the stripped public-input file. Scoring is a separate invocation
after an immutable inference receipt. File separation is not an adversarial sandbox.
"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as out:
        json.dump(value, out, indent=2, allow_nan=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def template_messages(messages):
    """Preserve native retry text while satisfying strict alternating-role chat templates."""
    normalized, mapping = [], []
    for i, message in enumerate(messages):
        if normalized and normalized[-1]['role'] == message['role']:
            normalized[-1]['content'] += '\n\n'+message['content']
            mapping[-1].append(i)
        else:
            normalized.append(dict(message))
            mapping.append([i])
    assert all(normalized[j]['content'] == '\n\n'.join(messages[i]['content'] for i in indices)
               for j, indices in enumerate(mapping))
    return normalized, mapping


def evaluator(upstream):
    source = upstream/'upstream/finqa/code/evaluate/evaluate.py'
    manifest = json.loads((upstream/'upstream-manifest.json').read_text())
    assert sha(source) == manifest['sources']['finqa']['files']['code/evaluate/evaluate.py']['sha256']
    spec = importlib.util.spec_from_file_location('finqa_official', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def llm_class():
    from llama_index.core.llms import CustomLLM, LLMMetadata, CompletionResponse, ChatResponse, ChatMessage
    from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback
    from pydantic import PrivateAttr

    class BeaconLLM(CustomLLM):
        _backend = PrivateAttr()
        _calls = PrivateAttr(default_factory=list)
        _maximum = PrivateAttr(default=6)

        def __init__(self, backend, **kwargs):
            super().__init__(**kwargs)
            self._backend = backend

        @property
        def metadata(self):
            return LLMMetadata(context_window=32768, num_output=512,
                               model_name='beacon-pinned-hf', is_chat_model=True)

        def generate(self, messages):
            if len(self._calls) >= self._maximum:
                raise RuntimeError('PRESPECIFIED_MODEL_CALL_LIMIT')
            before = time.monotonic()
            normalized, mapping = template_messages(messages)
            record = dict(messages=messages, template_messages=normalized, message_mapping=mapping)
            self._calls.append(record)
            try:
                response = self._backend(normalized)
                record.update(response=response, wall_seconds=time.monotonic()-before)
                return response['choices'][0]['message']['content']
            except Exception as exc:
                record.update(error_type=type(exc).__name__, error=str(exc),
                              wall_seconds=time.monotonic()-before)
                raise

        @llm_completion_callback()
        def complete(self, prompt, **kwargs):
            return CompletionResponse(text=self.generate([dict(role='user', content=prompt)]))

        @llm_completion_callback()
        def stream_complete(self, prompt, **kwargs):
            raise NotImplementedError('This study uses non-streaming ReActAgent')
            yield

        @llm_chat_callback()
        def chat(self, messages, **kwargs):
            public = [dict(role=m.role.value, content=m.content or '') for m in messages]
            return ChatResponse(message=ChatMessage(role='assistant', content=self.generate(public)))

        async def achat(self, messages, **kwargs):
            return self.chat(messages, **kwargs)

    return BeaconLLM


SYSTEM = '''Solve the supplied financial report question using its table and text.
Use execute_program to check your arithmetic. Its only input is a FinQA program string.
Allowed binary operations: add, subtract, multiply, divide, exp, greater.
Allowed table operations: table_sum, table_average, table_max, table_min; their first
argument is the exact row label and the second is none. Use #0, #1 to refer to preceding
step results. Constants include const_100 and const_1. Example syntax (unrelated values):
subtract(12, 10), divide(#0, 10), multiply(#1, const_100).
Keep units and the question's percentage convention explicit in your reasoning.
Follow the ReAct format supplied by the framework. Your final Answer must be exactly
one JSON object with a single key "program" containing the complete program string.
Do not put the final JSON inside a Markdown fence. No source beyond this report is available.
'''


async def episode(case, backend, official):
    from llama_index.core.agent.workflow import ReActAgent
    from llama_index.core.tools import FunctionTool
    llm = llm_class()(backend)
    receipts = []

    def execute_program(program: str) -> dict:
        """Execute a FinQA arithmetic program against the supplied report table; no answer labels."""
        tokens = official.program_tokenization(program)
        if len(tokens) > 81:
            raise ValueError('At most 20 operations are allowed')
        start = time.monotonic()
        invalid, result = official.eval_program(tokens, case['table'])
        receipt = dict(program=program, invalid=invalid, result=result,
                       seconds=time.monotonic()-start)
        receipts.append(receipt)
        return receipt

    agent = ReActAgent(tools=[FunctionTool.from_defaults(fn=execute_program)],
        llm=llm, system_prompt=SYSTEM, streaming=False, timeout=600)
    start = time.monotonic()
    final, error = None, None
    try:
        output = await agent.run(user_msg=json.dumps(case, ensure_ascii=False), max_iterations=8)
        final = str(output)
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
    return dict(id=case['id'], final=final, error=error, calls=llm._calls,
                tool_receipts=receipts, elapsed_seconds=time.monotonic()-start)


def software_qualification(official):
    from llama_index.core import Settings
    Settings.tokenizer = lambda text: list(text.encode('utf-8'))
    responses = iter([
        'Thought: I will compute the change.\nAction: execute_program\nAction Input: {"program":"subtract(12, 10)"}',
        'Thought: The tool returned 2.\nAnswer: {"program":"subtract(12, 10)"}',
    ])
    def scripted(messages):
        return dict(choices=[dict(message=dict(content=next(responses)))])
    row = asyncio.run(episode(dict(id='software-test', question='Change from 10 to 12?',
                                   pre_text=[], post_text=[], table=[]), scripted, official))
    assert row['error'] is None, row['error']
    assert len(row['tool_receipts']) == 1 and row['tool_receipts'][0]['result'] == 2
    assert json.loads(row['final'])['program'] == 'subtract(12, 10)'
    retry_responses = iter([
        'Thought: I know the answer.\nAction: None\nAnswer: {"program":"subtract(12, 10)"}',
        'Thought: I will compute.\nAction: execute_program\nAction Input: {"program":"subtract(12, 10)"}',
        'Thought: I have the result.\nAnswer: {"program":"subtract(12, 10)"}',
    ])
    def retry_backend(messages):
        roles=[m['role'] for m in messages]
        body=roles[1:] if roles[0]=='system' else roles
        assert all(role==('user' if i%2==0 else 'assistant') for i,role in enumerate(body)), roles
        return dict(choices=[dict(message=dict(content=next(retry_responses)))])
    retried = asyncio.run(episode(dict(id='software-retry-test',question='Change from 10 to 12?',
        pre_text=[],post_text=[],table=[]),retry_backend,official))
    assert retried['error'] is None, retried['error']
    assert json.loads(retried['final'])['program']=='subtract(12, 10)'
    assert len(retried['calls'])==3 and len(retried['tool_receipts'])==1
    return dict(passed=True, actual_reactagent=True, actual_functiontool=True,
                retry_template_qualification=True, retry_calls=3,
                scripted_software_test_only=True, model_quality_evidence=False)


def infer(args):
    assert os.environ.get('SLURM_JOB_ID')
    official = evaluator(args.upstream)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'software-qualification.json', software_qualification(official))
    datafile = args.upstream/'public-inputs/finqa-dev.json'
    data = json.loads(datafile.read_text())
    selected = sorted(data,key=lambda x:hashlib.sha256(('finqa-dev-calibration-v1|'+x['id']).encode()).hexdigest())[:4]
    assert all(set(r) == {'id','question','pre_text','post_text','table'} for r in selected)
    from transformers_chat import TransformersChat
    from llama_index.core.agent.workflow import ReActAgent
    import inspect
    write(args.output/'protocol.json', dict(model=args.model, revision=args.revision, seed=11,
        selected_ids=[r['id'] for r in selected], split='dev', input_sha256=sha(datafile),
        framework_version=importlib.metadata.version('llama-index-core'),
        framework_source=inspect.getfile(ReActAgent), framework_sha256=sha(Path(inspect.getfile(ReActAgent))),
        adapter_sha256=sha(Path(__file__)), system=SYSTEM, max_model_calls=6, max_output_tokens=512,
        context_limit=32768, max_framework_iterations=8, temperature=.1, parser='upstream ReAct + exact final JSON',
        template_adapter='Join adjacent same-role messages with two newlines, preserve every text and raw-to-template mapping; no parser changes.',
        study='development calibration only; no FinSkills treatment or formal test score'))
    backend = TransformersChat(args.model,args.revision,max_tokens=512,seed=11)
    from llama_index.core import Settings
    Settings.tokenizer = lambda text: backend.tokenizer.encode(text, add_special_tokens=False)
    records = []
    for i,case in enumerate(selected):
        backend.calls=0
        backend.seed=11+100*i
        row = asyncio.run(episode(case,backend,official))
        path = args.output/f'episode-{i}.json'
        write(path,row)
        records.append(dict(id=case['id'],file=path.name,sha256=sha(path)))
        print(json.dumps(dict(completed=i+1,planned=len(selected))),flush=True)
    write(args.output/'inference-receipt.json',dict(planned=4,completed=len(records),records=records))


def score(args):
    official = evaluator(args.upstream)
    receipt = json.loads((args.output/'inference-receipt.json').read_text())
    gold = {r['id']:r for r in json.loads((args.upstream/'upstream/finqa/dataset/dev.json').read_text())}
    results=[]
    for item in receipt['records']:
        path=args.output/item['file']; assert sha(path)==item['sha256']
        row=json.loads(path.read_text()); target=gold[row['id']]
        outcome=dict(id=row['id'],execution_correct=False,program_correct=False,format_error=None,
                     scorer_error=None,actual_calls=len(row['calls']),tool_calls=len(row['tool_receipts']))
        try:
            parsed=json.loads(row['final'])
            assert isinstance(parsed,dict) and set(parsed)=={'program'} and isinstance(parsed['program'],str)
            tokens=official.program_tokenization(parsed['program'])
            if len(tokens)>81: raise ValueError('Program exceeds operation budget')
        except Exception as exc:
            outcome['format_error']=type(exc).__name__
        else:
            try:
                invalid,result=official.eval_program(tokens,target['table'])
                outcome['execution_correct']=invalid==0 and result==target['qa']['exe_ans']
                outcome['program_correct']=bool(official.equal_program(
                    official.program_tokenization(target['qa']['program']),tokens))
            except Exception as exc:
                outcome['scorer_error']=dict(type=type(exc).__name__,message=str(exc))
        results.append(outcome)
    write(args.output/'scores.json',dict(planned=4,completed=len(results),rows=results,
        execution_correct=sum(x['execution_correct'] for x in results),
        program_correct=sum(x['program_correct'] for x in results),
        official_evaluator_sha256=sha(args.upstream/'upstream/finqa/code/evaluate/evaluate.py'),
        inference_receipt_sha256=sha(args.output/'inference-receipt.json'),
        limits='Development qualification, not held-out result; failed/unscorable rows retained.'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['infer','score'])
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model')
    parser.add_argument('--revision')
    args=parser.parse_args()
    globals()[args.mode](args)
