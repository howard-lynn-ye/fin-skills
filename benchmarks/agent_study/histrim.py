"""Paper-equation implementation for single-sequence Qwen2/Mistral inference.

This is newly authored code, not an imported author checkpoint. It performs nested
historical-item selection before two decoder layers and physically shorter KV caches.
Batch size is one: offsets [0, length] describe packed buffers; no multi-request
VarLen kernel speedup is claimed. All comparisons use the same explicit decoder.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np


def unit(x):
    return x / max(float(np.linalg.norm(x)), 1e-12)


class DualRouter:
    def __init__(self, gate, utility, gate_bias=0., utility_bias=0., threshold=0.):
        self.gate, self.utility = np.asarray(gate), np.asarray(utility)
        self.gate_bias, self.utility_bias, self.threshold = gate_bias, utility_bias, threshold

    def scores(self, features):
        x = np.asarray(features, dtype=np.float64)
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        g = 8*x@self.gate + self.gate_bias
        u = 8*x@self.utility + self.utility_bias
        return (1/(1+np.exp(-np.clip(g, -60, 60)))) * u

    def as_dict(self):
        return dict(gate=self.gate.tolist(), utility=self.utility.tolist(),
                    gate_bias=self.gate_bias, utility_bias=self.utility_bias, threshold=self.threshold)

    @classmethod
    def fit(cls, features, labels, seed=97):
        x, y = np.asarray(features, dtype=float), np.asarray(labels, dtype=float)
        if len(x) < 4 or set(y) != {-1., 1.}:
            raise ValueError('router calibration requires positive and negative items')
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        # Dual ridge solve avoids a hidden_size squared allocation. Normalize each
        # independently fitted fold direction before their weighted average.
        directions = []
        for group in (np.arange(len(x)), np.arange(len(x))[::2], np.arange(len(x))[1::2]):
            a, b = x[group], y[group]
            directions.append(unit(a.T @ np.linalg.solve(a @ a.T + .1*np.eye(len(a)), b)))
        initial = unit(sum(directions)/len(directions))
        router = cls(initial.copy(), initial.copy())
        rng = np.random.default_rng(seed)
        def loss():
            return float(np.mean((router.scores(x)-y)**2))
        # Forward-only paired perturbations; project direction gradients tangent
        # to the unit sphere. The calibration objective is item relevance MSE.
        for step in range(24):
            for name in ('gate', 'utility'):
                old = getattr(router, name).copy()
                direction = unit(rng.normal(size=old.shape))
                direction = unit(direction-old*np.dot(old, direction))
                setattr(router, name, unit(old + .04*direction)); plus = loss()
                setattr(router, name, unit(old - .04*direction)); minus = loss()
                gradient = (plus-minus)/.08 * direction
                gradient -= old*np.dot(old, gradient)
                setattr(router, name, unit(old-.05*gradient))
        scores = router.scores(x)
        z = (scores-scores.mean())/(scores.std()+1e-8)
        candidates = np.linspace(-2., 2., 41)
        router.threshold = float(min(candidates, key=lambda t: (np.mean((z >= t) != (y > 0)), abs(t))))
        return router


class PackedChat:
    def __init__(self, base, max_tokens=384):
        self.base, self.model, self.tokenizer = base, base.model, base.tokenizer
        self.max_tokens = max_tokens
        config = self.model.config
        if config.model_type not in ('qwen2', 'mistral'):
            raise ValueError('only audited Qwen2/Mistral layer signatures supported')
        if getattr(config, 'sliding_window', None):
            raise ValueError('sliding-window models need a separate audited implementation')
        self.layers = self.model.model.layers
        self.prune_layers = (len(self.layers)//3, 2*len(self.layers)//3)
        self.routers = {}

    def encode(self, system, protected, items):
        blocks, spans = [], []
        for i, item in enumerate(items):
            block = f"<history_{i}>\n{item['text']}\n</history_{i}>"
            blocks.append(block)
        content = '\n'.join(blocks) + '\nCURRENT TASK (protected):\n' + protected
        prompt = self.tokenizer.apply_chat_template([
            dict(role='system', content=system), dict(role='user', content=content)],
            tokenize=False, add_generation_prompt=True)
        for block in blocks:
            start = prompt.index(block)
            spans.append((start, start+len(block)))
        encoded = self.tokenizer(prompt, return_offsets_mapping=True, return_tensors='pt',
                                 add_special_tokens=False)
        offsets = encoded.pop('offset_mapping')[0].tolist()
        assignments = np.full(len(offsets), -1, dtype=int)
        for i, (start, end) in enumerate(spans):
            for j, (a, b) in enumerate(offsets):
                if a >= start and b <= end and b > a:
                    assignments[j] = i
        return encoded.input_ids.to(self.model.device), assignments, prompt

    def _prefill(self, ids, assignments, mode, capture=False):
        import torch
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        h = self.model.model.embed_tokens(ids)
        positions = torch.arange(ids.shape[1], device=h.device)
        active = assignments.copy()
        records, captured, evicted = [], {}, []
        initial_history = int(np.sum(assignments >= 0))
        for index, layer in enumerate(self.layers):
            if index in self.prune_layers:
                item_ids = sorted(set(active[active >= 0].tolist()))
                features = np.array([h[0, torch.as_tensor(active == i, device=h.device)].float().mean(0).cpu().numpy()
                                     for i in item_ids])
                if capture:
                    captured[index] = dict(ids=item_ids, features=features)
                if mode != 'full' and item_ids:
                    fraction = .5 if index == self.prune_layers[0] else .25
                    budget = max(1, math.floor(initial_history*fraction))
                    if mode == 'histrim':
                        r = self.routers[index]
                        scores = r.scores(features)
                        z = (scores-scores.mean())/(scores.std()+1e-8)
                        order = sorted(range(len(item_ids)), key=lambda k: (-z[k], item_ids[k]))
                        eligible = [k for k in order if z[k] >= r.threshold]
                    else:
                        scores = np.zeros(len(item_ids)); z = scores
                        eligible = list(reversed(range(len(item_ids))))
                    # Whole items only. Threshold can underfill a budget. This is
                    # a shared maximum budget, NOT a claim of identical token use.
                    chosen, used = [], 0
                    for k in eligible:
                        size = int(np.sum(active == item_ids[k]))
                        if used+size <= budget:
                            chosen.append(item_ids[k]); used += size
                    keep = (active < 0) | np.isin(active, chosen)
                    before = positions.tolist()
                    h = h[:, torch.as_tensor(keep, device=h.device)].contiguous()
                    positions = positions[torch.as_tensor(keep, device=h.device)]
                    active = active[keep]
                    records.append(dict(layer=index, retained_items=chosen,
                        retained_original_positions=positions.tolist(), previous_positions=before,
                        scores=dict(zip(map(str,item_ids), map(float,scores))), token_budget=budget,
                        history_tokens=used, packed_offsets=[0, len(active)]))
            count = ids.shape[1]-h.shape[1]
            evicted.append(count)
            length = h.shape[1]
            mask = torch.full((length, length), torch.finfo(h.dtype).min, dtype=h.dtype, device=h.device)
            mask = torch.triu(mask, diagonal=1)[None, None]
            if mode == 'histrim' and count:
                mask[..., 0] += .1*math.log1p(count)
            pos = positions[None]
            embeddings = self.model.model.rotary_emb(h, pos)
            h = layer(h, attention_mask=mask, position_ids=pos, past_key_values=cache,
                      use_cache=True, cache_position=positions, position_embeddings=embeddings)
        logits = self.model.lm_head(self.model.model.norm(h[:, -1:])).float()
        return logits, cache, records, captured, evicted

    def calibrate(self, examples):
        import torch
        features = {i: [] for i in self.prune_layers}
        labels = {i: [] for i in self.prune_layers}
        with torch.inference_mode():
            for example in examples:
                ids, assignments, _ = self.encode(example['system'], example['protected'], example['items'])
                logits, cache, _, captured, _ = self._prefill(ids, assignments, 'full', capture=True)
                for layer, value in captured.items():
                    features[layer].extend(value['features'])
                    labels[layer].extend(example['labels'][i] for i in value['ids'])
                del logits, cache
        self.routers = {i: DualRouter.fit(features[i], labels[i]) for i in self.prune_layers}
        return {str(i): r.as_dict() for i, r in self.routers.items()}

    def __call__(self, system, protected, items, *, mode='full', seed=0):
        import torch
        if mode not in ('full', 'recency', 'histrim'):
            raise ValueError('unknown context intervention')
        ids, assignments, prompt = self.encode(system, protected, items)
        if ids.shape[1] + self.max_tokens > 8192:
            raise ValueError('8192-token study limit exceeded, no silent truncation')
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        torch.manual_seed(seed)
        output = []
        with torch.inference_mode():
            logits, cache, records, _, evicted = self._prefill(ids, assignments, mode)
            torch.cuda.synchronize(); prefill_seconds = time.monotonic()-start
            kv_bytes = sum(layer.keys.numel()*layer.keys.element_size()+layer.values.numel()*layer.values.element_size()
                           for layer in cache.layers)
            cache_lengths = [layer.get_seq_length() for layer in cache.layers]
            for step in range(self.max_tokens):
                token = torch.multinomial(torch.softmax(logits[0,-1]/.1, -1), 1)
                output.append(int(token.item()))
                if token.item() == self.tokenizer.eos_token_id:
                    break
                pos = torch.tensor([[ids.shape[1]+step]], device=ids.device)
                h = self.model.model.embed_tokens(token[None])
                embeddings = self.model.model.rotary_emb(h, pos)
                for index, layer in enumerate(self.layers):
                    length = cache.layers[index].get_seq_length()+1
                    mask = torch.zeros((1,1,1,length), dtype=h.dtype, device=h.device)
                    if mode == 'histrim' and evicted[index]:
                        mask[...,0] = .1*math.log1p(evicted[index])
                    h = layer(h, attention_mask=mask, position_ids=pos, past_key_values=cache,
                              use_cache=True, cache_position=pos[0], position_embeddings=embeddings)
                logits = self.model.lm_head(self.model.model.norm(h)).float()
        torch.cuda.synchronize()
        return dict(text=self.tokenizer.decode(output, skip_special_tokens=True),
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            prompt_tokens=int(ids.shape[1]), completion_tokens=len(output),
            elapsed_seconds=time.monotonic()-start, prefill_seconds=prefill_seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), prefill_kv_bytes=kv_bytes,
            cache_lengths=cache_lengths, selection=records, mode=mode)

    def equivalence(self, system, protected, items):
        import torch
        ids, assignments, _ = self.encode(system, protected, items)
        with torch.inference_mode():
            expected = self.model(ids, use_cache=False).logits[:, -1:].float()
            actual, cache, _, _, _ = self._prefill(ids, assignments, 'full')
            error = float((expected-actual).abs().max())
            same_top = bool(expected.argmax(-1).eq(actual.argmax(-1)).all())
        del cache
        return dict(max_logit_absolute_error=error, top_token_equal=same_top,
                    passed=same_top and error < .15, tolerance=.15)
