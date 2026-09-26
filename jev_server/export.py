import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import openvino as ov
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel
from tokenizers import Tokenizer
from .api import SystemOneRequest, to_record
from .encoding import encode

class QwenGraph(nn.Module):

    def __init__(self, backbone, length):
        super().__init__()
        self.backbone = backbone
        cfg = backbone.config
        self.length = length
        self.heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.dim = cfg.head_dim
        cos, sin = backbone.rotary_emb(torch.zeros(1, length, cfg.hidden_size), torch.arange(length)[None])
        self.register_buffer('cos', cos[:, None])
        self.register_buffer('sin', sin[:, None])
        self.register_buffer('mask', torch.triu(torch.full((length, length), -10000.0), 1)[None, None])

    def rotary(self, x):
        a, b = x.chunk(2, -1)
        return x * self.cos + torch.cat((-b, a), -1) * self.sin

    def hidden(self, input_ids):
        h = self.backbone.embed_tokens(input_ids)
        for layer in self.backbone.layers:
            n = layer.input_layernorm(h)
            attn = layer.self_attn
            q = attn.q_proj(n).reshape(1, self.length, self.heads, self.dim).transpose(1, 2)
            k = attn.k_proj(n).reshape(1, self.length, self.kv_heads, self.dim).transpose(1, 2)
            v = attn.v_proj(n).reshape(1, self.length, self.kv_heads, self.dim).transpose(1, 2)
            q, k = (self.rotary(attn.q_norm(q)), self.rotary(attn.k_norm(k)))
            k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
            v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
            weights = torch.softmax(q @ k.transpose(-2, -1) * self.dim ** (-0.5) + self.mask, dim=-1)
            a = (weights @ v).transpose(1, 2).reshape(1, self.length, -1)
            h = h + attn.o_proj(a)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        h = self.backbone.norm(h)
        return h

    def forward(self, input_ids, leaf_map):
        return leaf_map @ self.hidden(input_ids)


class DecisionGraph(QwenGraph):
    def __init__(self, backbone, head, length):
        super().__init__(backbone, length)
        cfg = backbone.config
        self.q = nn.Linear(cfg.hidden_size, head['q.weight'].shape[0])
        self.k = nn.Linear(cfg.hidden_size, head['k.weight'].shape[0])
        self.q.load_state_dict({k[2:]: v for k, v in head.items() if k.startswith('q.')})
        self.k.load_state_dict({k[2:]: v for k, v in head.items() if k.startswith('k.')})
        self.temperature = 1.0

    def forward(self, input_ids, decide_map, option_map):
        h = self.hidden(input_ids)
        logits = (self.k(option_map @ h) @ self.q(decide_map @ h).transpose(-2, -1)).squeeze(-1)
        logits = logits * self.q.out_features ** (-0.5) / self.temperature
        logits = logits + (1 - option_map.sum(-1)) * -10000.0
        return torch.softmax(logits, dim=-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('artifacts/kev-fp16'))
    ap.add_argument('--length', type=int, default=256)
    ap.add_argument('--max-options', type=int, default=32)
    ap.add_argument('--pins', type=Path, default=Path('assets.lock.json'))
    args = ap.parse_args()
    torch.set_num_threads(4)
    pins = json.loads(args.pins.read_text())
    checkpoint = Path(snapshot_download(pins['checkpoint']['repo'], revision=pins['checkpoint']['revision']))
    for name, expected in pins['checkpoint']['files'].items():
        if hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'checkpoint digest mismatch: {name}')
    base = snapshot_download(pins['base']['repo'], revision=pins['base']['revision'], allow_patterns=['*.json', '*.safetensors', '*.txt'])
    backbone = AutoModel.from_pretrained(base, dtype=torch.float32, attn_implementation='eager', trust_remote_code=False).eval()
    adapter_config = json.loads((checkpoint / 'adapter_config.json').read_text())
    adapter = load_file(checkpoint / 'adapter_model.safetensors')
    scale = adapter_config['lora_alpha'] / adapter_config['r']
    with torch.no_grad():
        for name, a in adapter.items():
            if '.lora_A.' not in name:
                continue
            b = adapter[name.replace('.lora_A.', '.lora_B.')]
            module = name.removeprefix('base_model.model.').split('.lora_A.')[0]
            backbone.get_submodule(module).weight.add_(b.float() @ a.float() * scale)
    saved = torch.load(checkpoint / 'head.pt', map_location='cpu', weights_only=True)
    graph = DecisionGraph(backbone, saved['head'], args.length).eval()
    graph.temperature = saved.get('temperature', 1.0)
    tokenizer = Tokenizer.from_file(str(checkpoint / 'tokenizer.json'))
    fixtures = json.loads(Path('examples/fixtures.json').read_text())
    reference = []
    with torch.inference_mode():
        for fixture in fixtures:
            rec, _ = to_record(SystemOneRequest.model_validate(fixture['request']))
            for question in rec['questions']:
                inputs, count = encode(tokenizer, rec['state'], question, args.length, args.max_options)
                tensors = {k: torch.from_numpy(v) for k, v in inputs.items()}
                actual = graph(**tensors).numpy()
                ids = tensors['input_ids'][:, :count]
                h = backbone(ids, use_cache=False).last_hidden_state
                hd = tensors['decide_map'][:, :, :count] @ h
                ho = tensors['option_map'][:, :len(question['options']), :count] @ h
                logits = (graph.k(ho) @ graph.q(hd).transpose(-2, -1)).squeeze(-1)
                expected = torch.softmax(logits * graph.q.out_features ** (-0.5) / graph.temperature, -1).numpy()
                error = float(np.max(np.abs(actual[0, :len(question['options'])] - expected[0])))
                if not np.isfinite(error) or error > 0.0001:
                    raise RuntimeError(f'explicit attention parity failed: {error}')
                reference.append({'job': {'inputs': {k: v.tolist() for k, v in inputs.items()}, 'options': len(question['options'])}, 'probabilities': expected[0].tolist(), 'torch_error': error})
        print('Converting complete graph', flush=True)
        ir = ov.convert_model(graph, example_input=tuple(tensors.values()), input=[(name, value.shape) for name, value in tensors.items()])
    args.output.mkdir(parents=True, exist_ok=True)
    ov.save_model(ir, args.output / 'model.xml')
    tokenizer.save(str(args.output / 'tokenizer.json'))
    config = {'format_version': 1, 'model_id': pins['checkpoint']['repo'].split('/')[-1], 'adapter': 'kev-qwen3', 'length': args.length, 'max_options': args.max_options, 'precision': 'fp16', 'temperature': graph.temperature, 'pins': pins}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (args.output / 'reference.json').write_text(json.dumps(reference) + '\n')
    print(f'Saved {args.output}', flush=True)
if __name__ == '__main__':
    main()
