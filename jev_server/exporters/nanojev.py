import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from tokenizers import Tokenizer
from torch import nn
from transformers import AutoConfig, AutoModel

from jev_server.api.types import SystemOneRequest
from jev_server.bundle import save_metadata
from jev_server.exporters.qwen import QwenGraph
from jev_server.models.nanojev import NanoJevAdapter


class NanoHead(nn.Module):
    def __init__(self, hidden, width):
        super().__init__()
        self.width = width
        self.norm = nn.LayerNorm(hidden)
        self.scalar = nn.Linear(hidden, 1)
        self.set_project = nn.Linear(hidden + 1, 128)
        self.set_attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.set_output = nn.Linear(128, 1)

    def forward(self, leaves, valid, is_choice, is_boolean):
        h = self.norm(leaves)
        z = self.scalar(h).squeeze(-1)
        log_k = valid.sum(-1).log()[:, None, None].expand(-1, self.width, 1)
        u = self.set_project(torch.cat([h, log_k], -1))
        q, k, v = nn.functional.linear(
            u, self.set_attention.in_proj_weight, self.set_attention.in_proj_bias
        ).chunk(3, -1)
        q, k, v = [x.reshape(1, self.width, 4, 32).transpose(1, 2) for x in (q, k, v)]
        weights = torch.softmax(
            q @ k.transpose(-1, -2) * 32**-0.5 + (1 - valid[:, None, None]) * -10000, -1
        )
        mixed = (weights @ v).transpose(1, 2).reshape(1, self.width, 128)
        mixed = self.set_attention.out_proj(mixed)
        z = z + is_choice * self.set_output(torch.tanh(u + mixed)).squeeze(-1)
        boolean = torch.cat([z[:, :1] * 0, z[:, :1], z[:, 2:] * 0], -1)
        z = z * (1 - is_boolean) + boolean * is_boolean
        return torch.softmax(z + (1 - valid) * -10000, -1)

    def reference(self, leaves, typ):
        h = self.norm(leaves)
        z = self.scalar(h).squeeze(-1)
        if typ == "choice":
            log_k = torch.full((*h.shape[:2], 1), float(np.log(h.shape[1])))
            u = self.set_project(torch.cat([h, log_k], -1))
            mixed, _ = self.set_attention(u, u, u, need_weights=False)
            z = z + self.set_output(torch.tanh(u + mixed)).squeeze(-1)
        if typ == "boolean":
            z = torch.cat([z * 0, z], -1)
        return torch.softmax(z, -1)[0].numpy()


def load_backbone_config(path):
    body_config = AutoConfig.from_pretrained(path, trust_remote_code=False)
    rope = body_config.rope_parameters
    if rope["rope_type"] != "default":
        raise ValueError("Unsupported NanoJev rotary-position configuration")
    return body_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--max-options", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.max_options < 2:
        raise ValueError("At least two option slots are required")
    pins = json.loads(Path("models/nanojev.json").read_text())
    checkpoint = Path(
        snapshot_download(
            pins["repo"],
            revision=pins["revision"],
            allow_patterns=[
                "best.safetensors",
                "backbone_config/config.json",
                "tokenizer/*.json",
            ],
        )
    )
    with (checkpoint / "best.safetensors").open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != pins["weights_sha256"]:
            raise ValueError("Checkpoint digest mismatch")
    body_config = load_backbone_config(checkpoint / "backbone_config")
    backbone = AutoModel.from_config(
        body_config, attn_implementation="eager", trust_remote_code=False
    ).eval()
    head = NanoHead(body_config.hidden_size, args.max_options).eval()
    weights = load_file(checkpoint / "best.safetensors")
    backbone.load_state_dict(
        {
            k.removeprefix("backbone."): v
            for k, v in weights.items()
            if k.startswith("backbone.")
        }
    )
    head.load_state_dict(
        {k: v for k, v in weights.items() if not k.startswith("backbone.")}
    )
    del weights
    graph = QwenGraph(backbone, args.length).eval()
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer/tokenizer.json"))
    config = {
        "model_id": "nanojev",
        "release_date": pins["release_date"],
        "adapter": "nanojev-qwen3",
        "length": args.length,
        "max_options": args.max_options,
        "hidden_size": body_config.hidden_size,
        "eos_token_id": body_config.eos_token_id,
        "precision": "fp16",
        "pins": pins,
    }
    adapter = NanoJevAdapter(config, tokenizer)
    reference = []
    examples = {}

    class TorchBackend:
        def run(self, name, inputs):
            examples[name] = inputs
            tensors = {k: torch.from_numpy(v) for k, v in inputs.items()}
            return (graph if name == "backbone" else head)(**tensors).numpy()

    with torch.inference_mode():
        for fixture in json.loads(Path("examples/fixtures.json").read_text()):
            prepared = adapter.prepare(
                SystemOneRequest.model_validate(fixture["request"])
            )
            for qid, job in zip(fixture["request"]["questions"], prepared.jobs):
                leaves = torch.stack(
                    [
                        backbone(
                            torch.tensor([ids]), use_cache=False
                        ).last_hidden_state[0, -1]
                        for ids in job["paths"]
                    ]
                )[None]
                expected = head.reference(leaves, job["type"])
                actual = adapter.infer(job, TorchBackend())
                error = float(np.max(np.abs(actual - expected)))
                if not np.isfinite(error) or error > 0.0001:
                    raise RuntimeError(f"NanoJev export parity failed: {error}")
                reference.append(
                    {
                        "request": {
                            "state": fixture["request"]["state"],
                            "questions": {qid: fixture["request"]["questions"][qid]},
                        },
                        "probabilities": expected.tolist(),
                        "torch_error": error,
                    }
                )
        for name, model in (("backbone", graph), ("head", head)):
            print(f"Converting {name}", flush=True)
            inputs = {k: torch.from_numpy(v) for k, v in examples[name].items()}
            ir = ov.convert_model(
                model,
                example_input=tuple(inputs.values()),
                input=[(k, v.shape) for k, v in inputs.items()],
            )
            directory = args.output / name
            directory.mkdir(parents=True, exist_ok=True)
            ov.save_model(ir, directory / "model.xml")
    save_metadata(args.output, config, tokenizer, reference)
    print(
        json.dumps(
            {
                "reference_cases": len(reference),
                "max_torch_error": max(r["torch_error"] for r in reference),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
