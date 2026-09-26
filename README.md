# Jev-style models OpenVINO NPU server

Runs Kev-0.6B and NanoJev neural graphs on Intel NPU through OpenVINO, serving
the System One API. CPU handles tokenization and HTTP; NPU inference has no fallback.

Requires Docker and a target with an Intel NPU driver and `/dev/accel/accel0`.

```sh
make images export compile test
make upload run SSH_HOST=<target>
make tunnel SSH_HOST=<target>
```

Run `make smoke` in another terminal to verify NPU execution through the SDK.
`http://127.0.0.1:18009` serves `/healthz`, `/v1/models`, and `/v1/systemone`.
`jev-latest` selects the loaded model. Set `JEV_API_KEY` in the server container
to require bearer authentication; export the same key before `make smoke`.

To switch to NanoJev:

```sh
make stop SSH_HOST=<target>
make export compile upload run SSH_HOST=<target> \
  EXPORTER=jev_server.exporters.nanojev \
  MODEL_DIR="$PWD/artifacts/nanojev" MODEL_VOLUME=jev-nanojev
```

`PLATFORM` defaults to `3720`. Recompile after changing the NPU architecture or
OpenVINO version. Bundle format 3 verifies the tokenizer and complete reference
requests at startup; older bundles require re-export. After an interrupted export
or upload, repeat that step before starting the server.
The default context is 256 tokens; requests exceeding model limits return 422.
NanoJev requires text inputs and supports eight options; Kev supports 32.

Exports use fixed token lengths and pad shorter inputs, so larger contexts cost
more per inference. OpenVINO 2026.4's [dynamic-shape preview](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html#dynamic-shapes)
covers bounded vision models on NPU40XX and newer, excluding platform 3720.
Choose the smallest context that fits your requests; set
`EXPORT_ARGS="--length 1024"` when exporting, then recompile.

Model revisions are pinned in `assets.lock.json` and `models/nanojev.json`.
See [source attribution](NOTICE.md).

Benchmark a server with a reproducible labeled corpus:

```sh
make bench BENCH_ARGS='generate --output /reports/synthetic.jsonl'
make bench BENCH_ARGS='run --url http://127.0.0.1:18009 --corpus /reports/synthetic.jsonl --output /reports/baseline.json'
make bench BENCH_ARGS='compare /reports/baseline.json /reports/candidate.json'
```

`run --model-dir /models` measures the runtime directly; pass the NPU device and
render group through `BENCH_DOCKER_ARGS`, or select `--device CPU`.
Runtime reports include process peak RSS, excluding device allocations. HTTP runs include network time and cannot
measure server memory. Reports include warmup settings, repeated timings, labels,
probabilities, accuracy, Brier score, and failures. Score accuracy uses the most
probable level. Comparisons require identical corpora, settings, and complete runs.
Synthetic cases test controlled behavior, not general model quality.
The default `small` corpus fits both default bundles; `generate --profile context2048`
adds longer contexts and up to 32 options for larger Kev exports.
`import-kev --input FILE --sha256 HASH --output FILE` converts checksum-verified
[Kev evaluation data](https://github.com/jaredpalmer/kev/tree/main/evals).
Use development data for tuning; preserve test splits for final evaluation and
follow source dataset licenses. Keep downloaded data and reports untracked.
HTTP comparisons require a server exposing artifact hashes through `/healthz`.

Compiler experiments accept `COMPILE_ARGS='--optimization-level 2'` and
`--no-profiling`. Compile into separate model directories;
compare one setting at a time against the same FP16 baseline.

`make schema` regenerates Pydantic wire models from `schemas/systemone.json`,
a snapshot of the [System One OpenAPI definition](https://api.typesafe.ai/openapi.json).
Request limits and the default model alias live in `api/types.py`.

`models/` owns model adapters; `backends/` owns device execution; `exporters/`
owns graph conversion; `tools/` owns compile, validation, and SDK commands inside
`jev_server/`. Add a model through an exporter, adapter, registry entry, and
reference fixtures. Runtime and HTTP handling stay shared. This keeps the two
supported model layouts separate without a plugin framework; add backend-specific
artifact handling when a second backend requires it.
