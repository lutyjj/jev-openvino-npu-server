# Attribution

- `jev_server/api.py` adapts [Kev](https://github.com/jaredpalmer/kev/tree/5920c5fe4ca8e0970ed4209ac2c9b8e18bea5109),
  copyright 2026 Jared Palmer, under [Apache-2.0](LICENSE), with modified request
  handling and date preprocessing removed. Kev and Qwen3 weights are Apache-2.0.
- The exporter adapts the fixed attention layout and pointer selectors from
  [Fluid Inference's Kev exporter](https://huggingface.co/FluidInference/kev-0.6b-coreml/blob/main/source/export_model.py),
  Apache-2.0, for OpenVINO.
- NanoJev's tokenizer layout and decision head adapt
  [NanoJev](https://github.com/TianyuCodings/NanoJev), copyright 2026 OpenJev
  contributors, under [MIT](licenses/NanoJev.txt), with separate OpenVINO graphs.
