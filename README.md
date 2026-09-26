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
to require bearer authentication.

To switch to NanoJev:

```sh
make stop SSH_HOST=<target>
make export compile upload run SSH_HOST=<target> \
  EXPORTER=jev_server.export_nanojev \
  MODEL_DIR="$PWD/artifacts/nanojev" MODEL_VOLUME=jev-nanojev
```

`PLATFORM` defaults to `3720`. Recompile after changing the NPU architecture or
OpenVINO version. Startup verifies compatibility and reference probabilities.
The default context is 256 tokens; requests exceeding model limits return 422.
NanoJev requires text inputs and supports eight options; Kev supports 32.

Model revisions are pinned in `assets.lock.json` and `models/nanojev.json`.
See [source attribution](NOTICE.md).
