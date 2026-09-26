SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c

SSH_HOST ?=
MODEL_DIR ?= $(CURDIR)/artifacts/kev-fp16
HF_CACHE ?= $(CURDIR)/.cache
MODEL_VOLUME ?= jev-openvino-npu-model
DOCKER ?= docker
EXPORTER ?= jev_server.exporters.kev
EXPORT_IMAGE ?= jev-openvino-npu-server-export:local
EXPORT_ARGS ?=
PLATFORM ?= 3720

.PHONY: images export compile test schema upload run stop tunnel smoke

images:
	$(DOCKER) build --target export -t jev-openvino-npu-server-export:local .
	$(DOCKER) build --target runtime -t jev-openvino-npu-server:local .

export:
	mkdir -p "$(MODEL_DIR)" "$(HF_CACHE)"
	$(DOCKER) run --rm --memory 7g --memory-swap 9g --cpus 4 --user "$$(id -u):$$(id -g)" -e HOME=/tmp -e HF_HOME=/cache -v "$(HF_CACHE):/cache" -v "$(MODEL_DIR):/artifacts" --entrypoint python $(EXPORT_IMAGE) -m $(EXPORTER) --output /artifacts $(EXPORT_ARGS)

compile:
	$(DOCKER) run --rm --memory 7g --memory-swap 9g --cpus 2 --user "$$(id -u):$$(id -g)" -e HOME=/tmp --entrypoint python3 -v "$(MODEL_DIR):/models" jev-openvino-npu-server:local -m jev_server.tools.compile --platform "$(PLATFORM)"

test:
	$(DOCKER) run --rm --entrypoint python -e PYTHONDONTWRITEBYTECODE=1 -v "$(CURDIR)/jev_server:/app/jev_server:ro" -v "$(CURDIR)/examples:/app/examples:ro" -v "$(CURDIR)/schemas:/app/schemas:ro" -v "$(CURDIR)/tests:/app/tests:ro" -v "$(MODEL_DIR):/models:ro" -v "$(CURDIR)/artifacts/nanojev:/nano:ro" jev-openvino-npu-server-export:local -m pytest -q -p no:cacheprovider

schema:
	$(DOCKER) run --rm --user "$$(id -u):$$(id -g)" -e HOME=/tmp --entrypoint python -v "$(CURDIR)/jev_server:/app/jev_server" -v "$(CURDIR)/schemas:/app/schemas:ro" jev-openvino-npu-server-export:local -m jev_server.tools.generate_api

upload: require-host
	ssh "$(SSH_HOST)" 'test -z "$$(docker ps -q --filter volume=$(MODEL_VOLUME))"'
	$(DOCKER) save jev-openvino-npu-server:local | ssh "$(SSH_HOST)" docker load
	tar -C "$(MODEL_DIR)" --exclude="*.xml" --exclude="*.bin" -cf - . | ssh "$(SSH_HOST)" 'docker run --rm -i --user root -v "$(MODEL_VOLUME):/models" --entrypoint tar jev-openvino-npu-server:local -xf - -C /models'

run: require-host
	ssh "$(SSH_HOST)" 'docker run --rm -d --name jev-openvino-npu-server --memory 1600m --memory-swap 1600m --cpus 2 --device /dev/accel/accel0 --group-add $$(stat -c %g /dev/accel/accel0) -v "$(MODEL_VOLUME):/models:ro" -p 127.0.0.1:8009:8009 jev-openvino-npu-server:local --backend openvino --device NPU'

stop: require-host
	ssh "$(SSH_HOST)" docker rm -f jev-openvino-npu-server

tunnel: require-host
	ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18009:127.0.0.1:8009 "$(SSH_HOST)"

smoke:
	$(DOCKER) run --rm --network host -e JEV_API_KEY --entrypoint python jev-openvino-npu-server-export:local -m jev_server.tools.smoke

.PHONY: require-host
require-host:
	@test -n "$(SSH_HOST)" || { echo "Set SSH_HOST to the target SSH alias"; exit 2; }
