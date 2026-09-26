def load_backend(name, directory, graphs, device):
    if name != "openvino":
        raise ValueError(f"Unsupported backend: {name}")
    from jev_server.backends.openvino import OpenVINOBackend

    return OpenVINOBackend(directory, graphs, device)
