from jev_server.models.kev import KevAdapter
from jev_server.models.nanojev import NanoJevAdapter


def load_adapter(config, tokenizer):
    adapters = {"kev-qwen3": KevAdapter, "nanojev-qwen3": NanoJevAdapter}
    try:
        adapter = adapters[config["adapter"]]
    except KeyError as error:
        raise ValueError("Unsupported or missing model adapter") from error
    return adapter(config, tokenizer)
