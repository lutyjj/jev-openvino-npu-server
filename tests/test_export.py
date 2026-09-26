import numpy as np
import openvino as ov
import torch
from transformers import AutoModel, Qwen3Config

from jev_server.export import QwenGraph
from jev_server.export_nanojev import NanoHead


def converted_output(model, inputs):
    ir = ov.convert_model(model, example_input=inputs, input=[value.shape for value in inputs])
    compiled = ov.Core().compile_model(ir, "CPU", {
        "INFERENCE_PRECISION_HINT": "f32", "INFERENCE_NUM_THREADS": 2})
    return compiled([value.numpy() for value in inputs])[0]


def test_qwen_graph_matches_transformers():
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=16, rope_parameters={"rope_type": "default", "rope_theta": 1000000})
    backbone = AutoModel.from_config(config, attn_implementation="eager").eval()
    graph = QwenGraph(backbone, 8).eval()
    ids = torch.tensor([[1, 9, 3, 12, 7, 5, 2, 11]])
    selector = torch.eye(8)[None]
    with torch.inference_mode():
        expected = backbone(ids, use_cache=False).last_hidden_state.numpy()
        np.testing.assert_allclose(graph(ids, selector).numpy(), expected, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(converted_output(graph, (ids, selector)), expected, atol=1e-5, rtol=1e-5)


def test_nano_head_matches_torch_attention():
    torch.manual_seed(8)
    head = NanoHead(64, 8).eval()
    leaves = torch.randn(1, 8, 64)
    valid = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)
    with torch.inference_mode():
        expected = head.reference(leaves[:, :3], "choice")
        actual = converted_output(head, (leaves, valid, torch.ones(1), torch.zeros(1)))[0]
    np.testing.assert_allclose(actual[:3], expected, atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(actual[3:], 0)
