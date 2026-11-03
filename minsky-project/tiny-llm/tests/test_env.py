import importlib


def test_package_importable():
    mod = importlib.import_module("tiny_llm")
    assert mod.__version__


def test_core_deps():
    import tiktoken
    import torch

    enc = tiktoken.get_encoding("gpt2")
    assert enc.decode(enc.encode("Hello, world")) == "Hello, world"
    assert torch.tensor([1.0, 2.0]).sum().item() == 3.0
