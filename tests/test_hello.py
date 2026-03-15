import torch

def test_hello():
    print("Hello Pytest!")
    assert True

def test_version():
    assert torch.__version__.startswith("2.10")
