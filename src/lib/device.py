from torch import cuda


global_device = "cuda" if cuda.is_available() else "cpu"
