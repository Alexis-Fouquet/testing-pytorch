from torch import cuda
from rich.panel import Panel


global_device = "cuda" if cuda.is_available() else "cpu"
global_panel = Panel("Starting\n", title="Logs", height=20)
