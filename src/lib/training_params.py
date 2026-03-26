from pathlib import Path
from typing import cast

from torch import load, nn, save

from lib.device import global_device, global_panel


class TrainingParams:
    def __init__(self, epochs: int = 1, lr: float = 0.07, layers: int = 0) -> None:
        self.epochs = epochs
        self.lr = lr
        self.layers = layers
        self.path: Path | None = None

    def get_full_name(self, name: str):
        return f"n{name}_e{self.epochs}_lr{self.lr}_la{self.layers}"

    def save_exist(self, name: str):
        if self.path is None:
            self.path = self.file_path(name)
        return self.path.exists()

    def load(self, name: str, model: nn.Module):
        if self.path is None:
            self.path = self.file_path(name)
        model.load_state_dict(
            load(self.path, weights_only=True, map_location=global_device)
        )
        model.to(global_device)

    def save(self, name: str, model: nn.Module):
        if self.path is None:
            self.path = self.file_path(name)
        self.path.touch()
        save(model.state_dict(), self.path)

    def file_path(self, name: str):
        models = Path("models")
        models.mkdir(parents=True, exist_ok=True)
        return models / (self.get_full_name(name) + ".pth")

    def get_hparams_dict(self):
        return {"lr": self.lr, "layers": self.layers}

    def already_trained(self, name: str):
        rend = cast(str, global_panel.renderable)
        if rend.count("\n") > 15:
            rend = rend[rend.index("\n") + 1 :]
        rend += f"> Model {self.get_full_name(name)} already trained\n"
        global_panel.renderable = rend

    def training(self, name: str):
        rend = cast(str, global_panel.renderable)
        if rend.count("\n") > 15:
            rend = rend[rend.index("\n") + 1 :]
        rend += f"> Training {self.get_full_name(name)}\n"
        global_panel.renderable = rend
