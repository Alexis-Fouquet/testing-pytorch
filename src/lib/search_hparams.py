from itertools import product
from abc import abstractmethod
from math import inf
from pathlib import Path

from torch import load, nn, save
from tqdm.rich import tqdm

from lib.device import global_device
from lib import base
from lib.training_result import TrainingResult


class TrainingParams:
    def __init__(self, epochs: int = 1, lr: float = 0.07, layers: int = 0) -> None:
        self.epochs = epochs
        self.lr = lr
        self.layers = layers
        self.path: Path | None = None

    def get_full_name(self, name: str):
        return f"n{name}_e{self.epochs}_lr{self.lr}_la{self.layers}.pth"

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
        return models / self.get_full_name(name)

    def get_hparams_dict(self):
        return {"lr": self.lr, "layers": self.layers}


class ModelTraining:
    def __init__(
        self,
        epochs: int = 500,
        lrs: list[float] = [0.01, 0.001, 0.0003],
        layers: list[int] = [0, 2, 4],
    ) -> None:
        self.epochs = epochs
        self.lrs = lrs
        self.layers_nums = layers

    def train(self) -> float:
        mini = inf
        with tqdm(total=len(self.lrs)*len(self.layers_nums), desc="Hparams") as pbar:
            for lr, layers in product(self.lrs, self.layers_nums):
                base.seed()
                result = self.internal_train(TrainingParams(self.epochs, lr, layers))
                pbar.update(1)
                if result is None:
                    continue
                result.show(classification=True)
                if result.loss < mini:
                    mini = result.loss.item()
        if mini == inf:
            return 0.01
        return mini

    @abstractmethod
    def internal_train(self, params: TrainingParams) -> TrainingResult | None:
        pass
