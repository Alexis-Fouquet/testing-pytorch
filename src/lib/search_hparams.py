from itertools import product
from abc import abstractmethod
from math import inf

from lib import base
from lib.training_result import TrainingResult


class ModelTraining:
    def __init__(
        self,
        epochs: int = 500,
        lrs: list[float] = [0.1, 0.01, 0.001, 0.0003, 0.0001],
        layers: list[int] = [0, 2, 4],
    ) -> None:
        self.epochs = epochs
        self.lrs = lrs
        self.layers_nums = layers

    def train(self) -> float:
        mini = inf
        for lr, layers in product(self.lrs, self.layers_nums):
            base.seed()
            result = self.internal_train(epochs=self.epochs, lr=lr, layers=layers)
            result.show(classification=True)
            if result.loss < mini:
                mini = result.loss.item()
        return mini

    @abstractmethod
    def internal_train(self, epochs: int, lr: float, layers: int) -> TrainingResult:
        pass
