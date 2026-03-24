from itertools import product
from abc import abstractmethod
from math import inf

from tqdm.rich import tqdm

from lib import base
from lib.training_params import TrainingParams
from lib.training_result import TrainingResult


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
        with tqdm(total=len(self.lrs) * len(self.layers_nums), desc="Hparams") as pbar:
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
