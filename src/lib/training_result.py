from torch import inference_mode
from torch._prims_common import Tensor
from lib.models.base_model import BaseModel
from matplotlib.pyplot import ion, pause, subplots, show
from matplotlib import use
from lib.device import global_device


def to_cpu(t: Tensor | None) -> Tensor | None:
    if t is None:
        return None
    return t.cpu()


def list_to_float(t: list[Tensor]) -> list[float]:
    return [a.item() for a in t]


class TrainingResult:
    def __init__(
        self,
        model: BaseModel,
        loss: Tensor,
        training_losses: list[Tensor] = list(),
        test_losses: list[Tensor] = list(),
        in_training: Tensor | None = None,
        in_test: Tensor | None = None,
        out_training: Tensor | None = None,
        out_test: Tensor | None = None,
    ) -> None:
        self.model = model
        self.loss = loss
        self.training_losses = list_to_float(training_losses)
        self.test_losses = list_to_float(test_losses)
        self.in_training = to_cpu(in_training)
        self.in_test = to_cpu(in_test)
        self.out_training = to_cpu(out_training)
        self.out_test = to_cpu(out_test)

    def plot(self):
        use("QtAgg")
        ion()

        _, axes = subplots(2, 2)
        (sub0, sub1), (sub2, sub3) = axes

        sub0.plot(self.test_losses)
        sub0.set_title("Test loss over time")

        sub1.plot(self.training_losses)
        sub1.set_title("Training loss over time")

        sub2.set_title("Test data")
        sub3.set_title("Training data")
        with inference_mode():
            if self.in_test is not None and self.out_test is not None:
                idx = self.in_test.squeeze().argsort()
                x = self.in_test[idx].squeeze()
                sub2.plot(x, self.out_test.squeeze()[idx], ".", label="Expected")
                sub2.plot(
                    x,
                    self.model(self.in_test.to(global_device)).cpu().squeeze()[idx],
                    ".",
                    label="Got",
                )
            if self.in_training is not None and self.out_training is not None:
                idx = self.in_training.squeeze().argsort()
                x = self.in_training.squeeze()[idx]
                sub3.plot(x, self.out_training.squeeze()[idx], ".", label="Expected")
                sub3.plot(
                    x,
                    self.model(self.in_training.to(global_device)).cpu().squeeze()[idx],
                    ".",
                    label="Got",
                )
        sub2.legend()
        sub3.legend()

        show(block=True)
        pause(0.001)
