from torch import inference_mode
from torch._prims_common import Tensor
from lib.models.base_model import BaseModel
from torch.utils.tensorboard import SummaryWriter
from matplotlib.pyplot import ion, pause, subplots, show
from matplotlib import use
from lib.device import global_device, writer


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
        name: str,
        training_losses: list[Tensor] = list(),
        test_losses: list[Tensor] = list(),
        in_training: Tensor | None = None,
        in_test: Tensor | None = None,
        out_training: Tensor | None = None,
        out_test: Tensor | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.loss = loss
        self.training_losses = list_to_float(training_losses)
        self.test_losses = list_to_float(test_losses)
        self.in_training = to_cpu(in_training)
        self.in_test = to_cpu(in_test)
        self.out_training = to_cpu(out_training)
        self.out_test = to_cpu(out_test)

    def show(self):
        for (i, s) in enumerate(self.training_losses):
            writer.add_scalar(self.name + "/loss/training", s, i);
        for (i, s) in enumerate(self.test_losses):
            writer.add_scalar(self.name + "/loss/test", s, i);
        if self.in_test is not None:
            writer.add_graph(self.model, self.in_test.to(global_device))
        self.plot(writer)
        writer.flush()

    def plot(self, writer: SummaryWriter | None = None):
        if writer is None:
            use("QtAgg")
            ion()

        fig, axes = subplots(1, 2)
        (sub2, sub3) = axes

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

        if writer is None:
            show(block=True)
            pause(0.001)
        else:
            writer.add_figure(self.name + "/plot/all", fig);
