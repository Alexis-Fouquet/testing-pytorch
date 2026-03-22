import datetime
from torch import arange, dstack, inference_mode, meshgrid
from torch._prims_common import Tensor
from lib.models.base_model import BaseModel
from torch.utils.tensorboard import SummaryWriter
from matplotlib.pyplot import ion, pause, subplots, show
from matplotlib import use
from lib.device import global_device


def to_cpu(t: Tensor) -> Tensor:
    return t.cpu()


def list_to_float(t: list[Tensor]) -> list[float]:
    return [a.item() for a in t]


class TrainingResult:
    def __init__(
        self,
        model: BaseModel,
        loss: Tensor,
        name: str,
        in_training: Tensor,
        in_test: Tensor,
        out_training: Tensor,
        out_test: Tensor,
        training_losses: list[Tensor],
        test_losses: list[Tensor],
        hparams: dict,
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
        self.hparams = hparams

    def show(self, classification=False, plot=True):
        # May not work without this
        assert len(self.training_losses) > 0
        assert len(self.test_losses) > 0
        assert plot.__class__ is bool, plot.__class__

        writer = SummaryWriter(
            log_dir="runs/" + self.name + "/" + str(datetime.datetime.now().time())
        )

        for i, s in enumerate(self.training_losses):
            writer.add_scalar("loss/training", s, i)
        for i, s in enumerate(self.test_losses):
            writer.add_scalar("loss/test", s, i)
        if plot:
            self.plot(writer=writer, classification=classification)
        writer.add_hparams(
            self.hparams,
            {"loss/test": self.loss.item(), "loss/training": self.training_losses[-1]},
        )
        writer.flush()
        writer.close()

    def get_fct_figure(self):
        fig, axes = subplots(1, 2)
        (sub2, sub3) = axes

        sub2.set_title("Test data")
        sub3.set_title("Training data")
        with inference_mode():
            idx = self.in_test.squeeze().argsort()
            x = self.in_test.squeeze()[idx]
            sub2.plot(x, self.out_test.squeeze()[idx], ".", label="Expected")
            sub2.plot(
                x,
                self.model(self.in_test.to(global_device)).cpu().squeeze()[idx],
                ".",
                label="Got",
            )

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

        return fig

    def get_classification_figure(self):
        fig, axes = subplots(1, 2)
        sub1, sub2 = axes

        sub1.set_title("Test data")
        sub2.set_title("Training data")

        with inference_mode():
            x, y = meshgrid(arange(-0.1, 1.1, 0.02), arange(-0.1, 1.1, 0.02))
            input = dstack([x, y])
            z = self.model(input.to(global_device)).cpu().squeeze()
            assert z.size()[0] == x.size()[0]
            assert z.size()[1] == x.size()[1]

            sub1.contourf(x, y, z)
            sub2.contourf(x, y, z)

            color = "0.8"
            sub1.scatter(
                self.in_test[:, 0], self.in_test[:, 1], c=self.out_test, edgecolor=color
            )
            sub2.scatter(
                self.in_training[:, 0],
                self.in_training[:, 1],
                c=self.out_training,
                edgecolor=color,
            )

        return fig

    def plot(self, writer: SummaryWriter | None = None, classification=False):
        if writer is None:
            use("QtAgg")
            ion()

        fig = (
            self.get_classification_figure()
            if classification
            else self.get_fct_figure()
        )

        if writer is None:
            show(block=True)
        else:
            writer.add_figure("plot/all", fig, global_step=len(self.test_losses))
        pause(0.001)
