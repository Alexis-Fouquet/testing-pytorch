from torch import arange, dstack, inference_mode, meshgrid
from torch._prims_common import Tensor
from torch.utils.data import Dataset
from lib.data.tensor_data import TensorDatasetSaved
from lib.models.base_model import BaseModel
from torch.utils.tensorboard import SummaryWriter
from matplotlib.pyplot import ion, pause, subplots
from matplotlib import use
from lib.device import global_device
from lib.training_params import TrainingParams


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
        train_data: Dataset,
        test_data: Dataset,
        training_losses: list[Tensor],
        test_losses: list[Tensor],
        hparams: dict,
        params: TrainingParams,
    ) -> None:
        self.name = name + "_" + str(params.epochs)
        self.model = model
        self.loss = loss
        self.training_losses = list_to_float(training_losses)
        self.test_losses = list_to_float(test_losses)
        self.train_data = train_data
        self.test_data = test_data
        self.hparams = hparams
        self.params = params

    def show(self, classification=False, plot=True):
        assert len(self.training_losses) > 0
        assert len(self.test_losses) > 0

        writer = SummaryWriter(
            log_dir="runs/" + self.name + "/" + self.params.get_full_name(self.name)
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
        if (
            self.test_data is not TensorDatasetSaved
            or self.train_data is not TensorDatasetSaved
        ):
            return
        test_data: TensorDatasetSaved = self.test_data
        train_data: TensorDatasetSaved = self.test_data

        in_test = test_data.x
        out_test = test_data.y
        in_training = train_data.x
        out_training = train_data.y

        fig, axes = subplots(1, 2)
        (sub2, sub3) = axes

        sub2.set_title("Test data")
        sub3.set_title("Training data")
        with inference_mode():
            idx = in_test.squeeze().argsort()
            x = in_test.squeeze()[idx]
            sub2.plot(x, out_test.squeeze()[idx], ".", label="Expected")
            sub2.plot(
                x,
                self.model(in_test.to(global_device)).cpu().squeeze()[idx],
                ".",
                label="Got",
            )

            idx = in_training.squeeze().argsort()
            x = in_training.squeeze()[idx]
            sub3.plot(x, out_training.squeeze()[idx], ".", label="Expected")
            sub3.plot(
                x,
                self.model(in_training.to(global_device)).cpu().squeeze()[idx],
                ".",
                label="Got",
            )

        sub2.legend()
        sub3.legend()

        return fig

    def get_classification_figure(self):
        if (
            self.test_data is not TensorDatasetSaved
            or self.train_data is not TensorDatasetSaved
        ):
            return
        test_data: TensorDatasetSaved = self.test_data
        train_data: TensorDatasetSaved = self.test_data

        in_test = test_data.x
        out_test = test_data.y
        in_training = train_data.x
        out_training = train_data.y

        fig, axes = subplots(1, 2)
        sub1, sub2 = axes

        sub1.set_title("Test data")
        sub2.set_title("Training data")

        with inference_mode():
            x, y = meshgrid(arange(-0.1, 1.1, 0.02), arange(-0.1, 1.1, 0.02))
            input = dstack([x, y])
            z = self.model(input.to(global_device)).cpu().squeeze()
            assert z.size()[0] == x.size()[0], (z.size(), x.size())
            assert z.size()[1] == x.size()[1], (z.size(), x.size())
            if len(z.size()) > 2 and z.size()[2] == 2:
                z = (z > 0.5).float()
                z = z[..., 0] - z[..., 1]
            assert len(z.size()) == 2, z.size()
            assert z.size()[0] == x.size()[0], (z.size(), x.size())
            assert z.size()[1] == x.size()[1], (z.size(), x.size())

            cte = out_test
            ctr = out_training
            if cte.size()[1] == 2:
                cte = (cte > 0.5).float()
                cte = cte[..., 0] - cte[..., 1]
            if ctr.size()[1] == 2:
                ctr = (ctr > 0.5).float()
                ctr = ctr[..., 0] - ctr[..., 1]

            sub1.contourf(x, y, z)
            sub2.contourf(x, y, z)

            color = "0.8"
            sub1.scatter(in_test[:, 0], in_test[:, 1], c=cte, edgecolor=color)
            sub2.scatter(
                in_training[:, 0],
                in_training[:, 1],
                c=ctr,
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

        if fig is None:
            return
        if writer is None:
            fig.show()
        else:
            writer.add_figure("plot/all", fig, global_step=len(self.test_losses))
        pause(0.001)
