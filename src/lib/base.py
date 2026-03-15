from torch import Tensor, cuda, inference_mode, manual_seed, nn, optim
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from lib.models.base_model import BaseModel


global_device = "cuda" if cuda.is_available() else "cpu"


def seed(seed: int = 42):
    # Should always be the same
    manual_seed(seed)


def testing(model: nn.Module, in_test: Tensor, out_test: Tensor, fn_loss: _Loss):
    with inference_mode():
        out_model = model(in_test)
        loss = fn_loss(out_model, out_test)
        return loss


def train_step(
    model: nn.Module,
    in_training: Tensor,
    out_training: Tensor,
    fn_loss: _Loss,
    fn_opti: Optimizer,
):
    out_model = model(in_training)
    loss = fn_loss(out_model, out_training)
    fn_opti.zero_grad()
    # Much easier than in C
    loss.backward()
    fn_opti.step()
    return loss


def train(
    model: BaseModel,
    in_training: Tensor,
    out_training: Tensor,
    in_test: Tensor,
    out_test: Tensor,
    epochs: int = 1,
):
    fn_loss = nn.L1Loss()
    fn_opti = optim.SGD(model.parameters())

    in_training.to(model.device_str)

    for i in range(epochs):
        model.train()
        tr_loss = train_step(model, in_training, out_training, fn_loss, fn_opti)
        if i % 10 == 0:
            model.eval()
            te_loss = testing(model, in_test, out_test, fn_loss)
            print(f"Epoch {i} with tr {tr_loss} and te {te_loss}")
