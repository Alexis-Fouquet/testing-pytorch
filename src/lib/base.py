from torch import Tensor, inference_mode, manual_seed, nn, optim
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from tqdm.rich import tqdm

from lib.models.base_model import BaseModel
from lib.search_hparams import TrainingParams
from lib.training_result import TrainingResult


def seed(seed: int = 42):
    # Should always be the same
    manual_seed(seed)


def testing(
    model: BaseModel,
    in_test: Tensor,
    out_test: Tensor,
    fn_loss: _Loss,
    epoch: int,
    tr_loss: Tensor,
):
    with inference_mode():
        out_model = model(in_test)
        assert out_model.size() == out_test.size(), (
            out_model.size(),
            out_test.size(),
        )
        loss = fn_loss(out_model, out_test)
        model.print_epoch(out_test, out_model, tr_loss, loss, epoch)
        return loss


def train_step(
    model: nn.Module,
    in_training: Tensor,
    out_training: Tensor,
    fn_loss: _Loss,
    fn_opti: Optimizer,
):
    out_model = model(in_training)
    assert out_model.size() == out_training.size(), (
        out_model.size(),
        out_training.size(),
    )
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
    name: str,
    params: TrainingParams,
) -> TrainingResult:
    assert params.epochs > 0, params.epochs
    fn_loss = model.fn_loss
    fn_opti = optim.AdamW(model.parameters(), lr=params.lr)

    in_training = in_training.to(model.device_str)
    out_training = out_training.to(model.device_str)
    in_test = in_test.to(model.device_str)
    out_test = out_test.to(model.device_str)
    assert len(in_training.size()) == len(in_test.size()), (
        in_training.size(),
        in_test.size(),
    )
    assert len(out_training.size()) == len(out_test.size()), (
        out_training.size(),
        out_test.size(),
    )

    te_losses = []
    tr_losses = []

    te_loss = Tensor()
    tr_loss = Tensor()
    for i in tqdm(range(params.epochs), desc="Epochs"):
        model.train()
        tr_loss = train_step(model, in_training, out_training, fn_loss, fn_opti)
        tr_losses.append(tr_loss)
        if i % 10 == 0:
            model.eval()
            te_loss = testing(model, in_test, out_test, fn_loss, i, tr_loss=tr_loss)
            te_losses.append(te_loss)

    if (params.epochs - 1) % 10 != 0:
        model.eval()
        testing(model, in_test, out_test, fn_loss, params.epochs, tr_loss=tr_loss)

    return TrainingResult(
        model,
        te_loss,
        name,
        training_losses=tr_losses,
        test_losses=te_losses,
        in_test=in_test,
        out_test=out_test,
        in_training=in_training,
        out_training=out_training,
        hparams=params.get_hparams_dict(),
    )
