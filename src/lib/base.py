from torch import Tensor, inference_mode, manual_seed, optim, zeros
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from tqdm.rich import tqdm

from lib.data_saver.loader import BaseLoader
from lib.modules.base_model import BaseModel
from lib.training_params import TrainingParams
from lib.training_result import TrainingResult


def seed(seed: int = 42):
    # Should always be the same
    manual_seed(seed)


def testing(
    model: BaseModel,
    data: BaseLoader,
    fn_loss: _Loss,
    classification: bool = False,
) -> tuple[Tensor, Tensor]:
    with inference_mode():
        total_loss = zeros([1], device=model.device_str)
        # TODO: fix this for batched models
        last_accuracy = zeros([1], device=model.device_str)
        for x, y in data:
            x = x.to(model.device_str, non_blocking=True)
            y = y.to(model.device_str, non_blocking=True)
            out_model = model(x)
            assert (
                out_model.size()[0] == y.size()[0]
                and len(y.size()) <= len(out_model.size()) <= 2
            ), (
                out_model.size(),
                x.size(),
                y.size(),
            )
            loss: Tensor = fn_loss(out_model, y)
            total_loss += loss.detach()
            if classification:
                last_accuracy = (
                    out_model.argmax(dim=1) == y
                ).float().mean().detach() * 100
        return (total_loss, last_accuracy)


def train_step(
    model: BaseModel,
    data: BaseLoader,
    fn_loss: _Loss,
    fn_opti: Optimizer,
) -> Tensor:
    total_loss = zeros((), device=model.device_str)
    for x, y in data:
        x = x.to(model.device_str, non_blocking=True)
        y = y.to(model.device_str, non_blocking=True)
        out_model = model(x)
        loss: Tensor = fn_loss(out_model, y)
        fn_opti.zero_grad()
        # Much easier than in C
        loss.backward()
        fn_opti.step()
        total_loss += loss.detach()
    return total_loss


def train(
    model: BaseModel,
    training: BaseLoader,
    test_data: BaseLoader,
    name: str,
    params: TrainingParams,
) -> TrainingResult:
    assert params.epochs > 0, params.epochs
    params.training(name)
    fn_loss = model.fn_loss
    fn_opti = optim.AdamW(model.parameters(), lr=params.lr)

    te_losses = []
    tr_losses = []
    te_acc = []

    te_loss = Tensor()
    tr_loss = Tensor()
    for i in tqdm(range(params.epochs), desc="Epochs"):
        model.train()
        tr_loss = train_step(model, training, fn_loss, fn_opti)
        tr_losses.append(tr_loss)
        if i % 10 == 0:
            model.eval()
            (te_loss, acc) = testing(
                model, test_data, fn_loss, classification=params.classification
            )
            te_losses.append(te_loss)
            if params.classification:
                te_acc.append(acc)

    if (params.epochs - 1) % 10 != 0:
        model.eval()
        (te_loss, _) = testing(model, test_data, fn_loss)

    return TrainingResult(
        model,
        te_loss,
        name,
        training_losses=tr_losses,
        test_losses=te_losses,
        train_data=training,
        test_data=test_data,
        accuracy=te_acc,
        hparams=params.get_hparams_dict(),
        params=params,
    )
