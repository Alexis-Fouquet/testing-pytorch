from torch import Tensor, inference_mode, manual_seed, optim, zeros
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm.rich import tqdm
from rich import print

from lib.data.tensor_data import TensorDatasetSaved
from lib.modules.base_model import BaseModel
from lib.training_params import TrainingParams
from lib.training_result import TrainingResult


def seed(seed: int = 42):
    # Should always be the same
    manual_seed(seed)


def testing(
    model: BaseModel,
    data: DataLoader,
    fn_loss: _Loss,
) -> Tensor:
    with inference_mode():
        total_loss = zeros([1], device=model.device_str)
        for x, y in data:
            x = x.to(model.device_str, non_blocking=True)
            y = y.to(model.device_str, non_blocking=True)
            out_model = model(x)
            assert out_model.size() == y.size(), (
                out_model.size(),
                y.size(),
            )
            loss: Tensor = fn_loss(out_model, y)
            total_loss += loss
        return total_loss


def train_step(
    model: BaseModel,
    data: DataLoader,
    fn_loss: _Loss,
    fn_opti: Optimizer,
) -> Tensor:
    total_loss = zeros([1], device=model.device_str)
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
        fn_opti.zero_grad()
        # Much easier than in C
        loss.backward()
        fn_opti.step()
        total_loss += loss
    return total_loss


def train(
    model: BaseModel,
    training: Dataset,
    test_data: Dataset,
    name: str,
    params: TrainingParams,
) -> TrainingResult:
    assert params.epochs > 0, params.epochs
    fn_loss = model.fn_loss
    fn_opti = optim.AdamW(model.parameters(), lr=params.lr)

    # Note: cannot pin if already on GPU
    # Note: cannot use workers with pytest
    tr_saved = isinstance(training, TensorDatasetSaved)
    te_saved = isinstance(test_data, TensorDatasetSaved)
    train_dataloader = DataLoader(
        training,
        batch_size=len(training.x) if tr_saved else 1024,
        shuffle=not tr_saved,
        pin_memory=False,
        num_workers=0 if tr_saved else 3,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=len(test_data.x) if te_saved else 1024,
        shuffle=False,
        pin_memory=False,
        num_workers=0 if te_saved else 3,
    )

    te_losses = []
    tr_losses = []

    te_loss = Tensor()
    tr_loss = Tensor()
    for i in tqdm(range(params.epochs), desc="Epochs"):
        model.train()
        tr_loss = train_step(model, train_dataloader, fn_loss, fn_opti)
        tr_losses.append(tr_loss)
        if i % 10 == 0:
            model.eval()
            te_loss = testing(model, test_dataloader, fn_loss)
            te_losses.append(te_loss)

    if (params.epochs - 1) % 10 != 0:
        model.eval()
        te_loss = testing(model, test_dataloader, fn_loss)

    return TrainingResult(
        model,
        te_loss,
        name,
        training_losses=tr_losses,
        test_losses=te_losses,
        train_data=training,
        test_data=test_data,
        hparams=params.get_hparams_dict(),
        params=params,
    )
