from torch import cuda
from torch.utils.tensorboard import SummaryWriter


global_device = "cuda" if cuda.is_available() else "cpu"
writer = SummaryWriter()
