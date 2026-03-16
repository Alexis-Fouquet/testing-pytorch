from lib.models.base_model import BaseModel


class TrainingResult:
    def __init__(self, model: BaseModel, loss) -> None:
        self.model = model
        self.loss = loss
