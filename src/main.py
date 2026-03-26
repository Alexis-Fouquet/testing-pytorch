from rich import print
from tqdm.rich import tqdm
from lib.build_models.functions_2d import Cos2D, ThreeElements
from lib.build_models.mnist_simple import MnistSimple


if __name__ == "__main__":
    print("Training program")
    models = [
        Cos2D(epochs=3000, layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        Cos2D(epochs=2000, layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        Cos2D(epochs=1000, layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        Cos2D(epochs=100, layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        Cos2D(epochs=300, layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        Cos2D(epochs=500, layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        ThreeElements(epochs=300),
        ThreeElements(
            epochs=301, layers=[0, 1, 2, 3, 4, 5, 6], lrs=[0.1, 0.01, 0.001, 0.0003]
        ),
        ThreeElements(
            epochs=801, layers=[0, 1, 2, 3, 4, 5, 6], lrs=[0.1, 0.01, 0.001, 0.0003]
        ),
        ThreeElements(
            epochs=3000, layers=[0, 1, 2, 3, 4, 5, 6], lrs=[0.1, 0.01, 0.001, 0.0003]
        ),
        ThreeElements(
            epochs=223, layers=[0, 1, 2, 3, 4, 5, 6], lrs=[0.1, 0.01, 0.001, 0.0003]
        ),
        MnistSimple(epochs=223, layers=[0, 1, 2], lrs=[0.01, 0.001, 0.0003]),
    ]

    print("Training all")
    for model in tqdm(models, desc="Models"):
        model.train()
