from tqdm.rich import tqdm
from lib.functions_2d import Cos2D


if __name__ == "__main__":
    print("Training program")
    models = [Cos2D(epochs=3000, layers=[4, 5, 6, 7, 8, 9, 10, 11, 12])]

    print("Training all")
    for model in tqdm(models, desc="Models"):
        model.train()
