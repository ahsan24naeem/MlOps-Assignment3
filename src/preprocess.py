import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)["preprocess"]


def normalize(x):
    # teammate: standardize using Fashion-MNIST mean/std
    x = x.astype("float32") / 255.0
    return (x - 0.2860) / 0.3530
    
raw = np.load("data/raw/fashion_mnist.npz")
x_train, y_train = normalize(raw["x_train"]), raw["y_train"]
x_test, y_test = normalize(raw["x_test"]), raw["y_test"]

x_tr, x_val, y_tr, y_val = train_test_split(
    x_train, y_train,
    test_size=params["test_size"],
    random_state=params["seed"],
)

os.makedirs("data/processed", exist_ok=True)
np.savez("data/processed/train.npz", x=x_tr, y=y_tr)
np.savez("data/processed/val.npz", x=x_val, y=y_val)
np.savez("data/processed/test.npz", x=x_test, y=y_test)

print("Processed:", x_tr.shape, x_val.shape, x_test.shape)