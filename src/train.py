import os
import numpy as np
import yaml
from tensorflow import keras

with open("params.yaml") as f:
    p = yaml.safe_load(f)["train"]

train = np.load("data/processed/train.npz")
val = np.load("data/processed/val.npz")

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(p["dense_units"], activation="relu"),
    keras.layers.Dropout(p["dropout_rate"]),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

os.makedirs("models", exist_ok=True)

model.fit(
    train["x"], train["y"],
    validation_data=(val["x"], val["y"]),
    epochs=p["epochs"],
    batch_size=p["batch_size"],
    callbacks=[keras.callbacks.CSVLogger("models/history.csv")],
)

model.save("models/model.h5")
print("Model saved to models/model.h5")