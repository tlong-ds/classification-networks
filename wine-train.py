import json
import os

# Set backend before importing Keras
os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import BatchNormalization, Dense, LeakyReLU
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BATCH_SIZE = 16
EPOCHS = 25
ALPHA = 0.001
LAMBDA = 1e-3


def data_label(y):
    for i in range(len(y)):
        if y[i] == 1:
            y[i] = 0
        elif y[i] == 2:
            y[i] = 1
        else:
            y[i] = 2
    return y


def convert_label(y):
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = 1
        elif y[i] == 1:
            y[i] = 2
        else:
            y[i] = 3
    return y


raw_df = pd.read_csv("data/wine/raw/winedata.csv")
feature_cols = raw_df.columns[:13]
df = raw_df.drop_duplicates(subset=feature_cols, keep="first").reset_index(drop=True)

train, test = train_test_split(df, test_size=0.2, shuffle=True, stratify=df.iloc[:, 13])

# Save train and test data splits
os.makedirs("data/wine/processed", exist_ok=True)
train.to_csv("data/wine/processed/train.csv", index=False)
test.to_csv("data/wine/processed/test.csv", index=False)

X_tr, y_tr = np.array(train.iloc[:, 0:13]), np.array(train.iloc[:, 13])
X_test, y_test = np.array(test.iloc[:, 0:13]), np.array(test.iloc[:, 13])
y_tr, y_test = data_label(y_tr), data_label(y_test)
y_tr, y_test = (
    to_categorical(y_tr, num_classes=3),
    to_categorical(y_test, num_classes=3),
)

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_test_scaled = scaler.transform(X_test)

model = Sequential(
    [
        keras.Input(shape=(13,)),
        Dense(64, use_bias=True),
        BatchNormalization(epsilon=1e-5, momentum=0.9),
        LeakyReLU(negative_slope=0.1),
        Dense(32, use_bias=True),
        BatchNormalization(epsilon=1e-5, momentum=0.9),
        LeakyReLU(negative_slope=0.1),
        Dense(3, activation="softmax", use_bias=True),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_tr_scaled, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

# Save model and config
os.makedirs("models", exist_ok=True)
model.save("models/wine_model.keras")
with open("models/model_config.json", "w") as f:
    json.dump(model.get_config(), f, indent=2)

y_hat = model.predict(X_test_scaled)
y_hat = np.argmax(np.asarray(y_hat), axis=1)
y_test = np.argmax(np.asarray(y_test), axis=1)

y_hat = convert_label(y_hat)
y_test = convert_label(y_test)

eval_metrics = {
    "accuracy": float(accuracy_score(y_test, y_hat)),
    "precision": float(precision_score(y_test, y_hat, average="macro")),
    "recall": float(recall_score(y_test, y_hat, average="macro")),
    "f1_score": float(f1_score(y_test, y_hat, average="macro")),
}

# Save evaluation output
with open("models/evaluation.json", "w") as f:
    json.dump(eval_metrics, f, indent=2)

for metric, score in eval_metrics.items():
    print(f"{metric.capitalize()}: {score:.4f}")
