import json
import os

# Set backend before importing Keras
os.environ["KERAS_BACKEND"] = "torch"

import tempfile
import uuid

import keras
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import BatchNormalization, Dense, LeakyReLU
from keras.utils import to_categorical
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

BATCH_SIZE = 16
EPOCHS = 25
ALPHA = 0.001
LAMBDA = 1e-3

LABEL_MAP = {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}
REV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def data_label(y):
    return np.array([LABEL_MAP.get(val, val) for val in y], dtype=int)


def convert_label(y):
    return np.array([REV_LABEL_MAP.get(val, val) for val in y])


def save_evaluation(metrics, filepath="models/cars-evaluation.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    history = []
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                history = data if isinstance(data, list) else [data]
        except (json.JSONDecodeError, UnicodeDecodeError):
            history = []

    history.append(metrics)

    dir_name = os.path.dirname(filepath)
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as tf:
        json.dump(history, tf, indent=2)
        temp_name = tf.name

    os.replace(temp_name, filepath)


raw_df = pd.read_csv("data/cars/raw/cars.csv", index_col="stt")
feature_cols = raw_df.columns[:-1]
target_col = raw_df.columns[-1]
df = raw_df.drop_duplicates(subset=feature_cols, keep="first").reset_index(drop=True)

train, test = train_test_split(df, test_size=0.3, shuffle=True, stratify=df[target_col])

# Save train and test data splits
os.makedirs("data/cars/processed", exist_ok=True)
train.to_csv("data/cars/processed/train.csv", index=False)
test.to_csv("data/cars/processed/test.csv", index=False)

ordinal_cols = ["buying", "maint", "lug_boot", "safety"]
nominal_cols = ["doors", "persons"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "ordinal",
            Pipeline(
                [
                    ("encoder", OrdinalEncoder()),
                    ("scaler", StandardScaler()),
                ]
            ),
            ordinal_cols,
        ),
        (
            "onehot",
            OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            nominal_cols,
        ),
    ]
)

X_tr = preprocessor.fit_transform(train[feature_cols])
X_test = preprocessor.transform(test[feature_cols])

y_tr = np.array(train[target_col])
y_test = np.array(test[target_col])

y_tr, y_test = data_label(y_tr), data_label(y_test)

model = Sequential(
    [
        keras.Input(shape=(X_tr.shape[1],)),
        Dense(64, activation="relu", use_bias=True),
        Dense(32, activation="relu", use_bias=True),
        Dense(1, activation="sigmoid", use_bias=True),
        # Dense(64, use_bias=True),
        # BatchNormalization(epsilon=1e-5, momentum=0.9),
        # LeakyReLU(negative_slope=0.1),
        # Dense(32, use_bias=True),
        # BatchNormalization(epsilon=1e-5, momentum=0.9),
        # LeakyReLU(negative_slope=0.1),
        # Dense(3, activation="softmax", use_bias=True),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

# Save model and config
os.makedirs("models", exist_ok=True)
model.save("models/cars_model.keras")
with open("models/cars_model_config.json", "w") as f:
    json.dump(model.get_config(), f, indent=2)

y_hat = model.predict(X_test)
y_hat = (np.asarray(y_hat) >= 0.5).astype(int).flatten()

eval_metrics = {
    "attempt": str(uuid.uuid4()),
    "accuracy": float(accuracy_score(y_test, y_hat)),
    "precision": float(precision_score(y_test, y_hat)),
    "recall": float(recall_score(y_test, y_hat)),
    "f1_score": float(f1_score(y_test, y_hat)),
}

# Save evaluation output
save_evaluation(eval_metrics, "models/cars-evaluation.json")
