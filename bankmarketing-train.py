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


raw_df = pd.read_csv("data/bankmarketing/raw/bankmarketing.csv")
feature_cols = raw_df.columns[:-1]
target_col = raw_df.columns[-1]
df = raw_df.drop_duplicates(subset=feature_cols, keep="first").reset_index(drop=True)

train, test = train_test_split(df, test_size=0.3, shuffle=True, stratify=df[target_col])

# Save train and test data splits
os.makedirs("data/bankmarketing/processed", exist_ok=True)
train.to_csv("data/bankmarketing/processed/train.csv", index=False)
test.to_csv("data/bankmarketing/processed/test.csv", index=False)

numeric_cols = ["age", "duration"]
nominal_cols = ["education", "marital", "housing"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        (
            "cat",
            OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            nominal_cols,
        ),
    ]
)

X_tr = preprocessor.fit_transform(train[feature_cols])
X_test = preprocessor.transform(test[feature_cols])

y_tr = np.array(train[target_col])
y_test = np.array(test[target_col])

model = Sequential(
    [
        keras.Input(shape=(X_tr.shape[1],)),
        Dense(64, activation="relu", use_bias=True),
        BatchNormalization(epsilon=1e-5, momentum=0.9),
        LeakyReLU(negative_slope=0.1),
        Dense(32, activation="relu", use_bias=True),
        BatchNormalization(epsilon=1e-5, momentum=0.9),
        LeakyReLU(negative_slope=0.1),
        Dense(1, activation="sigmoid", use_bias=True),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

# Save model and config
os.makedirs("models", exist_ok=True)
model.save("models/bankmarket_model.keras")
with open("models/bankmarket_model_config.json", "w") as f:
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
save_evaluation(eval_metrics, "models/bankmarket-evaluation.json")
