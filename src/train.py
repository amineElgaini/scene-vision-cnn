import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.predict import load_image

SPLITS_DIR = Path("data/splits")
CLASSES_PATH = Path("models/class_names.json")
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def make_dataset(df, training=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (df["path"].values, df["label_id"].values)
    )
    if training:
        ds = ds.shuffle(len(df), seed=42)
    ds = ds.map(lambda p, y: (load_image(p), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)

    if training and augment:
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.1),
        ])

        def augment_batch(x, y):
            x = data_augmentation(x, training=True)
            return tf.clip_by_value(x, 0.0, 1.0), y

        ds = ds.map(augment_batch, num_parallel_calls=AUTOTUNE)

    return ds.prefetch(AUTOTUNE)


def build_model(filters=(32, 64, 128, 256), dropout=0.3, lr=0.001):
    model = keras.Sequential([keras.Input(shape=(150, 150, 3))])
    for f in filters:
        model.add(layers.Conv2D(f, 3))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(2))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(6, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the CNN.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--augment", action="store_true",
                        help="use Data Augmentation on the train set")
    parser.add_argument("--output", default="models/final_model.keras")
    args = parser.parse_args()

    df_train = pd.read_csv(SPLITS_DIR / "train.csv")
    df_val = pd.read_csv(SPLITS_DIR / "val.csv")
    df_test = pd.read_csv(SPLITS_DIR / "test.csv")

    class_names = sorted(df_train["label"].unique())
    CLASSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASSES_PATH.write_text(json.dumps(class_names))

    train_ds = make_dataset(df_train, training=True, augment=args.augment)
    val_ds = make_dataset(df_val)
    test_ds = make_dataset(df_test)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = build_model()
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(
            str(output), monitor="val_loss", save_best_only=True),
    ]

    model.fit(train_ds, validation_data=val_ds,
              epochs=args.epochs, callbacks=callbacks)

    model.save(output)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test loss: {test_loss:.3f}  Test accuracy: {test_acc:.3f}")
    print("Model saved to", output)


if __name__ == "__main__":
    main()