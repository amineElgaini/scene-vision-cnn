import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = 150
MODEL_PATH = Path("models/final_model.keras")
CLASSES_PATH = Path("models/class_names.json")


def load_image(path):
    img = tf.io.read_file(str(path))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img / 255.0


def predict_image(path, model, class_names):
    img = tf.expand_dims(load_image(path), 0)
    probs = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the scene of an image.")
    parser.add_argument("image", help="Path to a JPEG image")
    args = parser.parse_args()

    model = keras.models.load_model(MODEL_PATH)
    class_names = json.loads(CLASSES_PATH.read_text())

    label, prob = predict_image(args.image, model, class_names)
    print(f"Prediction: {label} ({prob:.1%})")