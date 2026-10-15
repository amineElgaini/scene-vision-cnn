import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from tensorflow import keras

from src.predict import CLASSES_PATH, MODEL_PATH, load_image, predict_image


def make_image(tmp_path, size=(150, 150)):
    path = tmp_path / "fake.jpg"
    array = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(array).save(path)
    return path


def test_load_image_shape_and_range(tmp_path):
    img = load_image(make_image(tmp_path))
    assert img.shape == (150, 150, 3)
    assert float(img.numpy().min()) >= 0.0
    assert float(img.numpy().max()) <= 1.0


def test_load_image_resizes_other_sizes(tmp_path):
    img = load_image(make_image(tmp_path, size=(150, 113)))
    assert img.shape == (150, 150, 3)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model file not found")
def test_predict_image_output(tmp_path):
    model = keras.models.load_model(MODEL_PATH)
    class_names = json.loads(CLASSES_PATH.read_text())
    label, prob = predict_image(make_image(tmp_path), model, class_names)
    assert label in class_names
    assert 0.0 <= prob <= 1.0