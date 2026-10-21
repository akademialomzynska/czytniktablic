import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def load_character_model(model_dir):
    model_dir = Path(model_dir)
    model_path = model_dir / "characters.keras"
    classes_path = model_dir / "classes.json"

    if not model_path.is_file() or not classes_path.is_file():
        raise FileNotFoundError(
            f"Brak modelu lub pliku classes.json w: {model_dir}"
        )

    class_names = json.loads(
        classes_path.read_text(encoding="utf-8")
    )

    model = tf.keras.models.load_model(
        str(model_path),
        compile=False,
    )

    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            "Liczba klas nie odpowiada liczbie wyjść modelu."
        )

    return model, class_names


def recognize_characters(characters, model, class_names):
    if not characters:
        return "", []

    if any(character.shape != (32, 32) for character in characters):
        raise ValueError("Każdy znak musi mieć rozmiar 32 x 32.")

    # Tworzymy paczkę: liczba znaków, wysokość, szerokość.
    batch = np.stack(characters).astype(np.float32)

    # Dodajemy wymiar pojedynczego kanału jasności.
    batch = batch[..., np.newaxis]

    # Przeliczanie pikseli na 0–1 jest już częścią modelu.
    predictions = model.predict(batch, verbose=0)

    recognized_characters = []
    scores = []

    for prediction in predictions:
        class_index = int(np.argmax(prediction))

        recognized_characters.append(class_names[class_index])
        scores.append(float(prediction[class_index]))

    return "".join(recognized_characters), scores