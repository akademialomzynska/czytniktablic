from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


TEMPLATES_DIR = (
    Path(__file__).resolve().parent
    / "data"
    / "plate_templates_source"
    / "images"
    / "z_samochodowe"
)


@lru_cache(maxsize=36)
def get_template_paths(character):
    if not TEMPLATES_DIR.is_dir():
        raise FileNotFoundError(f"Brak folderu: {TEMPLATES_DIR}")

    return tuple(sorted(TEMPLATES_DIR.glob(f"{character}*.png")))


@lru_cache(maxsize=256)
def load_glyph(file_path):
    with Image.open(file_path) as source:
        image = source.convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        gray = np.array(
            Image.alpha_composite(background, image).convert("L")
        )

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )

    points = cv2.findNonZero(mask)

    if points is None:
        raise ValueError(f"Brak znaku we wzorcu: {file_path}")

    x, y, width, height = cv2.boundingRect(points)

    # Zwracamy jasny znak na ciemnym tle.
    return 255 - gray[y:y + height, x:x + width]


def render_template(character, rng):
    files = get_template_paths(character)

    # Dla brakujących wzorców generator wykorzysta czcionki.
    if not files:
        return None

    file_path = files[int(rng.integers(len(files)))]
    glyph = load_glyph(file_path)

    height, width = glyph.shape
    target_height = int(rng.integers(60, 91))
    scale = min(target_height / height, 100 / width)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized = cv2.resize(
        glyph,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.full((128, 128), 255, dtype=np.uint8)
    left = (128 - new_width) // 2
    top = (128 - new_height) // 2

    # Umieszczamy ciemny znak na jasnym tle.
    canvas[
        top:top + new_height,
        left:left + new_width,
    ] = 255 - resized

    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)