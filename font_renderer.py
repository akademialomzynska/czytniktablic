from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=256)
def load_font(font_path, font_size):
    return ImageFont.truetype(font_path, font_size)


def render_character(character, rng, fonts):
    font_path = str(rng.choice(fonts))
    font_size = int(rng.integers(60, 91))
    stroke_width = int(rng.integers(0, 2))

    font = load_font(font_path, font_size)

    image = Image.new("L", (128, 128), color=255)
    draw = ImageDraw.Draw(image)

    # Wyznaczamy rzeczywisty obszar zajmowany przez znak.
    left, top, right, bottom = draw.textbbox(
        (0, 0),
        character,
        font=font,
        stroke_width=stroke_width,
    )

    width = right - left
    height = bottom - top

    position = (
        (128 - width) // 2 - left,
        (128 - height) // 2 - top,
    )

    draw.text(
        position,
        character,
        font=font,
        fill=0,
        stroke_width=stroke_width,
        stroke_fill=0,
    )

    return cv2.cvtColor(
        np.array(image),
        cv2.COLOR_GRAY2BGR,
    )