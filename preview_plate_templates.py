from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from segmentation import normalize_character, prepare_plate


def load_template(file_path):
    # Nakładamy ewentualną przezroczystość na białe tło.
    with Image.open(file_path) as source:
        image = source.convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(background, image).convert("RGB")

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    binary = prepare_plate(image)

    points = cv2.findNonZero(binary)

    if points is None:
        raise ValueError(f"Nie znaleziono ciemnego znaku: {file_path}")

    x, y, width, height = cv2.boundingRect(points)
    cropped = binary[y:y + height, x:x + width]

    return normalize_character(cropped)


def main():
    project_dir = Path(__file__).resolve().parent
    templates_dir = (
        project_dir
        / "data"
        / "plate_templates_source"
        / "images"
        / "z_samochodowe"
    )

    files = (
        sorted(templates_dir.glob("W*.png"))
        + sorted(templates_dir.glob("N*.png"))
    )

    if not files:
        print(f"Nie znaleziono wzorców w: {templates_dir}")
        return

    tiles = []

    for file_path in files:
        character = load_template(file_path)

        tile = np.full((64, 64, 3), 30, dtype=np.uint8)
        tile[24:56, 16:48] = cv2.cvtColor(
            character,
            cv2.COLOR_GRAY2BGR,
        )

        cv2.putText(
            tile,
            file_path.name,
            (2, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
        )

        tiles.append(tile)
        print(f"Wczytano: {file_path.name}")

    preview = cv2.hconcat(tiles)
    preview = cv2.resize(
        preview,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_NEAREST,
    )

    cv2.imshow("W i N - wzorce tablic", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()