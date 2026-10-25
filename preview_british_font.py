from pathlib import Path

import cv2
import numpy as np

from font_renderer import render_character
from preview_german_fonts import make_preview
from segmentation import normalize_character, prepare_plate


def reference_character(character, font_path):
    # Ten sam seed zapewnia jednakowy rozmiar i grubość
    # przy porównywaniu dwóch znaków.
    rng = np.random.default_rng(42)

    image = render_character(
        character,
        rng,
        fonts=[str(font_path)],
    )

    binary = prepare_plate(image)
    points = cv2.findNonZero(binary)

    if points is None:
        raise ValueError(f"Nie udało się przygotować znaku: {character}")

    x, y, width, height = cv2.boundingRect(points)
    cropped = binary[y:y + height, x:x + width]

    return normalize_character(cropped)


def main():
    project_dir = Path(__file__).resolve().parent
    font_path = (
        project_dir
        / "data"
        / "fonts_uk"
        / "UKNumberPlate.ttf"
    )

    if not font_path.is_file():
        raise FileNotFoundError(
            f"Nie znaleziono fontu: {font_path}"
        )

    for first, second in [("O", "0"), ("I", "1")]:
        first_image = reference_character(first, font_path)
        second_image = reference_character(second, font_path)

        identical = np.array_equal(first_image, second_image)
        result = "IDENTYCZNE" if identical else "różne"

        print(
            f"Porównanie {first}/{second} "
            f"po normalizacji 32x32: {result}"
        )

    preview = make_preview(font_path)

    cv2.imshow("Brytyjskie znaki - UK Number Plate", preview)
    print("Naciśnij klawisz w oknie podglądu, aby zakończyć.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()