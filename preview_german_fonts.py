from pathlib import Path

import cv2
import numpy as np

from generate_dataset import CHARACTERS, create_example


def make_preview(font_path):
    rng = np.random.default_rng(42)
    tiles = []

    for character in CHARACTERS:
        # Korzystamy z generatora, ale wyłącznie
        # ze wskazanego niemieckiego fontu.
        example = create_example(
            character,
            rng,
            fonts=[str(font_path)],
            template_probability=0.0,
        )

        tile = np.full((64, 64, 3), 30, dtype=np.uint8)

        cv2.putText(
            tile,
            character,
            (4, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (220, 220, 220),
            1,
        )

        enlarged = cv2.resize(
            example,
            (48, 48),
            interpolation=cv2.INTER_NEAREST,
        )

        tile[14:62, 8:56] = cv2.cvtColor(
            enlarged,
            cv2.COLOR_GRAY2BGR,
        )

        tiles.append(tile)

    rows = []

    for start in range(0, len(tiles), 6):
        rows.append(cv2.hconcat(tiles[start:start + 6]))

    preview = cv2.vconcat(rows)

    return cv2.resize(
        preview,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_NEAREST,
    )


def main():
    project_dir = Path(__file__).resolve().parent
    fonts_dir = project_dir / "data" / "fonts_de" / "fonts" / "ttf"

    font_names = [
        "GL-Nummernschild-Mtl.ttf",
        "GL-Nummernschild-Eng.ttf",
    ]

    for font_name in font_names:
        font_path = fonts_dir / font_name

        if not font_path.is_file():
            raise FileNotFoundError(
                f"Nie znaleziono fontu: {font_path}"
            )

    for number, font_name in enumerate(font_names, start=1):
        font_path = fonts_dir / font_name
        preview = make_preview(font_path)

        print(f"Podgląd {number}/2: {font_name}")
        print(
            "Naciśnij klawisz w oknie podglądu, aby przejść dalej. "
            "Esc kończy podgląd."
        )

        cv2.imshow(
            f"Niemieckie znaki - {font_path.stem}",
            preview,
        )

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == 27:
            break


if __name__ == "__main__":
    main()