import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from segmentation import normalize_character, prepare_plate
from font_renderer import render_character
from template_renderer import render_template

CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def create_example(character, rng, fonts, template_probability=0.0):
    canvas = None

    if template_probability > 0 and rng.random() < template_probability:
        canvas = render_template(character, rng)

    if canvas is None:
        canvas = render_character(character, rng, fonts)

        # Niewielkie nachylenie znaku.
    angle = float(rng.uniform(-8.0, 8.0))
    center = (canvas.shape[1] / 2, canvas.shape[0] / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    canvas = cv2.warpAffine(
        canvas,
        rotation_matrix,
        (canvas.shape[1], canvas.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    # Drobne zakłócenia jasności pikseli.
    noise_strength = float(rng.uniform(0.0, 8.0))
    noise = rng.normal(
        0.0,
        noise_strength,
        canvas.shape,
    ).astype(np.float32)

    canvas = np.clip(
        canvas.astype(np.float32) + noise,
        0,
        255,
    ).astype(np.uint8)

    # Używamy tych samych funkcji co przy prawdziwej tablicy.
    binary = prepare_plate(canvas)

    points = cv2.findNonZero(binary)

    if points is None:
        raise ValueError(f"Nie udało się wygenerować znaku: {character}")

    x, y, width, height = cv2.boundingRect(points)
    cropped = binary[y:y + height, x:x + width]

    return normalize_character(cropped)


def generate_split(output_dir, split_name, settings, used_hashes):
    rng = np.random.default_rng(settings["seed"])
    samples_per_class = settings["samples_per_class"]
    total = 0

    for character in CHARACTERS:
        class_dir = output_dir / split_name / character
        class_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        attempts = 0

        while saved < samples_per_class:
            attempts += 1

            if attempts > samples_per_class * 100:
                raise RuntimeError(
                    f"Za mało różnych wariantów dla: "
                    f"{split_name}/{character}"
                )

            example = create_example(
                character,
                rng,
                settings["fonts"],
                template_probability=settings.get(
                    "template_probability", 0.0
                ),
            )

            # Sprawdzamy zawartość pikseli, a nie nazwę pliku.
            fingerprint = hashlib.sha256(example.tobytes()).digest()

            if fingerprint in used_hashes:
                continue

            file_path = class_dir / f"{saved:04d}.png"

            if not cv2.imwrite(str(file_path), example):
                raise OSError(f"Nie udało się zapisać: {file_path}")

            used_hashes.add(fingerprint)
            saved += 1

        total += saved
        print(f"{split_name}: znak {character}, zapisano {saved}")

    return total


def main():
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "data" / "synthetic_v3"

    # Każdy nowy zbiór otrzymuje osobny folder.
    if output_dir.exists():
        print(f"Folder już istnieje: {output_dir}")
        print("Przerwano, aby zachować dotychczasowy zbiór.")
        return

    settings = {
        "class_names": list(CHARACTERS),
        "image_size": [32, 32],
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "train": {
            "samples_per_class": 200,
            "seed": 42,
            "template_probability": 0.5,
            "fonts": [
                r"C:\Windows\Fonts\arial.ttf",
                r"C:\Windows\Fonts\arialbd.ttf",
                r"C:\Windows\Fonts\ARIALN.TTF",
                r"C:\Windows\Fonts\ARIALNB.TTF",
                r"C:\Windows\Fonts\consola.ttf",
                r"C:\Windows\Fonts\consolab.ttf",
            ],
        },
        "validation": {
            "samples_per_class": 50,
            "seed": 2026,
            "template_probability": 0.0,
            "fonts": [
                r"C:\Windows\Fonts\bahnschrift.ttf",
            ],
        },
    }

    output_dir.mkdir(parents=True)
    used_hashes = set()
    totals = {}

    for split_name in ("train", "validation"):
        totals[split_name] = generate_split(
            output_dir,
            split_name,
            settings[split_name],
            used_hashes,
        )

    settings["totals"] = totals

    info_path = output_dir / "dataset_info.json"
    info_path.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nGenerowanie zakończone.")
    print(f"Zbiór treningowy: {totals['train']}")
    print(f"Zbiór walidacyjny: {totals['validation']}")
    print(f"Razem: {sum(totals.values())}")
    print(f"Opis zbioru: {info_path}")


if __name__ == "__main__":
    main()