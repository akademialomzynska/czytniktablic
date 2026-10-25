import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from generate_dataset import CHARACTERS, generate_split


def main():
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "data" / "synthetic_v4_pl_de"

    if output_dir.exists():
        print(f"Folder już istnieje: {output_dir}")
        print("Przerwano, aby zachować dotychczasowe dane.")
        return

    fonts_repo = project_dir / "data" / "fonts_de"
    fonts_dir = fonts_repo / "fonts" / "ttf"

    german_fonts = [
        str(fonts_dir / "GL-Nummernschild-Mtl.ttf"),
        str(fonts_dir / "GL-Nummernschild-Eng.ttf"),
    ]

    train_fonts = german_fonts + [
        r"C:\Windows\Fonts\ARIALN.TTF",
    ]

    validation_fonts = german_fonts + [
        r"C:\Windows\Fonts\bahnschrift.ttf",
    ]

    # Sprawdzamy fonty przed rozpoczęciem generowania.
    for font_path in set(train_fonts + validation_fonts):
        if not Path(font_path).is_file():
            raise FileNotFoundError(
                f"Nie znaleziono fontu: {font_path}"
            )

    templates_dir = (
        project_dir
        / "data"
        / "plate_templates_source"
        / "images"
        / "z_samochodowe"
    )

    if not templates_dir.is_dir():
        raise FileNotFoundError(
            f"Brak polskich wzorców: {templates_dir}"
        )

    german_revision = subprocess.check_output(
        [
            "git",
            "-C",
            str(fonts_repo),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()

    settings = {
        "class_names": list(CHARACTERS),
        "image_size": [32, 32],
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "sources": {
            "polish_templates": {
                "url": (
                    "https://github.com/"
                    "marcinos2079/polskie-tablice"
                ),
            },
            "german_fonts": {
                "url": (
                    "https://github.com/"
                    "Gutenberg-Labo/GL-Nummernschild"
                ),
                "revision": german_revision,
                "license_file": str(fonts_repo / "LICENSE.txt"),
            },
        },
        "validation_note": (
            "Nowe warianty syntetyczne znanych krojów tablic. "
            "Nie jest to niezależny test na rzeczywistych zdjęciach."
        ),
        "train": {
            "samples_per_class": 600,
            "seed": 42,
            "template_probability": 0.5,
            "fonts": train_fonts,
        },
        "validation": {
            "samples_per_class": 150,
            "seed": 2026,
            "template_probability": 0.5,
            "fonts": validation_fonts,
        },
    }

    output_dir.mkdir(parents=True)

    # Wspólny zbiór skrótów zapobiega zapisaniu identycznych
    # obrazków w treningu i walidacji.
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