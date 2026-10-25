import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from generate_dataset import CHARACTERS, generate_split


def main():
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "data" / "synthetic_v5_pl_de_uk"

    if output_dir.exists():
        print(f"Folder już istnieje: {output_dir}")
        print("Przerwano, aby zachować dotychczasowe dane.")
        return

    german_repo = project_dir / "data" / "fonts_de"
    german_dir = german_repo / "fonts" / "ttf"

    british_font = (
        project_dir
        / "data"
        / "fonts_uk"
        / "UKNumberPlate.ttf"
    )

    plate_fonts = [
        str(german_dir / "GL-Nummernschild-Mtl.ttf"),
        str(german_dir / "GL-Nummernschild-Eng.ttf"),
        str(british_font),
    ]

    train_fonts = plate_fonts + [
        r"C:\Windows\Fonts\ARIALN.TTF",
    ]

    validation_fonts = plate_fonts + [
        r"C:\Windows\Fonts\bahnschrift.ttf",
    ]

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
            str(german_repo),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()

    british_font_hash = hashlib.sha256(
        british_font.read_bytes()
    ).hexdigest()

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
                "license_file": str(german_repo / "LICENSE.txt"),
            },
            "british_font": {
                "author": "Gareth Attrill",
                "url": (
                    "https://www.attrill.uk/"
                    "fonts/uknumberplate/"
                ),
                "file": str(british_font),
                "sha256": british_font_hash,
            },
        },
        "validation_note": (
            "Nowe warianty syntetyczne znanych krojów tablic. "
            "Nie jest to niezależny test na rzeczywistych zdjęciach."
        ),
        "train": {
            "samples_per_class": 800,
            "seed": 42,
            "template_probability": 0.5,
            "fonts": train_fonts,
        },
        "validation": {
            "samples_per_class": 200,
            "seed": 2026,
            "template_probability": 0.5,
            "fonts": validation_fonts,
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