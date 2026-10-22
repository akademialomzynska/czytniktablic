import cv2
import numpy as np
from PIL import Image

from plate_detection import (
    find_bright_candidates,
    find_plate_candidates,
    find_text_candidates,
    select_plate_candidate,
)
from recognition import load_character_model, recognize_characters
from segmentation import (
    normalize_character,
    prepare_plate,
    segment_characters,
)
from perspective import find_perspective_candidates


class PlateReader:
    def __init__(self, model_dir):
        # Model wczytujemy raz przy tworzeniu czytnika.
        self.model, self.class_names = load_character_model(model_dir)

    def read(self, image_rgb):
        # GUI przekazuje obraz Pillow w kolorach RGB.
        # OpenCV pracuje tutaj na obrazie BGR.
        image = cv2.cvtColor(
            np.array(image_rgb.convert("RGB")),
            cv2.COLOR_RGB2BGR,
        )

        height, width = image.shape[:2]
        scale = min(1.0, 1200 / width, 800 / height)
        preview = image

        if scale < 1.0:
            preview = cv2.resize(
                image,
                (
                    max(1, round(width * scale)),
                    max(1, round(height * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )

        gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 80, 200)

        _, candidates = find_plate_candidates(edges)

        text_candidates, _ = find_text_candidates(gray)
        candidates.extend(text_candidates)

        bright_candidates = find_bright_candidates(gray)
        candidates.extend(bright_candidates)

        selected = select_plate_candidate(
            image,
            preview,
            candidates,
        )

        # Jeśli zwykłe wycinanie nie dało wyniku,
        # sprawdzamy czworokąty po korekcji perspektywy.
        if selected is None:
            perspective_candidates = find_perspective_candidates(
                gray
            )

            print(
                "Próba korekcji perspektywy: "
                f"{len(perspective_candidates)} kandydatów"
            )

            selected = select_plate_candidate(
                image,
                preview,
                perspective_candidates,
                rectify=True,
            )

        if selected is None:
            return None

        plate = selected["plate"]
        binary = prepare_plate(plate)
        _, characters = segment_characters(binary)

        if not characters:
            return None

        normalized_characters = [
            normalize_character(character)
            for character in characters
        ]

        number, scores = recognize_characters(
            normalized_characters,
            self.model,
            self.class_names,
        )

        plate_rgb = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)

        return {
            "number": number,
            "scores": scores,
            "plate": Image.fromarray(plate_rgb),
        }