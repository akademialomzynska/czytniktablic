import cv2
import numpy as np
from PIL import Image

from plate_detection import (
    find_bright_candidates,
    find_plate_candidates,
    find_text_candidates,
    select_plate_candidate,
)
from perspective import (
    find_edge_candidates,
    find_perspective_candidates,
)
from recognition import load_character_model, recognize_characters
from segmentation import (
    normalize_character,
    prepare_plate,
    segment_characters,
)


class PlateReader:
    def __init__(self, model_dir):
        self.model, self.class_names = load_character_model(model_dir)

    def read(self, image_rgb):
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
        candidates.extend(find_bright_candidates(gray))

        candidate_groups = [
            (
                "Zwykłe wycinanie",
                candidates,
                False,
            ),
            (
                "Perspektywa z jasnych obszarów",
                find_perspective_candidates(gray),
                True,
            ),
            (
                "Perspektywa z krawędzi",
                find_edge_candidates(edges),
                True,
            ),
        ]

        selected = None
        selected_method = ""

        for method, method_candidates, rectify in candidate_groups:
            print(
                f"{method}: "
                f"{len(method_candidates)} kandydatów"
            )

            current = select_plate_candidate(
                image,
                preview,
                method_candidates,
                rectify=rectify,
            )

            if current is None:
                continue

            if (
                selected is None
                or current["score"] > selected["score"]
            ):
                selected = current
                selected_method = method

        if selected is None:
            return None

        print(f"Wybrana metoda: {selected_method}")

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