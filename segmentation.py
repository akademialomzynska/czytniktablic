import cv2

import numpy as np


def prepare_plate(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )

    return binary


def segment_characters(binary):
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    plate_height = binary.shape[0]
    boxes = []

    # Element 0 opisuje tło, więc zaczynamy od elementu 1.
    for component in range(1, component_count):
        x, y, width, height, area = stats[component]

        relative_height = height / plate_height
        aspect_ratio = width / height
        fill_ratio = area / (width * height)

        # Znak powinien zajmować znaczną część wysokości tablicy.
        if not 0.45 <= relative_height <= 0.95:
            continue

        # Zachowujemy również wąskie znaki, takie jak cyfra 1.
        if not 0.08 <= aspect_ratio <= 0.95:
            continue

        # Odrzucamy bardzo rzadkie skupiska pikseli.
        if fill_ratio < 0.15:
            continue

        boxes.append((int(x), int(y), int(width), int(height)))

    # Kolejność czytania: od lewej do prawej.
    boxes.sort(key=lambda box: box[0])

    characters = []

    for x, y, width, height in boxes:
        character = binary[y:y + height, x:x + width].copy()
        characters.append(character)

    return boxes, characters

def normalize_character(character, size=32, padding=4):
    height, width = character.shape
    available_size = size - 2 * padding

    # Dobieramy skalę z zachowaniem proporcji znaku.
    scale = min(
        available_size / width,
        available_size / height,
    )

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized = cv2.resize(
        character,
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST,
    )

    # Tworzymy czarne tło.
    canvas = np.zeros((size, size), dtype=np.uint8)

    # Umieszczamy znak na środku obrazka.
    left = (size - new_width) // 2
    top = (size - new_height) // 2

    canvas[
        top:top + new_height,
        left:left + new_width,
    ] = resized

    return canvas