import cv2
import numpy as np


def prepare_plate(plate):
    # Powiększamy małe wycinki przed filtrowaniem,
    # aby filtr działał łagodniej względem rozmiaru znaków.
    if plate.shape[0] < 64:
        plate = cv2.resize(
            plate,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(
        hsv,
        np.array([15, 80, 60], dtype=np.uint8),
        np.array([40, 255, 255], dtype=np.uint8),
    )

    yellow_ratio = (
        np.count_nonzero(yellow_mask) / yellow_mask.size
    )

    if yellow_ratio > 0.25:
        # Lokalny próg pomaga oddzielić znaki od ramki
        # i nierównomiernie oświetlonego żółtego tła.
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        )
    else:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        _, binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        )

    return correct_character_slant(binary)


def segment_characters(binary):
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    plate_height, plate_width = binary.shape
    boxes = []

    for component in range(1, component_count):
        x, y, width, height, area = stats[component]

        # Pomijamy elementy dotykające brzegu wycinka.
        # Mogą pochodzić z ramki lub niebieskiego paska.
        if (
            x == 0
            or y == 0
            or x + width >= plate_width
            or y + height >= plate_height
        ):
            continue

        relative_height = height / plate_height
        aspect_ratio = width / height
        fill_ratio = area / (width * height)

        if not 0.45 <= relative_height <= 0.95:
            continue

        if not 0.08 <= aspect_ratio <= 0.95:
            continue

        if fill_ratio < 0.15:
            continue

        boxes.append((
            int(x),
            int(y),
            int(width),
            int(height),
        ))

    boxes.sort(key=lambda box: box[0])

    characters = []

    for x, y, width, height in boxes:
        character = binary[
            y:y + height,
            x:x + width,
        ].copy()
        characters.append(character)

    return boxes, characters


def correct_character_slant(binary):
    # Wstępnie wydzielamy znaki, aby oszacować ich wspólny skos.
    _, characters = segment_characters(binary)

    if len(characters) < 4:
        return binary

    slopes = []

    for character in characters:
        moments = cv2.moments(character)

        if abs(moments["mu02"]) < 0.01:
            continue

        slope = moments["mu11"] / moments["mu02"]
        slopes.append(slope)

    if not slopes:
        return binary

    # Mediana ogranicza wpływ pojedynczych nietypowych znaków.
    slope = float(np.clip(np.median(slopes), -0.5, 0.5))

    # Pomijamy bardzo małe korekty.
    if abs(slope) < 0.05:
        return binary

    height, width = binary.shape
    center_y = (height - 1) / 2

    matrix = np.array(
        [
            [1, -slope, slope * center_y],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )

    return cv2.warpAffine(
        binary,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def normalize_character(character, size=32, padding=4):
    height, width = character.shape
    available_size = size - 2 * padding

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

    canvas = np.zeros((size, size), dtype=np.uint8)

    left = (size - new_width) // 2
    top = (size - new_height) // 2

    canvas[
        top:top + new_height,
        left:left + new_width,
    ] = resized

    return canvas