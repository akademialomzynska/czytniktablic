import cv2
import numpy as np

from segmentation import prepare_plate, segment_characters

def find_text_candidates(gray):
    image_height, image_width = gray.shape
    image_area = image_height * image_width

    # Uwydatniamy ciemne szczegóły na jasnym tle.
    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (17, 7),
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        blackhat_kernel,
    )

    _, binary = cv2.threshold(
        blackhat,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )

    # Łączymy pobliskie elementy w poziome grupy.
    joining_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (13, 3),
    )

    text_mask = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        joining_kernel,
    )

    contours, _ = cv2.findContours(
        text_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)

        if height < 8 or width < 30:
            continue

        aspect_ratio = width / height
        relative_area = (width * height) / image_area

        if not 2.0 <= aspect_ratio <= 8.0:
            continue

        if not 0.0005 <= relative_area <= 0.10:
            continue

        # Dodajemy margines wokół znalezionej grupy znaków.
        padding_x = max(2, round(width * 0.08))
        padding_y = max(2, round(height * 0.35))

        left = max(0, x - padding_x)
        top = max(0, y - padding_y)
        right = min(image_width - 1, x + width + padding_x)
        bottom = min(image_height - 1, y + height + padding_y)

        polygon = np.array(
            [
                [[left, top]],
                [[right, top]],
                [[right, bottom]],
                [[left, bottom]],
            ],
            dtype=np.int32,
        )

        candidates.append(polygon)

    return candidates, text_mask

def find_bright_candidates(gray):
    image_height, image_width = gray.shape
    image_area = image_height * image_width

    _, bright_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )

    contours, _ = cv2.findContours(
        bright_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)

        if width < 30 or height < 8:
            continue

        aspect_ratio = width / height
        relative_area = width * height / image_area
        fill_ratio = cv2.contourArea(contour) / (width * height)

        if not 2.0 <= aspect_ratio <= 8.0:
            continue

        if not 0.0005 <= relative_area <= 0.10:
            continue

        if fill_ratio < 0.55:
            continue

        padding = max(1, round(height * 0.08))

        left = max(0, x - padding)
        top = max(0, y - padding)
        right = min(image_width - 1, x + width + padding)
        bottom = min(image_height - 1, y + height + padding)

        polygon = np.array(
            [
                [[left, top]],
                [[right, top]],
                [[right, bottom]],
                [[left, bottom]],
            ],
            dtype=np.int32,
        )

        candidates.append(polygon)

    return candidates


def select_plate_candidate(image, preview, candidates):
    scale_x = image.shape[1] / preview.shape[1]
    scale_y = image.shape[0] / preview.shape[0]

    best = None

    for number, polygon in enumerate(candidates, start=1):
        x, y, width, height = cv2.boundingRect(polygon)

        left = max(0, int(np.floor(x * scale_x)))
        top = max(0, int(np.floor(y * scale_y)))
        right = min(
            image.shape[1],
            int(np.ceil((x + width) * scale_x)),
        )
        bottom = min(
            image.shape[0],
            int(np.ceil((y + height) * scale_y)),
        )

        if right <= left or bottom <= top:
            continue

        plate = image[top:bottom, left:right].copy()
        binary = prepare_plate(plate)
        boxes, _ = segment_characters(binary)
        count = len(boxes)

        print(f"Kandydat {number}: {count} znaków")

        if not 4 <= count <= 10:
            continue

        foreground_ratio = np.count_nonzero(binary) / binary.size

        if not 0.08 <= foreground_ratio <= 0.65:
            continue

        heights = np.array(
            [box[3] for box in boxes],
            dtype=float,
        )
        centers = np.array(
            [box[1] + box[3] / 2 for box in boxes],
            dtype=float,
        )

        height_variation = heights.std() / heights.mean()
        line_variation = centers.std() / heights.mean()

        coverage = (
            boxes[-1][0] + boxes[-1][2] - boxes[0][0]
        ) / plate.shape[1]

        if height_variation > 0.35 or line_variation > 0.30:
            continue

        if coverage < 0.45:
            continue

        score = (
            min(count, 8)
            + 2 * coverage
            - 3 * height_variation
            - 2 * line_variation
        )

        if best is None or score > best["score"]:
            best = {
                "number": number,
                "polygon": polygon,
                "plate": plate,
                "score": float(score),
            }

    return best

def find_plate_candidates(edges):
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    image_height, image_width = edges.shape
    image_area = image_height * image_width
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Pomijamy obiekty zbyt małe lub zbyt duże względem zdjęcia.
        if not 0.001 <= area / image_area <= 0.15:
            continue

        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Szukamy wypukłych czworokątów.
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue

        x, y, width, height = cv2.boundingRect(polygon)
        aspect_ratio = width / height

        # Tablica jednorzędowa jest wyraźnie szersza niż wyższa.
        if not 2.0 <= aspect_ratio <= 6.0:
            continue

        # Odrzucamy kontury słabo wypełniające otaczający prostokąt.
        rectangularity = area / (width * height)

        if rectangularity < 0.65:
            continue

        candidates.append(polygon)

    candidates.sort(key=cv2.contourArea, reverse=True)

    return len(contours), candidates