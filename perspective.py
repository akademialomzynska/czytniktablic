import cv2
import numpy as np


def order_corners(polygon):
    points = polygon.reshape(4, 2).astype(np.float32)

    # Dzielimy narożniki na lewą i prawą stronę.
    points = points[np.argsort(points[:, 0])]
    left = points[:2]
    right = points[2:]

    # Na każdej stronie ustalamy punkt górny i dolny.
    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]

    top_left, bottom_left = left
    top_right, bottom_right = right

    return np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32,
    )


def find_perspective_candidates(gray):
    otsu_threshold, _ = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )

    # Sprawdzamy również jaśniejszy wariant maski.
    thresholds = sorted({
        int(otsu_threshold),
        min(254, int(otsu_threshold) + 40),
    })

    candidates = []
    seen = set()

    for threshold in thresholds:
        _, mask = cv2.threshold(
            gray,
            threshold,
            255,
            cv2.THRESH_BINARY,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            if not 0.0005 <= area / gray.size <= 0.10:
                continue

            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(
                contour,
                0.02 * perimeter,
                True,
            )

            if len(polygon) != 4:
                continue

            if not cv2.isContourConvex(polygon):
                continue

            top_left, top_right, bottom_right, bottom_left = (
                order_corners(polygon)
            )

            # Mierzymy boki czworokąta zamiast prostokąta
            # otaczającego pochyloną tablicę.
            width = max(
                np.linalg.norm(top_right - top_left),
                np.linalg.norm(bottom_right - bottom_left),
            )
            height = max(
                np.linalg.norm(bottom_left - top_left),
                np.linalg.norm(bottom_right - top_right),
            )

            if width < 30 or height < 10:
                continue

            if not 1.3 <= width / height <= 8.0:
                continue

            key = tuple(order_corners(polygon).ravel().tolist())

            if key in seen:
                continue

            seen.add(key)
            candidates.append(polygon)

    return candidates


def rectify_plate(image, polygon):
    source = order_corners(polygon)

    width = 360
    height = 80

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(
        source,
        destination,
    )

    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

def find_edge_candidates(edges):
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []
    seen = set()

    for contour in contours:
        # Obejmujemy zewnętrzne punkty poszarpanego konturu.
        hull = cv2.convexHull(contour)
        area = cv2.contourArea(hull)

        if not 0.0005 <= area / edges.size <= 0.10:
            continue

        perimeter = cv2.arcLength(hull, True)
        polygon = None

        # Najpierw próbujemy dotychczasowego uproszczenia.
        # Jeśli nie daje czworokąta, próbujemy mocniejszego.
        for factor in (0.02, 0.03):
            approximation = cv2.approxPolyDP(
                hull,
                factor * perimeter,
                True,
            )

            if (
                len(approximation) == 4
                and cv2.isContourConvex(approximation)
            ):
                polygon = approximation
                break

        if polygon is None:
            continue

        top_left, top_right, bottom_right, bottom_left = (
            order_corners(polygon)
        )

        width = max(
            np.linalg.norm(top_right - top_left),
            np.linalg.norm(bottom_right - bottom_left),
        )
        height = max(
            np.linalg.norm(bottom_left - top_left),
            np.linalg.norm(bottom_right - top_right),
        )

        if width < 30 or height < 10:
            continue

        if not 1.3 <= width / height <= 8.0:
            continue

        key = tuple(order_corners(polygon).ravel().tolist())

        if key in seen:
            continue

        seen.add(key)
        candidates.append(polygon)

    return candidates