from pathlib import Path

import cv2

from recognition import load_character_model, recognize_characters

from segmentation import (
    normalize_character,
    prepare_plate,
    segment_characters,
)

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


def main():
    project_dir = Path(__file__).resolve().parent
    image_path = project_dir / "data" / "samochod.jpg"

    if not image_path.is_file():
        print(f"Nie znaleziono zdjęcia: {image_path}")
        return

    image = cv2.imread(str(image_path))

    if image is None:
        print("Nie udało się odczytać zdjęcia. Sprawdź format pliku.")
        return

    model_dir = project_dir / "models" / "20261021_220903"
    print(f"Wczytywany model: {model_dir}")
    model, class_names = load_character_model(model_dir)

    height, width = image.shape[:2]
    print(f"Wczytano zdjęcie: {image_path.name}")
    print(f"Wymiary: {width} x {height} pikseli")

    # Przygotowujemy pomniejszony obraz do podglądu i analizy.
    scale = min(1.0, 1200 / width, 800 / height)
    preview = image

    if scale < 1.0:
        preview = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    # Zamiana obrazu kolorowego na skalę szarości.
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)

    # Wygładzanie drobnych szczegółów i szumu.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Wykrywanie krawędzi algorytmem Canny'ego.
    edges = cv2.Canny(blurred, 80, 200)

    contour_count, candidates = find_plate_candidates(edges)

    print(f"Liczba znalezionych konturów: {contour_count}")
    print(f"Liczba kandydatów na tablicę: {len(candidates)}")

    # Rysujemy na kopii, aby zachować obraz do dalszej analizy.
    detection_preview = preview.copy()

    for number, polygon in enumerate(candidates, start=1):
        cv2.polylines(
            detection_preview,
            [polygon],
            True,
            (0, 255, 0),
            2,
        )

        x, y, width, height = cv2.boundingRect(polygon)

        cv2.putText(
            detection_preview,
            str(number),
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    if len(candidates) == 1:
        polygon = candidates[0]
        x, y, plate_width, plate_height = cv2.boundingRect(polygon)

        # Kontur znaleźliśmy na pomniejszonym obrazie.
        # Przeliczamy współrzędne na rozmiar oryginalnego zdjęcia.
        scale_x = image.shape[1] / preview.shape[1]
        scale_y = image.shape[0] / preview.shape[0]

        left = max(0, int(x * scale_x))
        top = max(0, int(y * scale_y))
        right = min(image.shape[1], int((x + plate_width) * scale_x) + 1)
        bottom = min(image.shape[0], int((y + plate_height) * scale_y) + 1)

        plate = image[top:bottom, left:right].copy()

        print(
            f"Wymiary wyciętej tablicy: "
            f"{plate.shape[1]} x {plate.shape[0]} pikseli"
        )

        # Powiększamy tylko podgląd wycinka.
        plate_preview = cv2.resize(
            plate,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )

        cv2.imshow("Wycieta tablica", plate_preview)

        binary = prepare_plate(plate)
        boxes, characters = segment_characters(binary)

        print(f"Liczba wydzielonych znaków: {len(characters)}")
        if characters:
            normalized_characters = []

            for character in characters:
                normalized = normalize_character(character)
                normalized_characters.append(normalized)

            plate_number, scores = recognize_characters(
                normalized_characters,
                model,
                class_names,
            )

            print(f"\nOdczytany numer: {plate_number}")
            print("Wyniki dla kolejnych znaków:")

            for position, (character, score) in enumerate(
                zip(plate_number, scores),
                start=1,
            ):
                print(
                    f"Pozycja {position}: {character}, "
                    f"wynik softmax: {score:.1%}"
                )

            # Łączymy obrazki w jeden poziomy pasek.
            characters_strip = cv2.hconcat(normalized_characters)

            # Powiększamy pasek wyłącznie do oglądania.
            characters_preview = cv2.resize(
                characters_strip,
                None,
                fx=4,
                fy=4,
                interpolation=cv2.INTER_NEAREST,
            )

            cv2.imshow("Znaki 32x32", characters_preview)

            print(
                f"Przygotowano {len(normalized_characters)} "
                "obrazków o rozmiarze 32 x 32 piksele."
            )

        segmentation_preview = cv2.cvtColor(
            binary,
            cv2.COLOR_GRAY2BGR,
        )

        for x, y, width, height in boxes:
            cv2.rectangle(
                segmentation_preview,
                (x, y),
                (x + width - 1, y + height - 1),
                (0, 255, 0),
                1,
            )

        binary_preview = cv2.resize(
            binary,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_NEAREST,
        )

        segmentation_preview = cv2.resize(
            segmentation_preview,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_NEAREST,
        )

        cv2.imshow("Tablica binarna", binary_preview)
        cv2.imshow("Podzial na znaki", segmentation_preview)

    elif len(candidates) == 0:
        print("Nie znaleziono kandydata na tablicę.")

    else:
        print("Znaleziono kilku kandydatów. Wybór tablicy dodamy później.")

    cv2.imshow("Zdjecie samochodu", preview)
    cv2.imshow("Skala szarosci", gray)
    cv2.imshow("Krawedzie", edges)
    cv2.imshow("Kandydaci na tablice", detection_preview)
    print("Naciśnij dowolny klawisz w oknie zdjęcia, aby zakończyć.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()