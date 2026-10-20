from pathlib import Path

import cv2


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

    cv2.imshow("Zdjecie samochodu", preview)
    cv2.imshow("Skala szarosci", gray)
    cv2.imshow("Krawedzie", edges)
    print("Naciśnij dowolny klawisz w oknie zdjęcia, aby zakończyć.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()