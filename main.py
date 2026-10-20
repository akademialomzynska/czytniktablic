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

    # Zmniejszamy wyłącznie podgląd, zachowując proporcje.
    scale = min(1.0, 1200 / width, 800 / height)
    preview = image

    if scale < 1.0:
        preview = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    cv2.imshow("Zdjecie samochodu", preview)
    print("Naciśnij dowolny klawisz w oknie zdjęcia, aby zakończyć.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()