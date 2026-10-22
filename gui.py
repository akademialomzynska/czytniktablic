from pathlib import Path
from queue import Empty, Queue
from threading import Thread
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk


class PlateReaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Czytnik tablic rejestracyjnych")
        self.root.geometry("1000x850")
        self.root.minsize(760, 700)

        self.image_path = None
        self.original_image = None
        self.preview_photo = None
        self.plate_photo = None

        self.reader = None
        self.busy = False
        self.result_queue = Queue()

        container = ttk.Frame(root, padding=16)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="Czytnik tablic rejestracyjnych",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w", pady=(0, 12))

        toolbar = ttk.Frame(container)
        toolbar.pack(fill="x")

        self.select_button = ttk.Button(
            toolbar,
            text="Wybierz zdjęcie",
            command=self.choose_image,
        )
        self.select_button.pack(side="left")

        self.read_button = ttk.Button(
            toolbar,
            text="Odczytaj tablicę",
            command=self.start_reading,
            state="disabled",
        )
        self.read_button.pack(side="left", padx=(10, 0))

        self.filename_label = ttk.Label(
            container,
            text="Nie wybrano zdjęcia.",
        )
        self.filename_label.pack(anchor="w", pady=(10, 12))

        self.preview_canvas = tk.Canvas(
            container,
            background="#eeeeee",
            highlightthickness=0,
        )
        self.preview_canvas.pack(fill="both", expand=True)
        self.preview_canvas.bind("<Configure>", self.update_preview)

        result_frame = ttk.LabelFrame(
            container,
            text="Wynik odczytu",
            padding=10,
        )
        result_frame.pack(fill="x", pady=(12, 0))

        self.number_label = ttk.Label(
            result_frame,
            text="—",
            font=("Segoe UI", 26, "bold"),
        )
        self.number_label.pack(anchor="w")

        self.plate_label = ttk.Label(result_frame)
        self.plate_label.pack(anchor="w", pady=(4, 8))

        self.confidence_label = ttk.Label(
            result_frame,
            text="Pewność modelu: —",
        )
        self.confidence_label.pack(anchor="w")

        self.confidence_bar = ttk.Progressbar(
            result_frame,
            mode="determinate",
            maximum=100,
        )
        self.confidence_bar.pack(fill="x", pady=(6, 6))

        self.details_label = ttk.Label(
            result_frame,
            text="",
            wraplength=700,
        )
        self.details_label.pack(anchor="w")

        self.status_label = ttk.Label(
            container,
            text="Wybierz zdjęcie samochodu w formacie JPG lub PNG.",
            wraplength=720,
        )
        self.status_label.pack(anchor="w", pady=(12, 0))

    def clear_result(self):
        self.number_label.config(text="—")
        self.plate_label.config(image="")
        self.plate_photo = None
        self.confidence_label.config(text="Pewność modelu: —")
        self.confidence_bar["value"] = 0
        self.details_label.config(text="")

    def choose_image(self):
        if self.busy:
            return

        file_path = filedialog.askopenfilename(
            parent=self.root,
            title="Wybierz zdjęcie samochodu",
            filetypes=[
                ("Zdjęcia JPG i PNG", ("*.jpg", "*.jpeg", "*.png")),
                ("Zdjęcia JPG", ("*.jpg", "*.jpeg")),
                ("Zdjęcia PNG", "*.png"),
            ],
        )

        if not file_path:
            return

        try:
            with Image.open(file_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, ValueError) as error:
            messagebox.showerror(
                "Błąd odczytu zdjęcia",
                f"Nie udało się otworzyć zdjęcia.\n\n{error}",
                parent=self.root,
            )
            return

        self.image_path = Path(file_path)
        self.original_image = image
        self.clear_result()

        self.filename_label.config(
            text=f"Wybrany plik: {self.image_path.name}"
        )

        width, height = image.size
        self.status_label.config(
            text=f"Wczytano zdjęcie: {width} × {height} pikseli."
        )

        self.read_button.config(state="normal")
        self.update_preview()

    def update_preview(self, event=None):
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()

        self.preview_canvas.delete("all")

        if self.original_image is None:
            self.preview_canvas.create_text(
                canvas_width // 2,
                canvas_height // 2,
                text="Tutaj pojawi się wybrane zdjęcie",
                fill="#666666",
                font=("Segoe UI", 13),
            )
            return

        preview = self.original_image.copy()
        preview.thumbnail(
            (
                max(1, canvas_width - 20),
                max(1, canvas_height - 20),
            ),
            Image.Resampling.LANCZOS,
        )

        self.preview_photo = ImageTk.PhotoImage(
            preview,
            master=self.root,
        )

        self.preview_canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=self.preview_photo,
            anchor="center",
        )

    def start_reading(self):
        if self.busy or self.original_image is None:
            return

        # Przekazujemy do analizy osobną kopię zdjęcia.
        image = self.original_image.copy()

        self.busy = True
        self.select_button.config(state="disabled")
        self.read_button.config(state="disabled")
        self.clear_result()

        if self.reader is None:
            status = "Wczytywanie modelu i analiza zdjęcia…"
        else:
            status = "Trwa analiza zdjęcia…"

        self.status_label.config(text=status)

        worker = Thread(
            target=self.read_in_background,
            args=(image,),
            daemon=True,
        )
        worker.start()

        self.root.after(100, self.check_result)

    def read_in_background(self, image):
        # Ta metoda nie zmienia żadnych elementów okna.
        try:
            if self.reader is None:
                # TensorFlow importujemy w tle przy pierwszym odczycie.
                from reader import PlateReader

                project_dir = Path(__file__).resolve().parent
                model_dir = (
                    project_dir / "models" / "20261021_220903"
                )

                self.reader = PlateReader(model_dir)

            result = self.reader.read(image)
            self.result_queue.put(("success", result))

        except Exception as error:
            traceback.print_exc()
            error_message = str(error) or type(error).__name__
            self.result_queue.put(("error", error_message))

    def check_result(self):
        # Ta metoda działa w głównym wątku i aktualizuje GUI.
        try:
            kind, result = self.result_queue.get_nowait()
        except Empty:
            self.root.after(100, self.check_result)
            return

        self.busy = False
        self.select_button.config(state="normal")
        self.read_button.config(state="normal")

        if kind == "error":
            self.status_label.config(text="Odczyt zakończony błędem.")
            messagebox.showerror(
                "Błąd analizy",
                result,
                parent=self.root,
            )
            return

        if result is None:
            self.number_label.config(text="Nie wykryto tablicy")
            self.status_label.config(
                text="Nie znaleziono obszaru umożliwiającego odczyt."
            )
            return

        self.show_result(result)

    def show_result(self, result):
        number = result["number"]
        scores = result["scores"]
        mean_score = sum(scores) / len(scores)

        self.number_label.config(text=number)
        self.confidence_label.config(
            text=f"Pewność modelu (średnia): {mean_score:.1%}"
        )
        self.confidence_bar["value"] = mean_score * 100

        details = "  |  ".join(
            f"{character}: {score:.1%}"
            for character, score in zip(number, scores)
        )
        self.details_label.config(text=details)

        plate_preview = ImageOps.contain(
            result["plate"],
            (360, 80),
            Image.Resampling.LANCZOS,
        )
        self.plate_photo = ImageTk.PhotoImage(
            plate_preview,
            master=self.root,
        )
        self.plate_label.config(image=self.plate_photo)

        self.status_label.config(
            text=f"Analiza zakończona. Rozpoznano {len(number)} znaków."
        )


def main():
    root = tk.Tk()
    app = PlateReaderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()