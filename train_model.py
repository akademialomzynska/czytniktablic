import json
from pathlib import Path
from datetime import datetime

import tensorflow as tf


def load_datasets(dataset_dir, batch_size=64):
    info_path = dataset_dir / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))

    class_names = info["class_names"]
    image_size = tuple(info["image_size"])
    datasets = {}

    for split_name in ("train", "validation"):
        split_dir = dataset_dir / split_name

        image_count = len(list(split_dir.glob("*/*.png")))
        expected_count = info["totals"][split_name]

        if image_count != expected_count:
            raise ValueError(
                f"{split_name}: znaleziono {image_count} obrazków, "
                f"a oczekiwano {expected_count}."
            )

        dataset = tf.keras.utils.image_dataset_from_directory(
            str(split_dir),
            labels="inferred",
            label_mode="int",
            class_names=class_names,
            color_mode="grayscale",
            image_size=image_size,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            seed=42,
            interpolation="nearest",
        )

        datasets[split_name] = dataset
        print(f"{split_name}: poprawnie znaleziono {image_count} plików.")

    return datasets["train"], datasets["validation"], class_names

def build_model(number_of_classes):
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(32, 32, 1)),

            # Przeliczamy wartości pikseli z 0–255 na 0–1.
            tf.keras.layers.Rescaling(1.0 / 255),

            tf.keras.layers.Conv2D(
                32, (3, 3), padding="same", activation="relu"
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(
                64, (3, 3), padding="same", activation="relu"
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(
                number_of_classes,
                activation="softmax",
            ),
        ],
        name="character_cnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    project_dir = Path(__file__).resolve().parent
    dataset_dir = project_dir / "data" / "synthetic_v1"
    info_path = dataset_dir / "dataset_info.json"

    if not info_path.is_file():
        print(f"Brak opisu gotowego zbioru w: {dataset_dir}")
        return

    tf.keras.utils.set_random_seed(42)

    train_dataset, validation_dataset, class_names = load_datasets(
        dataset_dir
    )

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    # Każdy trening zapisujemy w osobnym folderze.
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_dir / "models" / run_name
    run_dir.mkdir(parents=True)

    model_path = run_dir / "characters.keras"

    # Kolejność klas musi być zachowana przy późniejszym odczycie.
    (run_dir / "classes.json").write_text(
        json.dumps(class_names, indent=2),
        encoding="utf-8",
    )

    # Zachowujemy opis danych użytych do tego treningu.
    (run_dir / "dataset_info.json").write_text(
        info_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    model = build_model(len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(run_dir / "history.csv")
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=20,
        callbacks=callbacks,
    )

    # Oceniamy model zapisany przy najniższym val_loss.
    best_model = tf.keras.models.load_model(str(model_path))

    metrics = best_model.evaluate(
        validation_dataset,
        return_dict=True,
        verbose=0,
    )

    results = {
        "epochs_completed": len(history.epoch),
        "validation_accuracy": float(metrics["accuracy"]),
        "validation_loss": float(metrics["loss"]),
        "tensorflow_version": tf.__version__,
    }

    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nTrening zakończony.")
    print(f"Liczba wykonanych epok: {len(history.epoch)}")
    print(f"Dokładność walidacyjna: {metrics['accuracy']:.2%}")
    print(f"Błąd walidacyjny: {metrics['loss']:.4f}")
    print(f"Model: {model_path}")
    print(f"Historia treningu: {run_dir / 'history.csv'}")


if __name__ == "__main__":
    main()