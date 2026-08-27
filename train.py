import json
from pathlib import Path

import cv2 as cv
import numpy as np

from .config import KNOWN_FACES_DIR, MODEL_PATH, LABELS_PATH


def parse_image_name(path: Path) -> tuple[str, int]:
    """
    Reads name and user ID from the image filename.

    Expected filename format:
    name.user_id.frame_number.jpg

    Example:
    captain_demming.1.5.jpg
    """
    parts = path.stem.split(".")

    if len(parts) < 3:
        raise ValueError(f"Invalid training filename: {path.name}")

    name = parts[0].replace("_", " ").title()
    user_id = int(parts[1])

    return name, user_id


def train_recognizer() -> None:
    """
    Trains the LBPH face recognizer using saved face images.
    """
    images = []
    labels = []
    id_to_name = {}

    image_paths = sorted(
        list(KNOWN_FACES_DIR.glob("*.jpg"))
        + list(KNOWN_FACES_DIR.glob("*.png"))
        + list(KNOWN_FACES_DIR.glob("*.jpeg"))
    )

    if not image_paths:
        raise RuntimeError(f"No training images found in {KNOWN_FACES_DIR}")

    for path in image_paths:
        name, user_id = parse_image_name(path)

        gray_image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)

        if gray_image is None:
            print(f"Skipping unreadable file: {path}")
            continue

        images.append(gray_image)
        labels.append(user_id)
        id_to_name[str(user_id)] = name

    if not images:
        raise RuntimeError("No valid training images were loaded.")

    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    recognizer.write(str(MODEL_PATH))

    with open(LABELS_PATH, "w", encoding="utf-8") as file:
        json.dump(id_to_name, file, indent=2, ensure_ascii=False)

    print("Training complete.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Labels saved to: {LABELS_PATH}")