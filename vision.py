from pathlib import Path
from typing import List, Tuple, Optional

import cv2 as cv
import numpy as np


def get_face_detector(cascade_path: Optional[Path] = None) -> cv.CascadeClassifier:
    """
    Loads the Haar Cascade classifier used for face detection.
    """
    if cascade_path and cascade_path.exists():
        path = str(cascade_path)
    else:
        path = cv.data.haarcascades + "haarcascade_frontalface_default.xml"

    detector = cv.CascadeClassifier(path)

    if detector.empty():
        raise RuntimeError("Could not load Haar cascade face detector.")

    return detector


def detect_faces(
    gray_frame: np.ndarray,
    detector: cv.CascadeClassifier
) -> List[Tuple[int, int, int, int]]:
    """
    Detects faces in a grayscale frame.
    Returns rectangles in the form: x, y, width, height.
    """
    faces = detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80),
    )

    return faces


def crop_face(
    gray_frame: np.ndarray,
    rect: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Cuts the detected face region from the frame.
    """
    x, y, w, h = rect
    return gray_frame[y:y + h, x:x + w]


def resize_face(
    face: np.ndarray,
    size: tuple[int, int] = (200, 200)
) -> np.ndarray:
    """
    Resizes all face images to the same size.
    This improves training stability.
    """
    return cv.resize(face, size, interpolation=cv.INTER_AREA)


def draw_face_box(
    frame: np.ndarray,
    rect: Tuple[int, int, int, int],
    label: str,
    granted: bool
) -> None:
    """
    Draws a colored rectangle and label around the detected face.
    Green means access granted, red means access denied.
    """
    x, y, w, h = rect

    if granted:
        color = (0, 180, 0)
    else:
        color = (0, 0, 220)

    cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv.rectangle(frame, (x, y - 32), (x + w, y), color, cv.FILLED)

    cv.putText(
        frame,
        label,
        (x + 8, y - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )