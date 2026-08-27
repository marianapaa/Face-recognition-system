import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2 as cv

from .config import LOG_PATH, SNAPSHOT_DIR


def safe_timestamp() -> str:
    """
    Creates a filename-safe timestamp.
    It avoids characters like ':' that may be invalid in filenames.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_attempt(
    frame,
    name: str,
    user_id: Optional[int],
    confidence: Optional[float],
    granted: bool,
) -> Path:
    """
    Saves one access attempt:
    1. screenshot from the webcam;
    2. row in the CSV log file.
    """
    timestamp = safe_timestamp()
    status = "granted" if granted else "denied"

    safe_name = name.replace(" ", "_")
    snapshot_name = f"{timestamp}_{status}_{safe_name}.jpg"
    snapshot_path = SNAPSHOT_DIR / snapshot_name

    cv.imwrite(str(snapshot_path), frame)

    new_file = not LOG_PATH.exists()

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if new_file:
            writer.writerow(
                [
                    "timestamp",
                    "name",
                    "user_id",
                    "confidence",
                    "status",
                    "snapshot",
                ]
            )

        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                name,
                user_id if user_id is not None else "",
                round(confidence, 3) if confidence is not None else "",
                status,
                str(snapshot_path),
            ]
        )

    return snapshot_path