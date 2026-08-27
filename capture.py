import time
import cv2 as cv

from .audio import Speaker
from .config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    KNOWN_FACES_DIR,
    CASCADE_PATH,
)
from .vision import get_face_detector, detect_faces, crop_face, resize_face


def draw_capture_interface(frame, count: int, samples: int, message: str) -> None:
    """
    Draws a professional interface on the webcam window.
    """
    height, width = frame.shape[:2]

    cv.rectangle(frame, (0, 0), (width, 90), (20, 20, 20), cv.FILLED)

    cv.putText(
        frame,
        "FACE IMAGE CAPTURE MODE",
        (25, 35),
        cv.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    cv.putText(
        frame,
        f"Captured images: {count}/{samples}",
        (25, 70),
        cv.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
    )

    cv.rectangle(frame, (0, height - 70), (width, height), (20, 20, 20), cv.FILLED)

    cv.putText(
        frame,
        message,
        (25, height - 28),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )


def show_flash_effect(frame) -> None:
    """
    Shows a short white flash effect when a photo is captured.
    """
    flash = frame.copy()
    flash[:] = (255, 255, 255)

    blended = cv.addWeighted(frame, 0.35, flash, 0.65, 0)
    cv.imshow("Capture Training Images", blended)
    cv.waitKey(80)


def capture_training_images(name: str, user_id: int, samples: int = 30) -> None:
    """
    Captures face images from the webcam and saves them for training.

    The camera window shows:
    - detected face rectangle;
    - number of saved photos;
    - instruction messages;
    - flash effect after each captured photo.
    """
    speaker = Speaker()
    detector = get_face_detector(CASCADE_PATH)

    cap = cv.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    safe_name = name.strip().lower().replace(" ", "_")
    count = 0

    speaker.say("Face capture mode started.")
    speaker.say("Please look directly at the camera.")
    speaker.say("The system will take several photos for training.")
    speaker.say("Slowly move your head left and right.")
    speaker.say("Try a neutral, happy, and serious expression.")
    speaker.say("Keep your face inside the green rectangle.")

    message = "Look at the camera. Press Q to stop early."
    last_capture_time = 0
    capture_delay = 0.45

    while count < samples:
        success, frame = cap.read()

        if not success:
            continue

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = detect_faces(gray, detector)

        if len(faces) == 0:
            message = "No face detected. Please move closer to the camera."

        for rect in faces:
            x, y, w, h = rect

            cv.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3,
            )

            cv.putText(
                frame,
                "Face detected",
                (x, y - 12),
                cv.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
            )

            current_time = time.time()

            if current_time - last_capture_time >= capture_delay:
                face = crop_face(gray, rect)
                face = resize_face(face)

                count += 1

                filename = KNOWN_FACES_DIR / f"{safe_name}.{user_id}.{count}.jpg"
                cv.imwrite(str(filename), face)

                message = f"Photo saved: {filename.name}"

                print(f"[SAVED] {filename}")

                show_flash_effect(frame)

                last_capture_time = current_time

                if count in [1, 10, 20]:
                    speaker.say(f"{count} photos captured.")

                if count >= samples:
                    break

        draw_capture_interface(frame, count, samples, message)

        cv.imshow("Capture Training Images", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    speaker.say("Image capture completed.")
    speaker.say(f"{count} training images were saved.")

    cap.release()
    cv.destroyAllWindows()

    print(f"Saved {count} image(s) to {KNOWN_FACES_DIR}")