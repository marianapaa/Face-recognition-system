import json
import time
from typing import Dict, Optional

import cv2 as cv

from .audio import Speaker
from .config import (
    ACCESS_PASSWORD_HASH,
    AUTHORIZED_ID,
    AUTHORIZED_NAME,
    CAMERA_INDEX,
    CASCADE_PATH,
    CONFIDENCE_THRESHOLD,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    LABELS_PATH,
    MODEL_PATH,
)
from .logger import save_attempt
from .security import ask_password, verify_password
from .vision import (
    get_face_detector,
    detect_faces,
    crop_face,
    resize_face,
)


VERIFICATION_SHOTS = 5


def load_labels() -> Dict[int, str]:
    if not LABELS_PATH.exists():
        return {}

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        raw_labels = json.load(file)

    return {int(key): value for key, value in raw_labels.items()}


def system_startup_sequence(speaker: Speaker) -> None:
    print("=" * 60)
    print("LABORATORY SECURITY SYSTEM")
    print("=" * 60)

    speaker.say("Laboratory security system activated.")
    time.sleep(0.3)


def password_instruction_sequence(speaker: Speaker) -> None:
    speaker.say("Please enter the access password.")
    time.sleep(0.3)


def camera_instruction_sequence(speaker: Speaker) -> None:
    speaker.say("Password accepted.")
    time.sleep(0.3)
    speaker.say("Look at the camera and place your face inside the frame.")
    time.sleep(0.3)


def access_granted_sequence(speaker: Speaker) -> None:
    speaker.say("Identity confirmed.")
    time.sleep(0.3)
    speaker.say(f"Access granted. Welcome, {AUTHORIZED_NAME}.")
    time.sleep(0.3)


def wrong_password_sequence(speaker: Speaker) -> None:
    speaker.say("Access denied. Incorrect password.")
    time.sleep(0.3)


def access_denied_sequence(speaker: Speaker) -> None:
    speaker.say("Identity could not be verified.")
    time.sleep(0.3)
    speaker.say("Access denied. Security snapshot saved.")
    time.sleep(0.3)


def no_face_detected_sequence(speaker: Speaker) -> None:
    speaker.say("No face detected. Access denied.")
    time.sleep(0.3)


def draw_access_interface(
    frame,
    status_text: str,
    instruction_text: str,
    shot_count: int,
    confidence_text: str = "",
    granted: Optional[bool] = None,
) -> None:
    height, width = frame.shape[:2]

    if granted is True:
        main_color = (0, 220, 0)
    elif granted is False:
        main_color = (0, 0, 230)
    else:
        main_color = (0, 180, 255)

    cv.rectangle(frame, (0, 0), (width, 130), (18, 18, 18), cv.FILLED)

    cv.putText(
        frame,
        "BIOMETRIC VERIFICATION MODE",
        (25, 38),
        cv.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
    )

    cv.putText(
        frame,
        status_text,
        (25, 78),
        cv.FONT_HERSHEY_SIMPLEX,
        0.78,
        main_color,
        2,
    )

    cv.putText(
        frame,
        f"Verification shots: {shot_count}/{VERIFICATION_SHOTS}",
        (width - 230, 78),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (230, 230, 230),
        2,
    )

    box_width = 260
    box_height = 320

    x1 = width // 2 - box_width // 2
    y1 = height // 2 - box_height // 2 + 50
    x2 = x1 + box_width
    y2 = y1 + box_height

    cv.rectangle(frame, (x1, y1), (x2, y2), main_color, 3)

    corner = 45
    thickness = 5

    cv.line(frame, (x1, y1), (x1 + corner, y1), main_color, thickness)
    cv.line(frame, (x1, y1), (x1, y1 + corner), main_color, thickness)

    cv.line(frame, (x2, y1), (x2 - corner, y1), main_color, thickness)
    cv.line(frame, (x2, y1), (x2, y1 + corner), main_color, thickness)

    cv.line(frame, (x1, y2), (x1 + corner, y2), main_color, thickness)
    cv.line(frame, (x1, y2), (x1, y2 - corner), main_color, thickness)

    cv.line(frame, (x2, y2), (x2 - corner, y2), main_color, thickness)
    cv.line(frame, (x2, y2), (x2, y2 - corner), main_color, thickness)

    cv.rectangle(frame, (0, height - 120), (width, height), (18, 18, 18), cv.FILLED)

    cv.putText(
        frame,
        instruction_text,
        (25, height - 65),
        cv.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
    )

    if confidence_text:
        cv.putText(
            frame,
            confidence_text,
            (25, height - 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.62,
            (210, 210, 210),
            2,
        )

    cv.putText(
        frame,
        "Press Q to exit",
        (width - 190, height - 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 180, 180),
        2,
    )


def draw_detected_face(
    frame,
    rect,
    label: str,
    granted: Optional[bool] = None,
) -> None:
    x, y, w, h = rect

    if granted is True:
        color = (0, 220, 0)
    elif granted is False:
        color = (0, 0, 230)
    else:
        color = (0, 180, 255)

    cv.rectangle(frame, (x, y), (x + w, y + h), color, 1)
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


def show_verification_flash(frame) -> None:
    flash = frame.copy()
    flash[:] = (255, 255, 255)

    blended = cv.addWeighted(frame, 0.4, flash, 0.6, 0)
    cv.imshow("Face Access Control", blended)
    cv.waitKey(70)


def run_access_control() -> None:
    speaker = Speaker()

    system_startup_sequence(speaker)

    if not MODEL_PATH.exists():
        raise RuntimeError("Trained model not found. Run: python main.py train")

    if not ACCESS_PASSWORD_HASH:
        raise RuntimeError("ACCESS_PASSWORD_HASH is missing. Add it to .env first.")

    password_instruction_sequence(speaker)
    password = ask_password()

    if not verify_password(password, ACCESS_PASSWORD_HASH):
        wrong_password_sequence(speaker)
        return

    camera_instruction_sequence(speaker)

    detector = get_face_detector(CASCADE_PATH)

    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))

    labels = load_labels()

    cap = cv.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    start_time = time.time()
    max_seconds = 18

    final_frame = None
    best_result = None

    verification_results = []
    shot_count = 0
    last_shot_time = 0
    shot_delay = 0.65

    face_detected_announced = False

    status_text = "POSITION FACE INSIDE THE FRAME"
    instruction_text = "Look at the camera. Keep your face inside the scanner frame."
    confidence_text = ""

    while time.time() - start_time < max_seconds and shot_count < VERIFICATION_SHOTS:
        success, frame = cap.read()

        if not success:
            continue

        final_frame = frame.copy()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = detect_faces(gray, detector)

        if len(faces) == 0:
            status_text = "WAITING FOR FACE"
            instruction_text = "Move closer and place your face inside the frame."
            confidence_text = ""

        for rect in faces:
            if not face_detected_announced:
                speaker.say("Face detected. Verification started.")
                face_detected_announced = True

            face = crop_face(gray, rect)
            face = resize_face(face)

            current_time = time.time()

            predicted_id = None
            confidence = None
            name = "Unknown"

            if current_time - last_shot_time >= shot_delay:
                predicted_id, confidence = recognizer.predict(face)
                name = labels.get(predicted_id, "Unknown")

                granted_single = (
                    predicted_id == AUTHORIZED_ID
                    and confidence <= CONFIDENCE_THRESHOLD
                )

                verification_results.append(granted_single)
                shot_count += 1
                last_shot_time = current_time

                status_text = "VERIFICATION SHOT CAPTURED"
                instruction_text = f"Verification image {shot_count} captured."
                confidence_text = f"Distance: {confidence:.2f}"

                draw_detected_face(
                    frame,
                    rect,
                    f"Shot {shot_count}/{VERIFICATION_SHOTS}",
                    None,
                )

                draw_access_interface(
                    frame,
                    status_text,
                    instruction_text,
                    shot_count,
                    confidence_text,
                    None,
                )

                show_verification_flash(frame)

                print(
                    f"[SHOT {shot_count}] "
                    f"Predicted ID: {predicted_id}, "
                    f"Name: {name}, "
                    f"Distance: {confidence:.2f}, "
                    f"Accepted: {granted_single}"
                )

                candidate = {
                    "frame": frame.copy(),
                    "name": name if granted_single else "Unknown",
                    "user_id": predicted_id,
                    "confidence": confidence,
                    "granted": granted_single,
                }

                if best_result is None or confidence < best_result["confidence"]:
                    best_result = candidate

            status_text = "SCANNING FACE"
            instruction_text = "Hold still while verification shots are captured."

            draw_detected_face(
                frame,
                rect,
                f"Capturing {shot_count}/{VERIFICATION_SHOTS}",
                None,
            )

        draw_access_interface(
            frame,
            status_text,
            instruction_text,
            shot_count,
            confidence_text,
            None,
        )

        cv.imshow("Face Access Control", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    successful_shots = verification_results.count(True)
    access_granted = successful_shots >= 3

    analysis_frame = final_frame.copy()

    draw_access_interface(
       analysis_frame,
       "ANALYZING BIOMETRIC DATA...",
       "Comparing captured images with authorized profile...",
       shot_count,
       granted=None,
    )

    cv.imshow("Face Access Control", analysis_frame)
    cv.waitKey(1000)
    speaker.say("Analyzing biometric data.")

    if access_granted and best_result is not None:
        result_frame = best_result["frame"]

        draw_access_interface(
            result_frame,
            "ACCESS GRANTED",
            f"Identity confirmed. Successful shots: {successful_shots}/{VERIFICATION_SHOTS}",
            shot_count,
            granted=True,
        )

        cv.imshow("Face Access Control", result_frame)
        cv.waitKey(900)

        snapshot = save_attempt(
            frame=result_frame,
            name=best_result["name"],
            user_id=best_result["user_id"],
            confidence=best_result["confidence"],
            granted=True,
        )

        access_granted_sequence(speaker)
        print(f"Snapshot saved to: {snapshot}")

    elif best_result is not None:
        result_frame = best_result["frame"]

        draw_access_interface(
            result_frame,
            "ACCESS DENIED",
            f"Identity not confirmed. Successful shots: {successful_shots}/{VERIFICATION_SHOTS}",
            shot_count,
            granted=False,
        )

        cv.imshow("Face Access Control", result_frame)
        cv.waitKey(900)

        snapshot = save_attempt(
            frame=result_frame,
            name=best_result["name"],
            user_id=best_result["user_id"],
            confidence=best_result["confidence"],
            granted=False,
        )

        access_denied_sequence(speaker)
        print(f"Snapshot saved to: {snapshot}")

    elif final_frame is not None:
        draw_access_interface(
            final_frame,
            "ACCESS DENIED",
            "No face was detected during verification.",
            shot_count,
            granted=False,
        )

        cv.imshow("Face Access Control", final_frame)
        cv.waitKey(900)

        snapshot = save_attempt(
            frame=final_frame,
            name="No face detected",
            user_id=None,
            confidence=None,
            granted=False,
        )

        no_face_detected_sequence(speaker)
        print(f"Snapshot saved to: {snapshot}")

    cap.release()
    cv.destroyAllWindows()