import argparse

from .capture import capture_training_images
from .train import train_recognizer
from .access import run_access_control
from .security import hash_password


def main() -> None:
    """
    Command-line interface for the project.
    """
    parser = argparse.ArgumentParser(
        description="Face Recognition Access Control Project"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    capture_parser = subparsers.add_parser(
        "capture",
        help="Capture training images from webcam",
    )

    capture_parser.add_argument(
        "--name",
        required=True,
        help="Person name",
    )

    capture_parser.add_argument(
        "--user-id",
        required=True,
        type=int,
        help="Numeric user ID",
    )

    capture_parser.add_argument(
        "--samples",
        default=30,
        type=int,
        help="Number of face samples",
    )

    subparsers.add_parser(
        "train",
        help="Train LBPH face recognizer",
    )

    subparsers.add_parser(
        "access",
        help="Run password and live video access control",
    )

    subparsers.add_parser(
        "hash-password",
        help="Generate SHA-256 password hash",
    )

    args = parser.parse_args()

    if args.command == "capture":
        capture_training_images(
            name=args.name,
            user_id=args.user_id,
            samples=args.samples,
        )

    elif args.command == "train":
        train_recognizer()

    elif args.command == "access":
        run_access_control()

    elif args.command == "hash-password":
        password = input("Enter password to hash: ")
        print(hash_password(password))