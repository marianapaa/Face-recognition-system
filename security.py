import getpass
import hashlib
import hmac


def hash_password(password: str) -> str:
    """
    Converts a plain-text password into a SHA-256 hash.
    The real password is not stored in the project.
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(plain_password: str, stored_hash: str) -> bool:
    """
    Compares the entered password with the saved hash.
    hmac.compare_digest is used for safer comparison.
    """
    if not stored_hash:
        return False

    entered_hash = hash_password(plain_password)
    return hmac.compare_digest(entered_hash, stored_hash)


def ask_password() -> str:
    """
    Reads the password from the terminal without displaying it.
    """
    return getpass.getpass("Enter access password: ")