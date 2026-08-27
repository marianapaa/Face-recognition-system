from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

KNOWN_FACES_DIR = DATA_DIR / "known_faces"
TEST_FACES_DIR = DATA_DIR / "test_faces"
TRAINER_DIR = DATA_DIR / "trainer"
SNAPSHOT_DIR = DATA_DIR / "access_snapshots"

MODEL_PATH = TRAINER_DIR / "lbph_trainer.yml"
LABELS_PATH = TRAINER_DIR / "labels.json"
LOG_PATH = DATA_DIR / "lab_access_log.csv"

CASCADE_PATH = (
    Path(os.getenv("OPENCV_HAAR_CASCADE", ""))
    if os.getenv("OPENCV_HAAR_CASCADE")
    else None
)

ACCESS_PASSWORD_HASH = os.getenv("ACCESS_PASSWORD_HASH", "")
AUTHORIZED_NAME = os.getenv("AUTHORIZED_NAME", "Captain Demming")
AUTHORIZED_ID = int(os.getenv("AUTHORIZED_ID", "1"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "80"))

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "960"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "540"))

for folder in [DATA_DIR, KNOWN_FACES_DIR, TEST_FACES_DIR, TRAINER_DIR, SNAPSHOT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)