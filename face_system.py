import cv2
import json
import math
import time

from pathlib import Path

import numpy as np

from modules.webcam import Webcam
from modules.face_landmarker import FaceLandmarker
from modules.face_embedding_extractor import FaceEmbeddingExtractor
from modules.face_embedding_database import FaceEmbeddingDatabase
from modules.face_embedding_recognizer import FaceEmbeddingRecognizer
from modules.profile_card import ProfileCard
from modules.face_age_clientscan import FaceAgeClientScan
from modules.tracker import CentroidTracker


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_NAME = "OMI Face AI"

REGISTRATION_TARGET_SAMPLES = 100

# Minimum time between accepted samples.
REGISTRATION_MIN_INTERVAL = 0.20

# Cosine similarity above this value means the new frame is
# too similar to the previous accepted frame.
REGISTRATION_MAX_DUPLICATE_SIMILARITY = 0.995

# Quality thresholds.
MIN_FACE_WIDTH = 90
MIN_FACE_HEIGHT = 90
MIN_FACE_AREA_RATIO = 0.025

MIN_BLUR_SCORE = 35.0
MIN_BRIGHTNESS = 40.0
MAX_BRIGHTNESS = 220.0
MIN_CONTRAST = 18.0


# ============================================================
# REGISTRATION POSES
# ============================================================

# Ten stages x ten samples = 100 samples.
#
# The yaw/pitch values below are based on normalized landmark
# geometry rather than fixed camera-specific degree estimates.
#
# LEFT / RIGHT sign is calibrated during the first strong
# movement so mirrored webcams do not break the system.

REGISTRATION_STAGES = [
    {
        "name": "FRONT",
        "instruction": "LOOK STRAIGHT AT THE CAMERA",
        "type": "front",
        "target_count": 10,
    },
    {
        "name": "SLIGHT LEFT",
        "instruction": "TURN YOUR HEAD SLIGHTLY LEFT",
        "type": "yaw",
        "target_count": 10,
        "strength": 0.90,
    },
    {
        "name": "MEDIUM LEFT",
        "instruction": "TURN YOUR HEAD FURTHER LEFT",
        "type": "yaw",
        "target_count": 10,
        "strength": 1.35,
    },
    {
        "name": "STRONG LEFT",
        "instruction": "TURN YOUR HEAD FAR LEFT",
        "type": "yaw",
        "target_count": 10,
        "strength": 1.75,
    },
    {
        "name": "SLIGHT RIGHT",
        "instruction": "TURN YOUR HEAD SLIGHTLY RIGHT",
        "type": "yaw",
        "target_count": 10,
        "strength": 0.90,
    },
    {
        "name": "MEDIUM RIGHT",
        "instruction": "TURN YOUR HEAD FURTHER RIGHT",
        "type": "yaw",
        "target_count": 10,
        "strength": 1.35,
    },
    {
        "name": "STRONG RIGHT",
        "instruction": "TURN YOUR HEAD FAR RIGHT",
        "type": "yaw",
        "target_count": 10,
        "strength": 1.75,
    },
    {
        "name": "CHIN UP",
        "instruction": "LOOK UP - RAISE YOUR CHIN",
        "type": "pitch",
        "target_count": 10,
    },
    {
        "name": "CHIN DOWN",
        "instruction": "LOOK DOWN - LOWER YOUR CHIN",
        "type": "pitch",
        "target_count": 10,
    },
    {
        "name": "HEAD TILT",
        "instruction": "TILT YOUR HEAD NATURALLY LEFT AND RIGHT",
        "type": "roll",
        "target_count": 10,
    },
]


# ============================================================
# BASIC FACE CROP
# ============================================================

def crop_face(frame, face, padding=40):
    bbox = face.get("bbox")

    if bbox is None:
        return None

    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
    except (TypeError, ValueError):
        return None

    height, width = frame.shape[:2]

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop.copy()


# ============================================================
# LANDMARK HELPERS
# ============================================================

def landmark_xyz(landmark):
    """
    Supports common MediaPipe landmark representations:

    tuple/list:
        (x, y)
        (x, y, z)

    dict:
        {"x": ..., "y": ..., "z": ...}

    object:
        landmark.x
        landmark.y
        landmark.z
    """

    if landmark is None:
        return None

    # tuple/list/np.ndarray
    if isinstance(landmark, (tuple, list, np.ndarray)):
        if len(landmark) < 2:
            return None

        try:
            x = float(landmark[0])
            y = float(landmark[1])

            if len(landmark) >= 3:
                z = float(landmark[2])
            else:
                z = 0.0

            return x, y, z
        except (TypeError, ValueError):
            return None

    # dictionary
    if isinstance(landmark, dict):
        try:
            x = float(landmark.get("x"))
            y = float(landmark.get("y"))
            z = float(landmark.get("z", 0.0))

            return x, y, z
        except (TypeError, ValueError):
            return None

    # MediaPipe landmark object
    try:
        x = float(landmark.x)
        y = float(landmark.y)
        z = float(getattr(landmark, "z", 0.0))

        return x, y, z
    except (AttributeError, TypeError, ValueError):
        return None


def get_face_landmarks(face):
    landmarks = face.get("landmarks")

    if landmarks is None:
        return None

    if not isinstance(landmarks, (list, tuple, np.ndarray)):
        return None

    if len(landmarks) < 478:
        return None

    output = []

    for landmark in landmarks[:478]:
        point = landmark_xyz(landmark)

        if point is None:
            return None

        output.append(point)

    return np.asarray(output, dtype=np.float32)


# ============================================================
# FACE GEOMETRY
# ============================================================

def calculate_face_geometry(face):
    """
    Produces normalized pose measurements.

    yaw:
        horizontal nose displacement relative to eye distance

    pitch:
        vertical nose/chin geometry

    roll:
        eye-line rotation in degrees
    """

    landmarks = get_face_landmarks(face)

    if landmarks is None:
        return None

    # MediaPipe landmarks.
    LEFT_EYE = 33
    RIGHT_EYE = 263
    NOSE = 1
    CHIN = 152

    left_eye = landmarks[LEFT_EYE]
    right_eye = landmarks[RIGHT_EYE]
    nose = landmarks[NOSE]
    chin = landmarks[CHIN]

    eye_mid = (
        left_eye[:2] + right_eye[:2]
    ) / 2.0

    eye_distance = np.linalg.norm(
        right_eye[:2] - left_eye[:2]
    )

    if eye_distance < 1e-6:
        return None

    # --------------------------------------------
    # YAW
    # --------------------------------------------

    yaw = (
        nose[0] - eye_mid[0]
    ) / eye_distance

    # --------------------------------------------
    # PITCH
    # --------------------------------------------

    face_vertical = (
        chin[1] - eye_mid[1]
    )

    if abs(face_vertical) < 1e-6:
        return None

    nose_vertical_ratio = (
        nose[1] - eye_mid[1]
    ) / face_vertical

    # Around frontal orientation this is roughly
    # centered. Convert the displacement into a
    # convenient signed value.
    pitch = (
        0.5 - nose_vertical_ratio
    )

    # --------------------------------------------
    # ROLL
    # --------------------------------------------

    dx = (
        right_eye[0] - left_eye[0]
    )

    dy = (
        right_eye[1] - left_eye[1]
    )

    roll = math.degrees(
        math.atan2(dy, dx)
    )

    return {
        "yaw": float(yaw),
        "pitch": float(pitch),
        "roll": float(roll),
    }


# ============================================================
# IMAGE QUALITY
# ============================================================

def calculate_image_quality(face_image):
    if face_image is None or face_image.size == 0:
        return None

    gray = cv2.cvtColor(
        face_image,
        cv2.COLOR_BGR2GRAY
    )

    blur_score = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    brightness = float(
        np.mean(gray)
    )

    contrast = float(
        np.std(gray)
    )

    return {
        "blur": blur_score,
        "brightness": brightness,
        "contrast": contrast,
    }


def quality_is_good(face_image):
    metrics = calculate_image_quality(
        face_image
    )

    if metrics is None:
        return False, None

    if metrics["blur"] < MIN_BLUR_SCORE:
        return False, metrics

    if metrics["brightness"] < MIN_BRIGHTNESS:
        return False, metrics

    if metrics["brightness"] > MAX_BRIGHTNESS:
        return False, metrics

    if metrics["contrast"] < MIN_CONTRAST:
        return False, metrics

    return True, metrics


# ============================================================
# FACE SIZE CHECK
# ============================================================

def face_size_is_good(frame, face):
    bbox = face.get("bbox")

    if bbox is None:
        return False

    try:
        x1, y1, x2, y2 = [
            int(v) for v in bbox
        ]
    except (TypeError, ValueError):
        return False

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    if width < MIN_FACE_WIDTH:
        return False

    if height < MIN_FACE_HEIGHT:
        return False

    frame_height, frame_width = frame.shape[:2]

    face_area = width * height
    frame_area = frame_width * frame_height

    if frame_area <= 0:
        return False

    area_ratio = (
        face_area / frame_area
    )

    if area_ratio < MIN_FACE_AREA_RATIO:
        return False

    return True


# ============================================================
# DUPLICATE CHECK
# ============================================================

def embedding_similarity(a, b):
    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
    )

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-8 or norm_b < 1e-8:
        return -1.0

    return float(
        np.dot(a, b)
        / (norm_a * norm_b)
    )


def sample_is_different(
    embedding,
    previous_embedding
):
    if previous_embedding is None:
        return True

    similarity = embedding_similarity(
        embedding,
        previous_embedding
    )

    return (
        similarity
        < REGISTRATION_MAX_DUPLICATE_SIMILARITY
    )


# ============================================================
# LANDMARK SERIALIZATION
# ============================================================

def extract_landmark_arrays(
    frame,
    face
):
    landmarks = get_face_landmarks(
        face
    )

    if landmarks is None:
        return None, None

    normalized = landmarks.copy()

    height, width = frame.shape[:2]

    pixel = np.zeros(
        (len(landmarks), 2),
        dtype=np.float32
    )

    pixel[:, 0] = (
        landmarks[:, 0] * width
    )

    pixel[:, 1] = (
        landmarks[:, 1] * height
    )

    return normalized, pixel


# ============================================================
# SAVE DEMOGRAPHICS
# ============================================================

def save_person_demographics(
    face_database,
    person_name,
    age,
    gender
):
    if age is None and gender is None:
        return False

    try:
        db_path = getattr(
            face_database,
            "db_path",
            None
        )

        if db_path is None:
            db_path = getattr(
                face_database,
                "database_path",
                None
            )

        if db_path is None:
            db_path = getattr(
                face_database,
                "db_file",
                None
            )

        if db_path is None:
            db_path = (
                face_database.data_directory
                / "omi.db"
            )

        connection = __import__(
            "sqlite3"
        ).connect(
            str(db_path)
        )

        try:
            cursor = connection.cursor()

            cursor.execute(
                """
                SELECT attributes_json
                FROM people
                WHERE name = ?
                """,
                (person_name,)
            )

            row = cursor.fetchone()

            attributes = {}

            if row and row[0]:
                try:
                    attributes = json.loads(
                        row[0]
                    )

                    if not isinstance(
                        attributes,
                        dict
                    ):
                        attributes = {}

                except Exception:
                    attributes = {}

            if gender is not None:
                attributes["gender"] = gender

            if age is not None:
                attributes["age_estimate"] = round(
                    float(age),
                    1
                )

            if age is not None:
                cursor.execute(
                    """
                    UPDATE people
                    SET age_estimate = ?
                    WHERE name = ?
                    """,
                    (
                        float(age),
                        person_name
                    )
                )

            cursor.execute(
                """
                UPDATE people
                SET attributes_json = ?
                WHERE name = ?
                """,
                (
                    json.dumps(
                        attributes
                    ),
                    person_name
                )
            )

            connection.commit()

            return True

        finally:
            connection.close()

    except Exception as error:
        print(
            f"[Omi] Demographic update failed: "
            f"{error}"
        )

        return False


# ============================================================
# PROFILE CARD
# ============================================================

def create_profile_card(
    face_database,
    profile_card,
    person_name,
    face_image,
    sample_count,
    age=None,
    gender=None
):
    person = face_database.get_person(
        person_name
    )

    if person is None:
        return False

    safe_name = (
        face_database._safe_name(
            person_name
        )
    )

    person_directory = (
        face_database.people_directory
        / (
            f"person_"
            f"{person['id']:04d}_"
            f"{safe_name}"
        )
    )

    profile_path = (
        person_directory
        / "profile.jpg"
    )

    extra_info = None

    if gender is not None:
        extra_info = {
            "Gender": gender
        }

    profile_card.create(
        person_name=person_name,
        face_image=face_image,
        sample_count=sample_count,
        output_path=profile_path,
        age=(
            None
            if age is None
            else round(
                float(age),
                1
            )
        ),
        age_confidence=None,
        extra_info=extra_info
    )

    relative_profile_path = (
        profile_path.relative_to(
            face_database.data_directory
        )
    )

    face_database.set_profile_image(
        person_name,
        relative_profile_path
    )

    return True


# ============================================================
# DRAW MANUAL FACE INFORMATION
# ============================================================

def draw_face_information(
    frame,
    face,
    include_identity=True
):
    bbox = face.get("bbox")

    if bbox is None:
        return frame

    try:
        x1, y1, x2, y2 = [
            int(v) for v in bbox
        ]
    except (TypeError, ValueError):
        return frame

    identity = face.get(
        "identity",
        "Unknown"
    )

    age = face.get("age")
    gender = face.get("gender")

    parts = []

    if (
        include_identity
        and identity
        and identity != "Unknown"
    ):
        parts.append(identity)

    if age is not None:
        parts.append(
            f"Age {float(age):.1f}"
        )

    if gender is not None:
        parts.append(
            str(gender).capitalize()
        )

    if not parts:
        return frame

    text = " | ".join(parts)

    text_y = y2 + 24

    if text_y >= frame.shape[0] - 5:
        text_y = max(
            25,
            y1 - 10
        )

    cv2.putText(
        frame,
        text,
        (
            max(5, x1),
            text_y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    return frame


# ============================================================
# REGISTRATION STAGE VALIDATION
# ============================================================

class RegistrationController:

    def __init__(self):
        self.active = False

        self.name = ""

        self.stage_index = 0

        self.stage_samples = []

        self.all_samples = []

        self.last_capture_time = 0.0

        self.previous_embedding = None

        self.front_yaw = None
        self.front_pitch = None

        self.left_sign = None
        self.up_sign = None

        self.status = "READY"

        self.last_rejection_reason = ""

        self.total_rejected = 0

        self.registration_started_at = None

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    def start(self, name):
        self.active = True

        self.name = name

        self.stage_index = 0

        self.stage_samples = []

        self.all_samples = []

        self.last_capture_time = 0.0

        self.previous_embedding = None

        self.front_yaw = None
        self.front_pitch = None

        self.left_sign = None
        self.up_sign = None

        self.status = "STARTING"

        self.last_rejection_reason = ""

        self.total_rejected = 0

        self.registration_started_at = (
            time.time()
        )

    # --------------------------------------------------------
    # CANCEL
    # --------------------------------------------------------

    def cancel(self):
        self.active = False

        self.name = ""

        self.stage_index = 0

        self.stage_samples = []

        self.all_samples = []

        self.status = "CANCELLED"

    # --------------------------------------------------------
    # CURRENT STAGE
    # --------------------------------------------------------

    def current_stage(self):
        if (
            self.stage_index
            >= len(REGISTRATION_STAGES)
        ):
            return None

        return REGISTRATION_STAGES[
            self.stage_index
        ]

    # --------------------------------------------------------
    # STAGE PROGRESS
    # --------------------------------------------------------

    def stage_progress(self):
        stage = self.current_stage()

        if stage is None:
            return 0, 0

        return (
            len(self.stage_samples),
            stage["target_count"]
        )

    # --------------------------------------------------------
    # TOTAL PROGRESS
    # --------------------------------------------------------

    def total_progress(self):
        return (
            len(self.all_samples),
            REGISTRATION_TARGET_SAMPLES
        )

    # --------------------------------------------------------
    # FRONT VALIDATION
    # --------------------------------------------------------

    def accept_front_pose(
        self,
        geometry
    ):
        yaw = abs(
            geometry["yaw"]
        )

        pitch = abs(
            geometry["pitch"]
        )

        roll = abs(
            geometry["roll"]
        )

        return (
            yaw < 0.09
            and pitch < 0.10
            and roll < 8.0
        )

    # --------------------------------------------------------
    # YAW VALIDATION
    # --------------------------------------------------------

    def accept_yaw_pose(
        self,
        geometry,
        desired,
        strength
    ):
        yaw = geometry["yaw"]

        if desired == "left":
            sign = self.left_sign

        else:
            if self.left_sign is None:
                return False

            sign = -self.left_sign

        if sign is None:
            return False

        signed_yaw = (
            yaw * sign
        )

        # --------------------------------------------
        # Calibration-friendly thresholds.
        # --------------------------------------------

        if strength <= 0.95:
            minimum = 0.085
            maximum = 0.24

        elif strength <= 1.40:
            minimum = 0.16
            maximum = 0.34

        else:
            minimum = 0.25
            maximum = 0.52

        return (
            signed_yaw >= minimum
            and signed_yaw <= maximum
        )

    # --------------------------------------------------------
    # PITCH VALIDATION
    # --------------------------------------------------------

    def accept_pitch_pose(
        self,
        geometry,
        desired
    ):
        pitch = geometry["pitch"]

        if self.up_sign is None:
            return False

        if desired == "up":
            signed_pitch = (
                pitch * self.up_sign
            )
        else:
            signed_pitch = (
                pitch * -self.up_sign
            )

        return abs(signed_pitch) >= 0.055

    # --------------------------------------------------------
    # ROLL VALIDATION
    # --------------------------------------------------------

    def accept_roll_pose(
        self,
        geometry
    ):
        roll = abs(
            geometry["roll"]
        )

        return (
            roll >= 8.0
            and roll <= 28.0
        )

    # --------------------------------------------------------
    # CALIBRATE LEFT/RIGHT
    # --------------------------------------------------------

    def calibrate_left_direction(
        self,
        geometry
    ):
        yaw = geometry["yaw"]

        if abs(yaw) < 0.10:
            return False

        self.left_sign = (
            1.0
            if yaw > 0
            else -1.0
        )

        return True

    # --------------------------------------------------------
    # CALIBRATE UP/DOWN
    # --------------------------------------------------------

    def calibrate_up_direction(
        self,
        geometry
    ):
        pitch = geometry["pitch"]

        if abs(pitch) < 0.06:
            return False

        self.up_sign = (
            1.0
            if pitch > 0
            else -1.0
        )

        return True

    # --------------------------------------------------------
    # POSE ACCEPTANCE
    # --------------------------------------------------------

    def pose_is_good(
        self,
        geometry
    ):
        stage = self.current_stage()

        if stage is None:
            return False

        stage_type = stage["type"]

        if stage_type == "front":
            return self.accept_front_pose(
                geometry
            )

        if stage_type == "yaw":

            stage_name = stage["name"]

            if "LEFT" in stage_name:

                desired = "left"

            else:

                desired = "right"

            # ----------------------------------------
            # Calibrate left direction during the
            # first substantial left movement.
            # ----------------------------------------

            if (
                desired == "left"
                and self.left_sign is None
            ):
                calibrated = (
                    self.calibrate_left_direction(
                        geometry
                    )
                )

                if not calibrated:
                    return False

            return self.accept_yaw_pose(
                geometry,
                desired,
                stage["strength"]
            )

        if stage_type == "pitch":

            stage_name = stage["name"]

            if "UP" in stage_name:

                desired = "up"

                if self.up_sign is None:

                    calibrated = (
                        self.calibrate_up_direction(
                            geometry
                        )
                    )

                    if not calibrated:
                        return False

            else:

                desired = "down"

                if self.up_sign is None:
                    return False

            return self.accept_pitch_pose(
                geometry,
                desired
            )

        if stage_type == "roll":

            return self.accept_roll_pose(
                geometry
            )

        return False

    # --------------------------------------------------------
    # ACCEPT SAMPLE
    # --------------------------------------------------------

    def accept_sample(
        self,
        frame,
        face,
        embedding,
        quality,
        geometry
    ):
        now = time.perf_counter()

        if (
            now - self.last_capture_time
            < REGISTRATION_MIN_INTERVAL
        ):
            return False

        if not sample_is_different(
            embedding,
            self.previous_embedding
        ):
            self.last_rejection_reason = (
                "TOO SIMILAR"
            )

            self.total_rejected += 1

            return False

        normalized_landmarks, pixel_landmarks = (
            extract_landmark_arrays(
                frame,
                face
            )
        )

        if (
            normalized_landmarks is None
            or pixel_landmarks is None
        ):
            self.last_rejection_reason = (
                "INVALID LANDMARKS"
            )

            self.total_rejected += 1

            return False

        sample_number = (
            len(self.all_samples) + 1
        )

        stage = self.current_stage()

        sample_data = {
            "sample_id": sample_number,
            "stage": stage["name"],
            "pose_type": stage["type"],
            "image": crop_face(
                frame,
                face
            ),
            "embedding": np.asarray(
                embedding,
                dtype=np.float32
            ).copy(),
            "landmarks_normalized": (
                normalized_landmarks.copy()
            ),
            "landmarks_pixel": (
                pixel_landmarks.copy()
            ),
            "geometry": dict(
                geometry
            ),
            "quality": dict(
                quality
            ),
        }

        if sample_data["image"] is None:
            return False

        self.all_samples.append(
            sample_data
        )

        self.stage_samples.append(
            sample_data
        )

        self.previous_embedding = (
            np.asarray(
                embedding,
                dtype=np.float32
            ).copy()
        )

        self.last_capture_time = now

        self.last_rejection_reason = ""

        # Update front calibration.
        if stage["type"] == "front":

            if self.front_yaw is None:

                self.front_yaw = (
                    geometry["yaw"]
                )

            if self.front_pitch is None:

                self.front_pitch = (
                    geometry["pitch"]
                )

        # Stage completed.
        if (
            len(self.stage_samples)
            >= stage["target_count"]
        ):

            self.stage_index += 1

            self.stage_samples = []

            self.previous_embedding = None

        return True

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    def complete(self):
        return (
            len(self.all_samples)
            >= REGISTRATION_TARGET_SAMPLES
        )


# ============================================================
# SAVE COMPLETE REGISTRATION DATA
# ============================================================

def save_registration_dataset(
    face_database,
    registration
):
    if not registration.all_samples:
        return False, None

    person = face_database.get_person(
        registration.name
    )

    if person is None:
        return False, None

    safe_name = face_database._safe_name(
        registration.name
    )

    person_directory = (
        face_database.people_directory
        / (
            f"person_"
            f"{person['id']:04d}_"
            f"{safe_name}"
        )
    )

    person_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset_directory = (
        person_directory
        / "registration_data"
    )

    dataset_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Build fixed-size arrays.
    # --------------------------------------------------------

    embeddings = np.stack(
        [
            sample["embedding"]
            for sample
            in registration.all_samples
        ],
        axis=0
    ).astype(
        np.float32
    )

    landmarks_normalized = np.stack(
        [
            sample[
                "landmarks_normalized"
            ]
            for sample
            in registration.all_samples
        ],
        axis=0
    ).astype(
        np.float32
    )

    landmarks_pixel = np.stack(
        [
            sample[
                "landmarks_pixel"
            ]
            for sample
            in registration.all_samples
        ],
        axis=0
    ).astype(
        np.float32
    )

    yaw = np.asarray(
        [
            sample["geometry"]["yaw"]
            for sample
            in registration.all_samples
        ],
        dtype=np.float32
    )

    pitch = np.asarray(
        [
            sample["geometry"]["pitch"]
            for sample
            in registration.all_samples
        ],
        dtype=np.float32
    )

    roll = np.asarray(
        [
            sample["geometry"]["roll"]
            for sample
            in registration.all_samples
        ],
        dtype=np.float32
    )

    blur = np.asarray(
        [
            sample["quality"]["blur"]
            for sample
            in registration.all_samples
        ],
        dtype=np.float32
    )

    brightness = np.asarray(
        [
            sample["quality"]["brightness"]
            for sample
            in registration.all_samples
        ],
        dtype=np.float32
    )

    contrast = np.asarray(
        [
            sample["quality"]["contrast"]
            for sample
            in registration.all_samples
        ],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Save numerical feature store.
    # --------------------------------------------------------

    feature_store_path = (
        dataset_directory
        / "feature_store.npz"
    )

    np.savez_compressed(
        str(feature_store_path),
        embeddings=embeddings,
        landmarks_normalized=(
            landmarks_normalized
        ),
        landmarks_pixel=landmarks_pixel,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        blur=blur,
        brightness=brightness,
        contrast=contrast,
    )

    # --------------------------------------------------------
    # Save manifest.
    # --------------------------------------------------------

    manifest_samples = []

    for sample in registration.all_samples:

        manifest_samples.append(
            {
                "sample_id": sample[
                    "sample_id"
                ],
                "stage": sample[
                    "stage"
                ],
                "pose_type": sample[
                    "pose_type"
                ],
                "yaw": sample[
                    "geometry"
                ]["yaw"],
                "pitch": sample[
                    "geometry"
                ]["pitch"],
                "roll": sample[
                    "geometry"
                ]["roll"],
                "blur": sample[
                    "quality"
                ]["blur"],
                "brightness": sample[
                    "quality"
                ]["brightness"],
                "contrast": sample[
                    "quality"
                ]["contrast"],
            }
        )

    manifest = {
        "omi_registration_version": 2,
        "person_name": registration.name,
        "sample_count": len(
            registration.all_samples
        ),
        "landmark_count": 478,
        "embedding_dimensions": (
            int(embeddings.shape[1])
        ),
        "created_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "pose_stages": [
            stage["name"]
            for stage
            in REGISTRATION_STAGES
        ],
        "samples": manifest_samples,
    }

    manifest_path = (
        dataset_directory
        / "registration_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2
        )

    return True, dataset_directory


# ============================================================
# REGISTRATION UI
# ============================================================

def draw_registration_ui(
    frame,
    registration
):
    if not registration.active:
        return frame

    height, width = frame.shape[:2]

    stage = registration.current_stage()

    total_current, total_target = (
        registration.total_progress()
    )

    if stage is None:
        return frame

    stage_current, stage_target = (
        registration.stage_progress()
    )

    # --------------------------------------------------------
    # Main stage box.
    # --------------------------------------------------------

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (20, 20),
        (width - 20, 175),
        (15, 15, 15),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.78,
        frame,
        0.22,
        0
    )

    # --------------------------------------------------------
    # Title.
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "OMI FACE REGISTRATION",
        (40, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Stage.
    # --------------------------------------------------------

    stage_number = (
        registration.stage_index + 1
    )

    stage_text = (
        f"STAGE {stage_number}/"
        f"{len(REGISTRATION_STAGES)}  "
        f"{stage['name']}"
    )

    cv2.putText(
        frame,
        stage_text,
        (40, 83),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Instruction.
    # --------------------------------------------------------

    cv2.putText(
        frame,
        stage["instruction"],
        (40, 113),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Progress.
    # --------------------------------------------------------

    progress_text = (
        f"SAMPLES: "
        f"{total_current}/"
        f"{total_target}"
        f"    STAGE: "
        f"{stage_current}/"
        f"{stage_target}"
    )

    cv2.putText(
        frame,
        progress_text,
        (40, 143),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Quality rejection message.
    # --------------------------------------------------------

    if registration.last_rejection_reason:

        cv2.putText(
            frame,
            f"REJECTED: "
            f"{registration.last_rejection_reason}",
            (40, height - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Escape hint.
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "ESC = CANCEL REGISTRATION",
        (40, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (190, 190, 190),
        1,
        cv2.LINE_AA
    )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Core systems.
    # --------------------------------------------------------

    camera = Webcam()

    face_landmarker = FaceLandmarker()

    face_embedding_extractor = (
        FaceEmbeddingExtractor()
    )

    face_database = (
        FaceEmbeddingDatabase()
    )

    face_recognizer = (
        FaceEmbeddingRecognizer(
            face_database
        )
    )

    profile_card = ProfileCard()

    # --------------------------------------------------------
    # FaceAge.
    # --------------------------------------------------------

    face_age = FaceAgeClientScan(
        model_path="models/faceage.onnx",
        update_interval=1.0,
        history_size=5
    )

    # --------------------------------------------------------
    # Tracking.
    # --------------------------------------------------------

    face_tracker = CentroidTracker()

    # --------------------------------------------------------
    # Window.
    # --------------------------------------------------------

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        1280,
        720
    )

    fullscreen = False

    # --------------------------------------------------------
    # Recognition cache.
    # --------------------------------------------------------

    frame_number = 0

    recognition_interval = 4

    last_recognition_frame = {}

    recognition_cache = {}

    # --------------------------------------------------------
    # Demographic profile cache.
    # --------------------------------------------------------

    profile_demographic_cache = {}

    # --------------------------------------------------------
    # Registration controller.
    # --------------------------------------------------------

    registration = (
        RegistrationController()
    )

    # --------------------------------------------------------
    # Facial mesh toggle.
    # --------------------------------------------------------

    face_lines_enabled = True

    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while True:

            frame = camera.get_frame()

            if frame is None:
                break

            frame_number += 1

            # =================================================
            # FACE LANDMARKS
            # =================================================

            frame, faces = (
                face_landmarker.detect(
                    frame
                )
            )

            # =================================================
            # FACE TRACKING
            # =================================================

            faces = face_tracker.update(
                faces
            )

            # =================================================
            # FACE AGE + GENDER
            # =================================================

            single_face_mode = (
                len(faces) == 1
            )

            active_age_keys = []

            for face_index, face in enumerate(
                faces
            ):

                face_id = face.get(
                    "id"
                )

                if single_face_mode:

                    age_key = (
                        "single_face"
                    )

                elif face_id is not None:

                    age_key = (
                        f"face_{face_id}"
                    )

                else:

                    age_key = (
                        f"temporary_{face_index}"
                    )

                active_age_keys.append(
                    age_key
                )

                if not registration.active:

                    age_face_image = crop_face(
                        frame,
                        face,
                        padding=0
                    )

                    if age_face_image is not None:

                        face_age.submit(
                            age_key,
                            frame,
                            face
                        )

                    age_result = face_age.get(
                        age_key
                    )

                    if age_result is not None:

                        face["age"] = (
                            age_result["age"]
                        )

                        face["gender"] = (
                            age_result["gender"]
                        )

                        face["age_stability"] = (
                            age_result["stability"]
                        )

                    else:

                        face["age"] = None

                        face["gender"] = None

                        face["age_stability"] = (
                            None
                        )

                else:

                    # Still allow FaceAge to run during
                    # registration so demographic data
                    # is ready when registration finishes.

                    age_face_image = crop_face(
                        frame,
                        face,
                        padding=0
                    )

                    if age_face_image is not None:

                        face_age.submit(
                            age_key,
                            frame,
                            face
                        )

                    age_result = face_age.get(
                        age_key
                    )

                    if age_result is not None:

                        face["age"] = (
                            age_result["age"]
                        )

                        face["gender"] = (
                            age_result["gender"]
                        )

                        face["age_stability"] = (
                            age_result["stability"]
                        )

            face_age.cleanup(
                active_age_keys
            )

            # =================================================
            # REGISTRATION
            # =================================================

            if registration.active:

                # Registration requires exactly one face.

                if len(faces) == 1:

                    current_face = faces[0]

                    # -----------------------------------------
                    # Face geometry.
                    # -----------------------------------------

                    geometry = (
                        calculate_face_geometry(
                            current_face
                        )
                    )

                    if geometry is None:

                        registration.last_rejection_reason = (
                            "INVALID POSE"
                        )

                    else:

                        # -------------------------------------
                        # Face size.
                        # -------------------------------------

                        if not face_size_is_good(
                            frame,
                            current_face
                        ):

                            registration.last_rejection_reason = (
                                "FACE TOO SMALL"
                            )

                        else:

                            # ---------------------------------
                            # Pose.
                            # ---------------------------------

                            pose_ok = (
                                registration.pose_is_good(
                                    geometry
                                )
                            )

                            if not pose_ok:

                                registration.last_rejection_reason = (
                                    "MOVE TO REQUESTED POSE"
                                )

                            else:

                                # -----------------------------
                                # Face crop.
                                # -----------------------------

                                face_image = crop_face(
                                    frame,
                                    current_face
                                )

                                quality_ok, quality = (
                                    quality_is_good(
                                        face_image
                                    )
                                )

                                if not quality_ok:

                                    if (
                                        quality is None
                                    ):

                                        registration.last_rejection_reason = (
                                            "IMAGE QUALITY"
                                        )

                                    elif quality[
                                        "blur"
                                    ] < MIN_BLUR_SCORE:

                                        registration.last_rejection_reason = (
                                            "TOO BLURRY"
                                        )

                                    elif quality[
                                        "brightness"
                                    ] < MIN_BRIGHTNESS:

                                        registration.last_rejection_reason = (
                                            "TOO DARK"
                                        )

                                    elif quality[
                                        "brightness"
                                    ] > MAX_BRIGHTNESS:

                                        registration.last_rejection_reason = (
                                            "TOO BRIGHT"
                                        )

                                    else:

                                        registration.last_rejection_reason = (
                                            "LOW CONTRAST"
                                        )

                                else:

                                    # -------------------------
                                    # ArcFace embedding.
                                    # -------------------------

                                    embedding = (
                                        face_embedding_extractor.extract(
                                            frame,
                                            current_face
                                        )
                                    )

                                    if embedding is None:

                                        registration.last_rejection_reason = (
                                            "EMBEDDING FAILED"
                                        )

                                    else:

                                        # ---------------------
                                        # Try accepting sample.
                                        # ---------------------

                                        accepted = (
                                            registration.accept_sample(
                                                frame,
                                                current_face,
                                                embedding,
                                                quality,
                                                geometry
                                            )
                                        )

                                        if accepted:

                                            progress, target = (
                                                registration.stage_progress()
                                            )

                                            total, total_target = (
                                                registration.total_progress()
                                            )

                                            if total % 5 == 0:

                                                print(
                                                    f"[Omi] Captured "
                                                    f"{total}/"
                                                    f"{total_target} "
                                                    f"| "
                                                    f"{registration.current_stage()['name'] if registration.current_stage() else 'DONE'}"
                                                )

                        # -------------------------------------
                        # Registration complete.
                        # -------------------------------------

                        if registration.complete():

                            print(
                                "\n[Omi] "
                                "100 samples captured."
                            )

                            print(
                                "[Omi] Saving detailed "
                                "face dataset..."
                            )

                            try:

                                saved = (
                                    face_database.add_person(
                                        registration.name,
                                        [
                                            {
                                                "embedding": (
                                                    sample[
                                                        "embedding"
                                                    ].tolist()
                                                ),
                                                "image": (
                                                    sample[
                                                        "image"
                                                    ]
                                                ),
                                            }
                                            for sample
                                            in registration.all_samples
                                        ]
                                    )
                                )

                            except Exception as error:

                                print(
                                    "[Omi] Database "
                                    "registration failed:"
                                )

                                print(error)

                                saved = False

                            if saved:

                                # ---------------------------------
                                # Save landmark/pose dataset.
                                # ---------------------------------

                                dataset_saved, dataset_path = (
                                    save_registration_dataset(
                                        face_database,
                                        registration
                                    )
                                )

                                if dataset_saved:

                                    print(
                                        "[Omi] Detailed "
                                        "registration dataset saved:"
                                    )

                                    print(
                                        f"[Omi] {dataset_path}"
                                    )

                                # ---------------------------------
                                # Demographics.
                                # ---------------------------------

                                registration_age = None
                                registration_gender = None

                                age_result = (
                                    face_age.get(
                                        "single_face"
                                    )
                                )

                                if age_result is not None:

                                    registration_age = (
                                        age_result.get(
                                            "age"
                                        )
                                    )

                                    registration_gender = (
                                        age_result.get(
                                            "gender"
                                        )
                                    )

                                if (
                                    registration_age
                                    is not None
                                ):

                                    save_person_demographics(
                                        face_database,
                                        registration.name,
                                        registration_age,
                                        registration_gender
                                    )

                                # ---------------------------------
                                # Profile card.
                                # ---------------------------------

                                first_sample = (
                                    registration.all_samples[
                                        0
                                    ]["image"]
                                )

                                try:

                                    create_profile_card(
                                        face_database=(
                                            face_database
                                        ),
                                        profile_card=(
                                            profile_card
                                        ),
                                        person_name=(
                                            registration.name
                                        ),
                                        face_image=(
                                            first_sample
                                        ),
                                        sample_count=100,
                                        age=(
                                            registration_age
                                        ),
                                        gender=(
                                            registration_gender
                                        )
                                    )

                                except Exception as error:

                                    print(
                                        "[Omi] Profile card "
                                        "creation failed:"
                                    )

                                    print(error)

                                print(
                                    "[Omi] Registration complete."
                                )

                                print(
                                    f"[Omi] Person: "
                                    f"{registration.name}"
                                )

                                print(
                                    "[Omi] "
                                    "100 face images saved."
                                )

                                print(
                                    "[Omi] "
                                    "100 ArcFace embeddings saved."
                                )

                                print(
                                    "[Omi] "
                                    "478 landmarks saved "
                                    "for every sample."
                                )

                            else:

                                print(
                                    "[Omi] Registration "
                                    "could not be saved."
                                )

                            # Reset registration state.

                            registration.active = False

                            registration.name = ""

                            registration.stage_index = 0

                            registration.stage_samples = []

                            registration.all_samples = []

                            registration.previous_embedding = None

                            registration.last_rejection_reason = ""

                            # Reset recognition.

                            try:
                                face_recognizer.reset()
                            except Exception:
                                pass

                            last_recognition_frame.clear()

                            recognition_cache.clear()

                else:

                    registration.last_rejection_reason = (
                        "SHOW EXACTLY ONE FACE"
                    )

            # =================================================
            # REGULAR RECOGNITION
            # =================================================

            else:

                active_face_keys = []

                single_face_mode = (
                    len(faces) == 1
                )

                for face_index, face in enumerate(
                    faces
                ):

                    face_id = face.get(
                        "id"
                    )

                    if single_face_mode:

                        face_key = (
                            "single_face"
                        )

                    elif face_id is not None:

                        face_key = (
                            f"face_{face_id}"
                        )

                    else:

                        face_key = (
                            f"temporary_{face_index}"
                        )

                    active_face_keys.append(
                        face_key
                    )

                    last_frame = (
                        last_recognition_frame.get(
                            face_key,
                            -recognition_interval
                        )
                    )

                    should_recognize = (
                        frame_number
                        - last_frame
                        >= recognition_interval
                    )

                    if should_recognize:

                        embedding = (
                            face_embedding_extractor.extract(
                                frame,
                                face
                            )
                        )

                        last_recognition_frame[
                            face_key
                        ] = frame_number

                        if embedding is None:

                            cached = (
                                recognition_cache.get(
                                    face_key
                                )
                            )

                            if cached is not None:

                                face["identity"] = (
                                    cached["name"]
                                )

                                face[
                                    "recognition_similarity"
                                ] = (
                                    cached[
                                        "similarity"
                                    ]
                                )

                                face[
                                    "recognition_margin"
                                ] = (
                                    cached[
                                        "margin"
                                    ]
                                )

                            else:

                                face["identity"] = (
                                    "Unknown"
                                )

                                face[
                                    "recognition_similarity"
                                ] = None

                                face[
                                    "recognition_margin"
                                ] = None

                        else:

                            result = (
                                face_recognizer.recognize(
                                    embedding,
                                    face_key
                                )
                            )

                            recognition_cache[
                                face_key
                            ] = {
                                "name": result[
                                    "name"
                                ],
                                "similarity": result[
                                    "similarity"
                                ],
                                "margin": result[
                                    "margin"
                                ],
                            }

                            face["identity"] = (
                                result["name"]
                            )

                            face[
                                "recognition_similarity"
                            ] = (
                                result["similarity"]
                            )

                            face[
                                "recognition_margin"
                            ] = (
                                result["margin"]
                            )

                    else:

                        cached = (
                            recognition_cache.get(
                                face_key
                            )
                        )

                        if cached is not None:

                            face["identity"] = (
                                cached["name"]
                            )

                            face[
                                "recognition_similarity"
                            ] = (
                                cached[
                                    "similarity"
                                ]
                            )

                            face[
                                "recognition_margin"
                            ] = (
                                cached[
                                    "margin"
                                ]
                            )

                        else:

                            face["identity"] = (
                                "Unknown"
                            )

                            face[
                                "recognition_similarity"
                            ] = None

                            face[
                                "recognition_margin"
                            ] = None

                # ---------------------------------------------
                # Recognition cleanup.
                # ---------------------------------------------

                try:
                    face_recognizer.cleanup(
                        active_face_keys
                    )
                except Exception:
                    pass

                stale_recognition_keys = [
                    key
                    for key
                    in list(
                        last_recognition_frame
                    )
                    if key
                    not in active_face_keys
                ]

                for key in stale_recognition_keys:
                    del last_recognition_frame[
                        key
                    ]

                stale_cache_keys = [
                    key
                    for key
                    in list(
                        recognition_cache
                    )
                    if key
                    not in active_face_keys
                ]

                for key in stale_cache_keys:
                    del recognition_cache[
                        key
                    ]

                # ---------------------------------------------
                # Update saved demographic information.
                # ---------------------------------------------

                for face in faces:

                    identity = face.get(
                        "identity",
                        "Unknown"
                    )

                    age = face.get(
                        "age"
                    )

                    gender = face.get(
                        "gender"
                    )

                    if (
                        identity
                        in (
                            None,
                            "Unknown",
                            "Registering..."
                        )
                        or age is None
                    ):
                        continue

                    rounded_age = round(
                        float(age),
                        1
                    )

                    demographic_key = (
                        identity,
                        rounded_age,
                        gender
                    )

                    if (
                        profile_demographic_cache.get(
                            identity
                        )
                        == demographic_key
                    ):
                        continue

                    try:

                        save_person_demographics(
                            face_database,
                            identity,
                            age,
                            gender
                        )

                        person = (
                            face_database.get_person(
                                identity
                            )
                        )

                        if person is not None:

                            safe_name = (
                                face_database._safe_name(
                                    identity
                                )
                            )

                            person_directory = (
                                face_database.people_directory
                                / (
                                    f"person_"
                                    f"{person['id']:04d}_"
                                    f"{safe_name}"
                                )
                            )

                            sample_path = (
                                person_directory
                                / "samples"
                                / "sample_001.jpg"
                            )

                            sample_image = None

                            if sample_path.exists():

                                sample_image = (
                                    cv2.imread(
                                        str(
                                            sample_path
                                        )
                                    )
                                )

                            if sample_image is not None:

                                create_profile_card(
                                    face_database,
                                    profile_card,
                                    identity,
                                    sample_image,
                                    100,
                                    age,
                                    gender
                                )

                        profile_demographic_cache[
                            identity
                        ] = demographic_key

                    except Exception as error:

                        print(
                            "[Omi] Profile update "
                            "failed for "
                            f"{identity}: {error}"
                        )

            # =================================================
            # DRAW FACIAL MESH / FACE INFORMATION
            # =================================================

            if face_lines_enabled:

                # FaceLandmarker already draws the identity.
                # Therefore we DO NOT draw the identity again.

                frame = (
                    face_landmarker.draw(
                        frame,
                        faces
                    )
                )

                for face in faces:

                    frame = draw_face_information(
                        frame,
                        face,
                        include_identity=False
                    )

            else:

                # No mesh.
                # Draw all information exactly once.

                for face in faces:

                    frame = draw_face_information(
                        frame,
                        face,
                        include_identity=True
                    )

            # =================================================
            # REGISTRATION UI
            # =================================================

            if registration.active:

                frame = draw_registration_ui(
                    frame,
                    registration
                )

            else:

                # Facial lines status.
                status = (
                    "FACIAL LINES: ON"
                    if face_lines_enabled
                    else "FACIAL LINES: OFF"
                )

                cv2.putText(
                    frame,
                    status,
                    (
                        20,
                        frame.shape[0] - 45
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    frame,
                    "R = REGISTER    "
                    "L = FACIAL LINES    "
                    "F = FULLSCREEN    "
                    "Q = QUIT",
                    (
                        20,
                        frame.shape[0] - 18
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (190, 190, 190),
                    1,
                    cv2.LINE_AA
                )

            # =================================================
            # DISPLAY
            # =================================================

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # =================================================
            # REGISTER
            # =================================================

            if (
                key == ord("r")
                and not registration.active
            ):

                if len(faces) == 0:

                    print(
                        "\n[Omi] "
                        "No face detected."
                    )

                    continue

                if len(faces) > 1:

                    print(
                        "\n[Omi] "
                        "Registration requires "
                        "exactly one face."
                    )

                    continue

                print(
                    "\n[Omi] Starting detailed "
                    "100-sample registration."
                )

                name = input(
                    "[Omi] Enter person's name: "
                ).strip()

                if not name:

                    print(
                        "[Omi] Registration cancelled."
                    )

                    continue

                registration.start(
                    name
                )

                print(
                    "\n[Omi] Registration plan:"
                )

                for index, stage in enumerate(
                    REGISTRATION_STAGES,
                    start=1
                ):

                    print(
                        f"  {index}. "
                        f"{stage['name']} "
                        f"({stage['target_count']} samples)"
                    )

                print(
                    "\n[Omi] Start by looking "
                    "straight at the camera."
                )

            # =================================================
            # CANCEL REGISTRATION
            # =================================================

            elif (
                key == 27
                and registration.active
            ):

                print(
                    "\n[Omi] Registration cancelled."
                )

                registration.cancel()

            # =================================================
            # FACIAL LINES
            # =================================================

            elif (
                key == ord("l")
            ):

                face_lines_enabled = (
                    not face_lines_enabled
                )

                print(
                    "[Omi] Facial lines: "
                    +
                    (
                        "ON"
                        if face_lines_enabled
                        else "OFF"
                    )
                )

            # =================================================
            # FULLSCREEN
            # =================================================

            elif key == ord("f"):

                fullscreen = (
                    not fullscreen
                )

                if fullscreen:

                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN
                    )

                else:

                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL
                    )

            # =================================================
            # QUIT
            # =================================================

            elif key == ord("q"):

                break

    finally:

        print(
            "[Omi] Shutting down..."
        )

        try:
            face_age.close()
        except Exception:
            pass

        try:
            face_embedding_extractor.close()
        except Exception:
            pass

        try:
            face_landmarker.close()
        except Exception:
            pass

        try:
            face_database.close()
        except Exception:
            pass

        camera.release()

        cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()