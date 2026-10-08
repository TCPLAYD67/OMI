import cv2
import time
import threading
import numpy as np
import onnxruntime as ort

from PIL import Image

from modules.face_landmarker import FaceLandmarker


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/faceage.onnx"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Minimum time between submitting age predictions
AGE_INTERVAL = 1.0


# ============================================================
# GLOBAL AGE STATE
# ============================================================

state_lock = threading.Lock()

latest_face_crop = None
new_face_available = False

latest_age = None
latest_gender = None

worker_running = False
worker_busy = False

worker_error = None

stop_worker = False


# ============================================================
# MODEL PREPROCESSING
# ============================================================

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def sigmoid(x):

    return 1.0 / (
        1.0 + np.exp(-x)
    )


def preprocess_face(face_bgr):

    # BGR -> RGB
    face_rgb = cv2.cvtColor(
        face_bgr,
        cv2.COLOR_BGR2RGB
    )

    # Official model preprocessing:
    # 224x224 bicubic
    image = Image.fromarray(
        face_rgb
    )

    image = image.resize(
        (224, 224),
        Image.Resampling.BICUBIC
    )

    image = np.asarray(
        image,
        dtype=np.float32
    )

    # 0..255 -> 0..1
    image /= 255.0

    # ImageNet normalization
    image = (
        image - MEAN
    ) / STD

    # HWC -> CHW
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    # Add batch dimension
    image = image[
        np.newaxis,
        ...
    ]

    return np.ascontiguousarray(
        image,
        dtype=np.float32
    )


# ============================================================
# FACE CROP
# ============================================================

def crop_face(
    frame,
    bbox,
    padding=0.10
):

    h, w = frame.shape[:2]

    x1, y1, x2, y2 = bbox

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return None

    # Official 10% proportional padding
    pad_x = int(
        width * padding
    )

    pad_y = int(
        height * padding
    )

    x1 -= pad_x
    y1 -= pad_y

    x2 += pad_x
    y2 += pad_y

    # Clamp
    x1 = max(
        0,
        x1
    )

    y1 = max(
        0,
        y1
    )

    x2 = min(
        w,
        x2
    )

    y2 = min(
        h,
        y2
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[
        y1:y2,
        x1:x2
    ].copy()


# ============================================================
# LANDMARK READER
# ============================================================

def get_xy(landmark):

    # Tuple/list/numpy
    if isinstance(
        landmark,
        (tuple, list, np.ndarray)
    ):

        if len(landmark) >= 2:

            return (
                float(landmark[0]),
                float(landmark[1])
            )

    # Dictionary
    if isinstance(
        landmark,
        dict
    ):

        if (
            "x" in landmark
            and
            "y" in landmark
        ):

            return (
                float(landmark["x"]),
                float(landmark["y"])
            )

    # MediaPipe landmark object
    if (
        hasattr(landmark, "x")
        and
        hasattr(landmark, "y")
    ):

        return (
            float(landmark.x),
            float(landmark.y)
        )

    return None


# ============================================================
# GET FACE BBOX
# ============================================================

def get_face_bbox(
    face,
    frame_shape
):

    h, w = frame_shape[:2]

    # --------------------------------------------------------
    # First preference:
    # if FaceLandmarker already supplies a bbox, use it.
    # --------------------------------------------------------

    possible_keys = [
        "bbox",
        "bounding_box",
        "box"
    ]

    for key in possible_keys:

        bbox = face.get(key)

        if bbox is None:
            continue

        try:

            if len(bbox) == 4:

                values = [
                    float(v)
                    for v in bbox
                ]

                x1, y1, x2, y2 = values

                # If normalized, convert to pixels.
                if (
                    max(abs(v) for v in values)
                    <= 2.0
                ):

                    x1 *= w
                    x2 *= w
                    y1 *= h
                    y2 *= h

                x1 = max(
                    0,
                    min(int(x1), w - 1)
                )

                y1 = max(
                    0,
                    min(int(y1), h - 1)
                )

                x2 = max(
                    0,
                    min(int(x2), w - 1)
                )

                y2 = max(
                    0,
                    min(int(y2), h - 1)
                )

                if (
                    x2 > x1
                    and
                    y2 > y1
                ):

                    return (
                        x1,
                        y1,
                        x2,
                        y2
                    )

        except Exception:
            pass


    # --------------------------------------------------------
    # Fallback:
    # build bbox from MediaPipe landmarks.
    # --------------------------------------------------------

    landmarks = face.get(
        "landmarks"
    )

    if not landmarks:
        return None

    xs = []
    ys = []

    for landmark in landmarks:

        point = get_xy(
            landmark
        )

        if point is None:
            continue

        x, y = point

        xs.append(x)
        ys.append(y)

    if not xs or not ys:
        return None

    min_x = min(xs)
    max_x = max(xs)

    min_y = min(ys)
    max_y = max(ys)

    # MediaPipe coordinates are normally normalized 0..1
    x1 = int(
        min_x * w
    )

    y1 = int(
        min_y * h
    )

    x2 = int(
        max_x * w
    )

    y2 = int(
        max_y * h
    )

    x1 = max(
        0,
        min(x1, w - 1)
    )

    y1 = max(
        0,
        min(y1, h - 1)
    )

    x2 = max(
        0,
        min(x2, w - 1)
    )

    y2 = max(
        0,
        min(y2, h - 1)
    )

    if (
        x2 <= x1
        or
        y2 <= y1
    ):
        return None

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# FACEAGE WORKER
# ============================================================

def age_worker():

    global worker_running
    global worker_busy
    global worker_error

    global latest_age
    global latest_gender

    global latest_face_crop
    global new_face_available

    worker_running = False

    print()
    print("=" * 60)
    print("FACEAGE WORKER STARTING")
    print("=" * 60)

    try:

        # ----------------------------------------------------
        # Create ONNX session INSIDE this worker thread.
        # ----------------------------------------------------

        available = (
            ort.get_available_providers()
        )

        print(
            "Available ORT providers:",
            available
        )

        if "CUDAExecutionProvider" in available:

            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider"
            ]

        else:

            providers = [
                "CPUExecutionProvider"
            ]

            print(
                "CUDAExecutionProvider "
                "not available."
            )


        print(
            "Loading:",
            MODEL_PATH
        )

        session = ort.InferenceSession(
            MODEL_PATH,
            providers=providers
        )

        print(
            "Active providers:",
            session.get_providers()
        )


        # ----------------------------------------------------
        # Model I/O
        # ----------------------------------------------------

        input_name = (
            session
            .get_inputs()[0]
            .name
        )

        print(
            "Input:",
            input_name
        )

        for output in session.get_outputs():

            print(
                "Output:",
                output.name,
                output.shape,
                output.type
            )

        print(
            "FACEAGE MODEL READY"
        )

        print("=" * 60)
        print()

        worker_running = True


    except Exception as e:

        worker_error = (
            f"Model initialization failed: {e}"
        )

        print()
        print(
            "FACEAGE INITIALIZATION ERROR:"
        )
        print(
            repr(e)
        )
        print()

        return


    # ========================================================
    # WORK LOOP
    # ========================================================

    while not stop_worker:

        face = None

        # ----------------------------------------------------
        # Get newest available face.
        # ----------------------------------------------------

        with state_lock:

            if new_face_available:

                face = latest_face_crop

                new_face_available = False

        # ----------------------------------------------------
        # Nothing to process.
        # ----------------------------------------------------

        if face is None:

            time.sleep(0.01)

            continue


        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        worker_busy = True

        try:

            start = (
                time.perf_counter()
            )

            tensor = preprocess_face(
                face
            )

            outputs = session.run(
                None,
                {
                    input_name: tensor
                }
            )

            # ------------------------------------------------
            # Official model:
            # output 0 = age_logits Bx100
            # output 1 = gender_logits Bx2
            # ------------------------------------------------

            age_logits = np.asarray(
                outputs[0]
            )

            gender_logits = np.asarray(
                outputs[1]
            )


            # Remove batch dimension
            age_logits = (
                age_logits[0]
            )

            gender_logits = (
                gender_logits[0]
            )


            # CORAL age decoding
            probabilities = sigmoid(
                age_logits
            )

            age = float(
                probabilities.sum()
            )


            # Gender
            gender_index = int(
                np.argmax(
                    gender_logits
                )
            )

            gender = (
                "male"
                if gender_index == 1
                else "female"
            )


            elapsed = (
                time.perf_counter()
                - start
            )


            with state_lock:

                latest_age = age
                latest_gender = gender
                worker_error = None


            print(
                f"[FaceAge] "
                f"AGE={age:.2f} "
                f"GENDER={gender} "
                f"TIME={elapsed:.2f}s"
            )


        except Exception as e:

            worker_error = str(e)

            print()
            print(
                "FACEAGE INFERENCE ERROR:"
            )
            print(
                repr(e)
            )
            print()


        finally:

            worker_busy = False


# ============================================================
# START AGE THREAD
# ============================================================

worker_thread = threading.Thread(
    target=age_worker,
    daemon=True
)

worker_thread.start()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    0
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not cap.isOpened():

    print(
        "ERROR: Camera could not be opened."
    )

    stop_worker = True

    raise SystemExit


# ============================================================
# FACE LANDMARKER
# ============================================================

face_landmarker = FaceLandmarker()


# ============================================================
# WINDOW
# ============================================================

WINDOW_NAME = (
    "FaceAge ClientScan"
)

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    960,
    720
)


# ============================================================
# TIMING
# ============================================================

last_submission = 0.0

face_seen = False


# ============================================================
# FPS
# ============================================================

fps_timer = time.perf_counter()

fps_count = 0

fps = 0.0


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            break


        # ----------------------------------------------------
        # Face Landmarker
        # ----------------------------------------------------

        processed_frame, faces = (
            face_landmarker.detect(
                frame
            )
        )


        face_seen = bool(
            faces
        )


        # ----------------------------------------------------
        # Find face
        # ----------------------------------------------------

        if faces:

            face = faces[0]

            bbox = get_face_bbox(
                face,
                frame.shape
            )

            if bbox is not None:

                x1, y1, x2, y2 = bbox

                # Draw bbox
                cv2.rectangle(
                    processed_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2
                )


                # --------------------------------------------
                # Crop
                # --------------------------------------------

                face_crop = crop_face(
                    frame,
                    bbox,
                    padding=0.10
                )


                # --------------------------------------------
                # Submit newest crop periodically
                # --------------------------------------------

                now = (
                    time.perf_counter()
                )

                if (
                    face_crop is not None
                    and
                    now - last_submission
                    >= AGE_INTERVAL
                ):

                    with state_lock:

                        latest_face_crop = (
                            face_crop
                        )

                        new_face_available = True

                    last_submission = now


        # ====================================================
        # READ AGE STATE
        # ====================================================

        with state_lock:

            displayed_age = latest_age
            displayed_gender = latest_gender
            displayed_error = worker_error


        # ====================================================
        # AGE DISPLAY
        # ====================================================

        if displayed_age is not None:

            cv2.putText(
                processed_frame,
                f"AGE: {displayed_age:.1f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            if displayed_gender is not None:

                cv2.putText(
                    processed_frame,
                    f"GENDER: {displayed_gender}",
                    (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        else:

            cv2.putText(
                processed_frame,
                "AGE: --",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )


        # ====================================================
        # STATUS
        # ====================================================

        if displayed_error:

            status = (
                "FaceAge ERROR"
            )

            status_color = (
                0,
                0,
                255
            )

        elif not face_seen:

            status = (
                "FaceAge: NO FACE"
            )

            status_color = (
                180,
                180,
                180
            )

        elif not worker_running:

            status = (
                "FaceAge: LOADING MODEL"
            )

            status_color = (
                0,
                255,
                255
            )

        elif worker_busy:

            status = (
                "FaceAge: INFERENCE"
            )

            status_color = (
                0,
                255,
                255
            )

        else:

            status = (
                "FaceAge: READY"
            )

            status_color = (
                0,
                255,
                0
            )


        cv2.putText(
            processed_frame,
            status,
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # FPS
        # ====================================================

        fps_count += 1

        now = (
            time.perf_counter()
        )

        elapsed = (
            now - fps_timer
        )

        if elapsed >= 1.0:

            fps = (
                fps_count / elapsed
            )

            fps_count = 0

            fps_timer = now


        cv2.putText(
            processed_frame,
            f"Preview FPS: {fps:.1f}",
            (30, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # ERROR ON SCREEN
        # ====================================================

        if displayed_error:

            # Keep it short enough for the window.
            error_text = str(
                displayed_error
            )[:80]

            cv2.putText(
                processed_frame,
                error_text,
                (30, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            processed_frame
        )


        # ====================================================
        # INPUT
        # ====================================================

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )

        if key == ord("q"):

            break


finally:

    stop_worker = True

    cap.release()

    cv2.destroyAllWindows()