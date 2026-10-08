import threading
import time
from collections import deque, Counter
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


class FaceAgeClientScan:
    """
    Background FaceAge ClientScan estimator.

    The camera/main loop never waits for age inference.
    One background worker owns the ONNX Runtime session and processes
    the newest crop available for each tracked face.
    """

    def __init__(
        self,
        model_path="models/faceage.onnx",
        update_interval=1.0,
        history_size=5,
    ):

        self.model_path = Path(model_path)
        self.update_interval = float(update_interval)
        self.history_size = max(1, int(history_size))

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

        self._pending = {}
        self._last_submitted = {}
        self._last_result_time = {}

        self._results = {}
        self._age_history = {}
        self._gender_history = {}

        self._stop = False
        self._ready = False
        self._busy = False
        self._error = None

        self._session = None
        self._input_name = None
        self._age_output_index = 0
        self._gender_output_index = 1

        self._worker = threading.Thread(
            target=self._worker_loop,
            name="Omi-FaceAge",
            daemon=True,
        )

        self._worker.start()

    # ==========================================================
    # PUBLIC STATE
    # ==========================================================

    @property
    def ready(self):
        with self._lock:
            return self._ready

    @property
    def busy(self):
        with self._lock:
            return self._busy

    @property
    def error(self):
        with self._lock:
            return self._error

    # ==========================================================
    # FACE CROP
    # ==========================================================

    @staticmethod
    def _crop_face(frame, face, padding=0.10):

        bbox = face.get("bbox")

        if bbox is None or len(bbox) != 4:
            return None

        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except (TypeError, ValueError):
            return None

        height, width = frame.shape[:2]

        face_width = x2 - x1
        face_height = y2 - y1

        if face_width <= 0 or face_height <= 0:
            return None

        # ClientScan uses 10% proportional padding.
        pad_x = int(face_width * padding)
        pad_y = int(face_height * padding)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return crop.copy()

    # ==========================================================
    # SUBMIT
    # ==========================================================

    def submit(self, face_key, frame, face):
        """
        Submit the newest crop for a face.
        This method is deliberately non-blocking.
        """

        now = time.perf_counter()

        with self._condition:

            last = self._last_submitted.get(face_key, 0.0)

            if now - last < self.update_interval:
                return False

            crop = self._crop_face(
                frame,
                face,
                padding=0.10,
            )

            if crop is None:
                return False

            # Replace an older pending crop for this same face.
            self._pending[face_key] = crop
            self._last_submitted[face_key] = now

            self._condition.notify()

            return True

    # ==========================================================
    # GET RESULT
    # ==========================================================

    def get(self, face_key):

        with self._lock:

            result = self._results.get(face_key)

            if result is None:
                return None

            return dict(result)

    # ==========================================================
    # CLEANUP FACE STATE
    # ==========================================================

    def cleanup(self, active_face_keys):

        active = set(active_face_keys)

        with self._condition:

            stale_pending = [
                key
                for key in self._pending
                if key not in active
            ]

            for key in stale_pending:
                del self._pending[key]

            stale_submit = [
                key
                for key in self._last_submitted
                if key not in active
            ]

            for key in stale_submit:
                del self._last_submitted[key]

            # Keep results for a while so temporary detection loss
            # does not immediately erase the displayed age/gender.
            # Stale result removal is handled conservatively below.
            now = time.perf_counter()

            stale_results = [
                key
                for key, timestamp in self._last_result_time.items()
                if (
                    key not in active
                    and now - timestamp > 15.0
                )
            ]

            for key in stale_results:

                self._results.pop(key, None)
                self._age_history.pop(key, None)
                self._gender_history.pop(key, None)
                self._last_result_time.pop(key, None)

    # ==========================================================
    # CLOSE
    # ==========================================================

    def close(self):

        with self._condition:
            self._stop = True
            self._condition.notify_all()

        if self._worker.is_alive():
            self._worker.join(timeout=2.0)

    # ==========================================================
    # MODEL SETUP
    # ==========================================================

    def _load_model(self):

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"FaceAge model not found: {self.model_path}"
            )

        available = ort.get_available_providers()

        if "CUDAExecutionProvider" in available:
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            providers = [
                "CPUExecutionProvider",
            ]

        print("[FaceAge] Available providers:", available)
        print("[FaceAge] Loading model:", self.model_path)

        session = ort.InferenceSession(
            str(self.model_path),
            providers=providers,
        )

        inputs = session.get_inputs()

        if not inputs:
            raise RuntimeError(
                "FaceAge model has no inputs."
            )

        input_name = inputs[0].name

        outputs = session.get_outputs()

        if len(outputs) < 2:
            raise RuntimeError(
                "FaceAge model must expose age and gender outputs."
            )

        age_output_index = 0
        gender_output_index = 1

        for index, output in enumerate(outputs):
            name = output.name.lower()

            if "age" in name:
                age_output_index = index

            elif "gender" in name or "sex" in name:
                gender_output_index = index

        self._session = session
        self._input_name = input_name
        self._age_output_index = age_output_index
        self._gender_output_index = gender_output_index

        print(
            "[FaceAge] Active providers:",
            session.get_providers(),
        )

        print(
            "[FaceAge] Input:",
            input_name,
            inputs[0].shape,
            inputs[0].type,
        )

        for output in outputs:
            print(
                "[FaceAge] Output:",
                output.name,
                output.shape,
                output.type,
            )

    # ==========================================================
    # PREPROCESSING
    # ==========================================================

    @staticmethod
    def _sigmoid(values):
        values = np.clip(values, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-values))

    @staticmethod
    def _preprocess(face_bgr):

        face_rgb = cv2.cvtColor(
            face_bgr,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(face_rgb)

        image = image.resize(
            (224, 224),
            Image.Resampling.BICUBIC,
        )

        image = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )

        image = (image - mean) / std

        image = np.transpose(
            image,
            (2, 0, 1),
        )

        image = np.expand_dims(
            image,
            axis=0,
        )

        return np.ascontiguousarray(
            image,
            dtype=np.float32,
        )

    # ==========================================================
    # INFERENCE
    # ==========================================================

    def _predict(self, face_crop):

        tensor = self._preprocess(face_crop)

        outputs = self._session.run(
            None,
            {
                self._input_name: tensor,
            },
        )

        age_logits = np.asarray(
            outputs[self._age_output_index]
        )

        gender_logits = np.asarray(
            outputs[self._gender_output_index]
        )

        if age_logits.ndim > 1:
            age_logits = age_logits[0]

        if gender_logits.ndim > 1:
            gender_logits = gender_logits[0]

        # FaceAge ClientScan age decoding:
        # age = sum(sigmoid(age_logits)).
        age_probabilities = self._sigmoid(
            age_logits
        )

        age = float(
            np.sum(age_probabilities)
        )

        gender_index = int(
            np.argmax(gender_logits)
        )

        # ClientScan gender head: 0 = female, 1 = male.
        gender = (
            "Male"
            if gender_index == 1
            else "Female"
        )

        return age, gender

    # ==========================================================
    # STABILIZATION
    # ==========================================================

    def _stabilize(
        self,
        face_key,
        raw_age,
        raw_gender,
    ):

        ages = self._age_history.setdefault(
            face_key,
            deque(maxlen=self.history_size),
        )

        genders = self._gender_history.setdefault(
            face_key,
            deque(maxlen=self.history_size),
        )

        ages.append(float(raw_age))
        genders.append(raw_gender)

        # Median gives much better resistance to occasional jumps.
        stable_age = float(
            np.median(
                np.asarray(ages, dtype=np.float32)
            )
        )

        stable_gender = Counter(
            genders
        ).most_common(1)[0][0]

        # Stability is NOT a calibrated model confidence.
        # It only measures how tightly recent age estimates cluster.
        if len(ages) >= 2:
            spread = float(
                np.std(
                    np.asarray(ages, dtype=np.float32)
                )
            )

            stability = max(
                0.0,
                min(
                    1.0,
                    1.0 - (spread / 5.0),
                ),
            )

        else:
            stability = 0.0

        return stable_age, stable_gender, stability

    # ==========================================================
    # WORKER
    # ==========================================================

    def _worker_loop(self):

        print("[FaceAge] Background worker starting...")

        try:
            self._load_model()

            with self._condition:
                self._ready = True
                self._error = None
                self._condition.notify_all()

            print("[FaceAge] Model ready.")

        except Exception as error:

            with self._condition:
                self._error = str(error)
                self._ready = False
                self._condition.notify_all()

            print(
                "[FaceAge] MODEL INITIALIZATION ERROR:",
                repr(error),
            )

            return

        while True:

            with self._condition:

                while (
                    not self._stop
                    and not self._pending
                ):
                    self._condition.wait(
                        timeout=0.25
                    )

                if self._stop:
                    return

                # Choose the face whose result is oldest.
                def priority(key):
                    return self._last_result_time.get(
                        key,
                        0.0,
                    )

                selected_key = min(
                    self._pending.keys(),
                    key=priority,
                )

                face_crop = self._pending.pop(
                    selected_key
                )

                self._busy = True

            start = time.perf_counter()

            try:
                raw_age, raw_gender = self._predict(
                    face_crop
                )

                with self._condition:

                    stable_age, stable_gender, stability = (
                        self._stabilize(
                            selected_key,
                            raw_age,
                            raw_gender,
                        )
                    )

                    self._results[selected_key] = {
                        "age": stable_age,
                        "gender": stable_gender,
                        "raw_age": raw_age,
                        "raw_gender": raw_gender,
                        "stability": stability,
                        "updated_at": time.time(),
                    }

                    self._last_result_time[
                        selected_key
                    ] = time.perf_counter()

                    self._error = None

                elapsed = (
                    time.perf_counter()
                    - start
                )


            except Exception as error:

                with self._condition:
                    self._error = str(error)

                print(
                    "[FaceAge] INFERENCE ERROR:",
                    repr(error),
                )

            finally:

                with self._condition:
                    self._busy = False
