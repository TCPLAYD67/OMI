from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class AgeEstimator:

    def __init__(self):

        project_root = (
            Path(__file__).resolve().parent.parent
        )

        self.model_path = (
            project_root
            / "models"
            / "fastface_age.onnx"
        )

        if not self.model_path.exists():

            raise FileNotFoundError(
                "\nFastFace model not found.\n\n"
                f"Expected:\n{self.model_path}\n"
            )

        print(
            "[AgeEstimator] Loading FastFace Large-128..."
        )

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider"
            ]
        )

        print(
            "[AgeEstimator] Providers:",
            self.session.get_providers()
        )

        # -------------------------------------------------
        # Input information
        # -------------------------------------------------

        input_info = (
            self.session.get_inputs()[0]
        )

        self.input_name = input_info.name
        self.input_shape = input_info.shape
        self.input_type = input_info.type

        print(
            "[AgeEstimator] Input:",
            self.input_name,
            self.input_shape,
            self.input_type
        )

        # -------------------------------------------------
        # Output information
        # -------------------------------------------------

        self.output_names = [
            output.name
            for output in self.session.get_outputs()
        ]

        self.output_shapes = [
            output.shape
            for output in self.session.get_outputs()
        ]

        print(
            "[AgeEstimator] Outputs:",
            list(
                zip(
                    self.output_names,
                    self.output_shapes
                )
            )
        )

        self.input_width = 128
        self.input_height = 128

        # FastFace release configuration uses a 20%
        # face crop margin.
        self.face_crop_margin = 0.20

    # =====================================================
    # FACE CROP
    # =====================================================

    def crop_face(
        self,
        frame,
        face
    ):

        bbox = face.get(
            "bbox"
        )

        if bbox is None:

            return None

        x1, y1, x2, y2 = bbox

        face_width = (
            x2 - x1
        )

        face_height = (
            y2 - y1
        )

        if (
            face_width <= 1
            or face_height <= 1
        ):

            return None

        # -------------------------------------------------
        # FastFace uses a 0.2 face-crop margin.
        # -------------------------------------------------

        margin_x = (
            face_width
            * self.face_crop_margin
        )

        margin_y = (
            face_height
            * self.face_crop_margin
        )

        left = int(
            x1 - margin_x
        )

        top = int(
            y1 - margin_y
        )

        right = int(
            x2 + margin_x
        )

        bottom = int(
            y2 + margin_y
        )

        frame_height, frame_width = (
            frame.shape[:2]
        )

        left = max(
            0,
            left
        )

        top = max(
            0,
            top
        )

        right = min(
            frame_width,
            right
        )

        bottom = min(
            frame_height,
            bottom
        )

        if (
            right <= left
            or bottom <= top
        ):

            return None

        crop = frame[
            top:bottom,
            left:right
        ]

        if crop.size == 0:

            return None

        return crop.copy()

    # =====================================================
    # PREPROCESS
    # =====================================================

    def preprocess(
        self,
        face_image
    ):

        if face_image is None:

            return None

        image = cv2.resize(
            face_image,
            (
                self.input_width,
                self.input_height
            ),
            interpolation=cv2.INTER_AREA
        )

        # Model expects RGB.
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # uint8 -> float32 [0, 1]
        image = (
            image.astype(
                np.float32
            )
            / 255.0
        )

        # -------------------------------------------------
        # Adapt automatically to the model's tensor layout.
        # -------------------------------------------------

        shape = self.input_shape

        if (
            len(shape) == 4
            and shape[1] == 3
        ):

            # NCHW
            image = np.transpose(
                image,
                (2, 0, 1)
            )

            image = np.expand_dims(
                image,
                axis=0
            )

        elif (
            len(shape) == 4
            and shape[-1] == 3
        ):

            # NHWC
            image = np.expand_dims(
                image,
                axis=0
            )

        else:

            raise RuntimeError(
                "Unsupported FastFace input shape: "
                f"{shape}"
            )

        return np.ascontiguousarray(
            image,
            dtype=np.float32
        )

    # =====================================================
    # FIND AGE OUTPUT
    # =====================================================

    def _find_age_output(
        self,
        outputs
    ):

        # -------------------------------------------------
        # First prefer an output explicitly named "age".
        # -------------------------------------------------

        for index, name in enumerate(
            self.output_names
        ):

            if "age" in name.lower():

                return np.asarray(
                    outputs[index]
                ).squeeze()

        # -------------------------------------------------
        # FastFace age head is a 0..100 distribution.
        # Therefore look for 101 values.
        # -------------------------------------------------

        for output in outputs:

            array = np.asarray(
                output
            ).squeeze()

            if array.size == 101:

                return array

        # -------------------------------------------------
        # Support scalar numeric-age outputs.
        # -------------------------------------------------

        for output in outputs:

            array = np.asarray(
                output
            ).squeeze()

            if array.size == 1:

                return array

        return None

    # =====================================================
    # DECODE AGE
    # =====================================================

    def _decode_age(
        self,
        age_output
    ):

        if age_output is None:

            return None

        values = np.asarray(
            age_output,
            dtype=np.float32
        ).flatten()

        # -------------------------------------------------
        # Scalar age output.
        # -------------------------------------------------

        if values.size == 1:

            age = float(
                values[0]
            )

            return max(
                0.0,
                min(100.0, age)
            )

        # -------------------------------------------------
        # 101-class age distribution.
        # -------------------------------------------------

        if values.size == 101:

            # If already normalized probabilities,
            # keep them. Otherwise treat as logits.
            total = float(
                np.sum(values)
            )

            all_non_negative = bool(
                np.all(values >= 0)
            )

            if (
                all_non_negative
                and abs(total - 1.0) < 0.01
            ):

                probabilities = values

            else:

                shifted = (
                    values
                    - np.max(values)
                )

                exponentials = np.exp(
                    shifted
                )

                probability_sum = (
                    np.sum(
                        exponentials
                    )
                )

                if probability_sum <= 1e-12:

                    return None

                probabilities = (
                    exponentials
                    / probability_sum
                )

            ages = np.arange(
                101,
                dtype=np.float32
            )

            age = float(
                np.sum(
                    ages
                    * probabilities
                )
            )

            return max(
                0.0,
                min(100.0, age)
            )

        return None

    # =====================================================
    # ESTIMATE
    # =====================================================

    def estimate(
        self,
        frame,
        face
    ):

        face_image = self.crop_face(
            frame,
            face
        )

        if face_image is None:

            return None

        tensor = self.preprocess(
            face_image
        )

        outputs = self.session.run(
            None,
            {
                self.input_name:
                    tensor
            }
        )

        age_output = (
            self._find_age_output(
                outputs
            )
        )

        age = (
            self._decode_age(
                age_output
            )
        )

        if age is None:

            return None

        return {
            "age": age
        }

    # =====================================================
    # CLOSE
    # =====================================================

    def close(self):

        self.session = None