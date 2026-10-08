from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class ProfileCard:

    def __init__(self):

        self.card_width = 1000
        self.card_height = 650

        self.background = (25, 25, 25)
        self.primary_text = (255, 255, 255)
        self.secondary_text = (180, 180, 180)
        self.accent = (0, 220, 255)

    # =====================================================
    # FIT IMAGE INTO BOX
    # =====================================================

    def _fit_image(
        self,
        image,
        width,
        height
    ):

        if image is None:
            return None

        image_height, image_width = (
            image.shape[:2]
        )

        if (
            image_width <= 0
            or image_height <= 0
        ):
            return None

        scale = min(
            width / image_width,
            height / image_height
        )

        new_width = max(
            1,
            int(image_width * scale)
        )

        new_height = max(
            1,
            int(image_height * scale)
        )

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        canvas = np.full(
            (
                height,
                width,
                3
            ),
            25,
            dtype=np.uint8
        )

        x = (
            width - new_width
        ) // 2

        y = (
            height - new_height
        ) // 2

        canvas[
            y:y + new_height,
            x:x + new_width
        ] = resized

        return canvas

    # =====================================================
    # TEXT
    # =====================================================

    def _draw_text(
        self,
        image,
        text,
        position,
        size=0.8,
        color=None,
        thickness=2
    ):

        if color is None:
            color = self.primary_text

        cv2.putText(
            image,
            str(text),
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            color,
            thickness,
            cv2.LINE_AA
        )

    # =====================================================
    # CREATE PROFILE CARD
    # =====================================================

    def create(
        self,
        person_name,
        face_image,
        sample_count,
        output_path,
        age=None,
        age_confidence=None,
        extra_info=None
    ):

        card = np.full(
            (
                self.card_height,
                self.card_width,
                3
            ),
            25,
            dtype=np.uint8
        )

        # =================================================
        # HEADER
        # =================================================

        cv2.rectangle(
            card,
            (0, 0),
            (
                self.card_width,
                85
            ),
            self.accent,
            -1
        )

        self._draw_text(
            card,
            "OMI PROFILE",
            (35, 55),
            size=1.1,
            color=(20, 20, 20),
            thickness=3
        )

        # =================================================
        # FACE
        # =================================================

        face_x = 40
        face_y = 125

        face_width = 370
        face_height = 430

        cv2.rectangle(
            card,
            (
                face_x - 5,
                face_y - 5
            ),
            (
                face_x
                + face_width
                + 5,
                face_y
                + face_height
                + 5
            ),
            (90, 90, 90),
            2
        )

        fitted_face = self._fit_image(
            face_image,
            face_width,
            face_height
        )

        if fitted_face is not None:

            card[
                face_y:
                face_y + face_height,

                face_x:
                face_x + face_width
            ] = fitted_face

        # =================================================
        # PERSON NAME
        # =================================================

        info_x = 470

        self._draw_text(
            card,
            person_name,
            (info_x, 170),
            size=1.2,
            color=self.accent,
            thickness=3
        )

        cv2.line(
            card,
            (info_x, 190),
            (940, 190),
            (80, 80, 80),
            2
        )

        # =================================================
        # NAME
        # =================================================

        self._draw_text(
            card,
            "Name:",
            (info_x, 240),
            size=0.7,
            color=self.secondary_text
        )

        self._draw_text(
            card,
            person_name,
            (620, 240),
            size=0.7
        )

        # =================================================
        # DATE
        # =================================================

        registered = datetime.now().strftime(
            "%d/%m/%Y"
        )

        self._draw_text(
            card,
            "Registered:",
            (info_x, 290),
            size=0.7,
            color=self.secondary_text
        )

        self._draw_text(
            card,
            registered,
            (620, 290),
            size=0.7
        )

        # =================================================
        # SAMPLES
        # =================================================

        self._draw_text(
            card,
            "Face samples:",
            (info_x, 340),
            size=0.7,
            color=self.secondary_text
        )

        self._draw_text(
            card,
            sample_count,
            (620, 340),
            size=0.7
        )

        # =================================================
        # AGE
        # =================================================

        self._draw_text(
            card,
            "Estimated age:",
            (info_x, 390),
            size=0.7,
            color=self.secondary_text
        )

        if age is None:

            age_text = "Not available"

        else:

            age_text = str(
                round(age, 1)
            )

            if age_confidence is not None:

                age_text += (
                    f" "
                    f"({age_confidence:.0%})"
                )

        self._draw_text(
            card,
            age_text,
            (620, 390),
            size=0.7
        )

        # =================================================
        # EXTRA INFORMATION
        # =================================================

        if extra_info:

            y = 440

            for key, value in (
                extra_info.items()
            ):

                if y > 540:
                    break

                self._draw_text(
                    card,
                    f"{key}:",
                    (info_x, y),
                    size=0.65,
                    color=self.secondary_text
                )

                self._draw_text(
                    card,
                    value,
                    (620, y),
                    size=0.65
                )

                y += 42

        # =================================================
        # FOOTER
        # =================================================

        cv2.line(
            card,
            (40, 585),
            (960, 585),
            (80, 80, 80),
            2
        )

        self._draw_text(
            card,
            "OMI - Omni Multimodal Interface",
            (40, 620),
            size=0.55,
            color=self.secondary_text,
            thickness=1
        )

        # =================================================
        # SAVE
        # =================================================

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        success = cv2.imwrite(
            str(output_path),
            card,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

        if not success:

            raise RuntimeError(
                "Could not save profile card."
            )

        return str(output_path)