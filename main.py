import cv2

from modules.webcam import Webcam
from modules.face_detector import FaceDetector
from modules.hand_detector import HandDetector
from modules.gesture_detector import GestureDetector
from modules.pinch_detector import PinchDetector
from modules.finger_tracker import FingerTracker
from modules.landmark_filter import LandmarkFilter
from modules.mouse_controller import MouseController
from modules.scroll_controller import ScrollController
from modules.sound_manager import SoundManager
from modules.fps_counter import FPSCounter
from modules.tracker import CentroidTracker
from modules.renderer import Renderer
from modules.hud import HUD
from modules.effects import Effects


def main():

    # ========================================================
    # CAMERA
    # ========================================================

    camera = Webcam()


    # ========================================================
    # BASIC FACE + HAND DETECTION
    # ========================================================

    face_detector = FaceDetector()

    hand_detector = HandDetector()


    # ========================================================
    # TRACKING
    # ========================================================

    face_tracker = CentroidTracker()

    hand_tracker = CentroidTracker()

    landmark_filter = LandmarkFilter()


    # ========================================================
    # GESTURE SYSTEM
    # ========================================================

    finger_tracker = FingerTracker()

    gesture_detector = GestureDetector()

    pinch_detector = PinchDetector()


    # ========================================================
    # CONTROL
    # ========================================================

    sound = SoundManager()

    mouse_controller = MouseController(
        sound
    )

    scroll_controller = ScrollController(
        sound
    )


    # ========================================================
    # VISUALS
    # ========================================================

    renderer = Renderer()

    fps_counter = FPSCounter()

    hud = HUD()

    effects = Effects()


    # ========================================================
    # WINDOW
    # ========================================================

    WINDOW_NAME = "MotionOS v1.0"

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

    control_enabled = False


    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while True:

            # ------------------------------------------------
            # CAMERA
            # ------------------------------------------------

            frame = camera.get_frame()

            if frame is None:
                break


            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            fps = fps_counter.update()


            # =================================================
            # FACE DETECTION
            # =================================================

            frame, faces = (
                face_detector.detect(
                    frame
                )
            )


            # =================================================
            # HAND DETECTION
            # =================================================

            frame, hands = (
                hand_detector.detect(
                    frame
                )
            )


            # =================================================
            # TRACKING
            # =================================================

            faces = face_tracker.update(
                faces
            )

            hands = hand_tracker.update(
                hands
            )


            # =================================================
            # HAND SMOOTHING
            # =================================================

            hands = (
                landmark_filter.update(
                    hands
                )
            )


            # =================================================
            # PRIMARY HAND
            # =================================================

            primary_hand = None


            for hand in hands:

                if hand.get(
                    "handedness"
                ) == "Right":

                    primary_hand = hand

                    break


            if (
                primary_hand is None
                and
                hands
            ):

                primary_hand = hands[0]


            # =================================================
            # PROCESS HAND
            # =================================================

            if primary_hand:

                hand = primary_hand


                # ------------------------------------------------
                # Finger tracking
                # ------------------------------------------------

                finger_tracker.update(
                    hand
                )


                # ------------------------------------------------
                # Gesture
                # ------------------------------------------------

                hand["gesture"] = (
                    gesture_detector.detect(
                        hand
                    )
                )


                # ------------------------------------------------
                # Pinch
                # ------------------------------------------------

                pinch_detector.detect(
                    hand
                )


                # =================================================
                # CONTROL ENABLED
                # =================================================

                if control_enabled:

                    scroll = (
                        scroll_controller.update(
                            hand
                        )
                    )


                    # --------------------------------------------
                    # SCROLL UP
                    # --------------------------------------------

                    if scroll == "up":

                        effects.scroll(
                            hand["landmarks"][8],
                            "up"
                        )


                    # --------------------------------------------
                    # SCROLL DOWN
                    # --------------------------------------------

                    elif scroll == "down":

                        effects.scroll(
                            hand["landmarks"][8],
                            "down"
                        )


                    # --------------------------------------------
                    # CURSOR
                    # --------------------------------------------

                    elif scroll is False:

                        mouse_controller.update(
                            hand,
                            frame
                        )


                        # ----------------------------------------
                        # Mouse buttons
                        # ----------------------------------------

                        click = (
                            mouse_controller.handle_buttons(
                                hand
                            )
                        )


                        if click == "left":

                            effects.click(
                                hand["landmarks"][8],
                                (0, 255, 0)
                            )

                            effects.notify(
                                "LEFT CLICK",
                                (0, 255, 0)
                            )


                        elif click == "right":

                            effects.click(
                                hand["landmarks"][8],
                                (255, 0, 0)
                            )

                            effects.notify(
                                "RIGHT CLICK",
                                (255, 0, 0)
                            )


            # =================================================
            # DRAW FACES
            # =================================================

            frame = renderer.draw_faces(
                frame,
                faces
            )


            # =================================================
            # DRAW HANDS
            # =================================================

            frame = renderer.draw_hands(
                frame,
                hands
            )


            # =================================================
            # EFFECTS
            # =================================================

            frame = effects.draw(
                frame,
                hands
            )


            # =================================================
            # HUD
            # =================================================

            gesture = "None"


            if primary_hand:

                gesture = primary_hand.get(
                    "gesture",
                    "Unknown"
                )


            frame = hud.draw(
                frame,
                fps,
                len(faces),
                len(hands),
                gesture,
                control_enabled
            )


            # =================================================
            # DISPLAY
            # =================================================

            cv2.imshow(
                WINDOW_NAME,
                frame
            )


            # =================================================
            # KEYBOARD
            # =================================================

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )


            # =================================================
            # C - CONTROL
            # =================================================

            if key == ord("c"):

                control_enabled = (
                    not control_enabled
                )


                if control_enabled:

                    sound.enable()

                    hud.show_notification(
                        "CONTROL ENABLED",
                        (0, 255, 0)
                    )

                    effects.notify(
                        "CONTROL ENABLED",
                        (0, 255, 0)
                    )


                else:

                    sound.disable()

                    hud.show_notification(
                        "CONTROL DISABLED",
                        (0, 0, 255)
                    )

                    effects.notify(
                        "CONTROL DISABLED",
                        (0, 0, 255)
                    )


            # =================================================
            # E - HUD
            # =================================================

            elif key == ord("e"):

                hud.toggle()


            # =================================================
            # F - FULLSCREEN
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
            # Q - QUIT
            # =================================================

            elif key == ord("q"):

                break


    finally:

        camera.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()