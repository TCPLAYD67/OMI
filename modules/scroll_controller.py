import time
import math
import pyautogui


class ScrollController:

    def __init__(self, sound):

        self.sound = sound

        # =====================================
        # Camera Resolution
        # =====================================

        self.camera_height = 480

        # =====================================
        # Scroll Zones
        # =====================================

        self.top_zone = 160
        self.bottom_zone = 320

        # =====================================
        # Scroll Settings
        # =====================================

        self.scroll_amount = 60

        self.cooldown = 0.05
        self.last_scroll = 0

        # =====================================
        # Gesture Hold
        # =====================================

        self.was_scrolling = False
        self.gesture_start = None

        # Hold peace sign before enabling scroll
        self.activation_delay = 0.15

        # Fingers must be separated by this much
        self.min_finger_gap = 25

    # =====================================
    # Peace Gesture
    # =====================================

    def is_scroll_gesture(self, hand):

        landmarks = hand["landmarks"]

        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        index_pip = landmarks[6]
        middle_pip = landmarks[10]
        ring_pip = landmarks[14]
        pinky_pip = landmarks[18]

        index_up = index_tip[1] < index_pip[1]
        middle_up = middle_tip[1] < middle_pip[1]

        ring_down = ring_tip[1] > ring_pip[1]
        pinky_down = pinky_tip[1] > pinky_pip[1]

        finger_gap = math.hypot(
            index_tip[0] - middle_tip[0],
            index_tip[1] - middle_tip[1]
        )

        return (
            index_up
            and middle_up
            and ring_down
            and pinky_down
            and finger_gap > self.min_finger_gap
        )

    # =====================================
    # Update
    # =====================================

    def update(self, hand):

        scrolling = self.is_scroll_gesture(hand)

        current_time = time.time()

        # ---------------------------------
        # Gesture Lost
        # ---------------------------------

        if not scrolling:

            self.was_scrolling = False
            self.gesture_start = None

            return False

        # ---------------------------------
        # Gesture Started
        # ---------------------------------

        if self.gesture_start is None:

            self.gesture_start = current_time

            return "idle"

        # ---------------------------------
        # Wait for Hold
        # ---------------------------------

        if current_time - self.gesture_start < self.activation_delay:

            return "idle"

        # ---------------------------------
        # Enter Scroll Mode
        # ---------------------------------

        if not self.was_scrolling:

            self.sound.scroll_start()

            self.was_scrolling = True

        # ---------------------------------
        # Cooldown
        # ---------------------------------

        if current_time - self.last_scroll < self.cooldown:

            return "idle"

        y = hand["landmarks"][8][1]

        # ---------------------------------
        # Scroll Up
        # ---------------------------------

        if y < self.top_zone:

            pyautogui.scroll(self.scroll_amount)

            self.last_scroll = current_time

            return "up"

        # ---------------------------------
        # Scroll Down
        # ---------------------------------

        elif y > self.bottom_zone:

            pyautogui.scroll(-self.scroll_amount)

            self.last_scroll = current_time

            return "down"

        # ---------------------------------
        # Middle Zone
        # ---------------------------------

        self.last_scroll = current_time

        return "idle"