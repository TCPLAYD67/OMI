import math
import time
import pyautogui


class MouseButtons:

    def __init__(self):

        # Current state
        self.was_pinching = False
        self.dragging = False

        # Pinch info
        self.pinch_start_time = None
        self.start_position = None

        # Tuning
        self.click_time = 0.35          # seconds
        self.drag_distance = 15         # pixels

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0

    def update(self, hand):

        pinching = hand.get("pinching", False)

        index_x, index_y = hand["landmarks"][8]

        current_time = time.time()

        # -----------------------------------
        # Pinch Started
        # -----------------------------------

        if pinching and not self.was_pinching:

            self.pinch_start_time = current_time
            self.start_position = (index_x, index_y)

        # -----------------------------------
        # While Pinching
        # -----------------------------------

        if pinching and not self.dragging:

            if self.start_position is not None:

                dx = index_x - self.start_position[0]
                dy = index_y - self.start_position[1]

                distance = math.hypot(dx, dy)

                held_time = current_time - self.pinch_start_time

                # Start drag if moved enough
                # OR held intentionally
                if (
                    distance >= self.drag_distance
                    or held_time >= self.click_time
                ):

                    pyautogui.mouseDown()
                    self.dragging = True

        # -----------------------------------
        # Pinch Released
        # -----------------------------------

        if not pinching and self.was_pinching:

            held_time = (
                current_time - self.pinch_start_time
                if self.pinch_start_time is not None
                else 0
            )

            if self.dragging:

                pyautogui.mouseUp()
                self.dragging = False

            else:

                if (
                    held_time < self.click_time
                    and self.start_position is not None
                ):

                    dx = index_x - self.start_position[0]
                    dy = index_y - self.start_position[1]

                    distance = math.hypot(dx, dy)

                    if distance < self.drag_distance:

                        pyautogui.click()

            self.start_position = None
            self.pinch_start_time = None

        self.was_pinching = pinching