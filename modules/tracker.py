import math


class CentroidTracker:

    def __init__(self, max_distance=80, max_disappeared=20):

        self.next_id = 1
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

        self.objects = {}

    def update(self, detections):

        # -----------------------------
        # No detections
        # -----------------------------

        if len(detections) == 0:

            remove = []

            for object_id in self.objects:

                self.objects[object_id]["disappeared"] += 1

                if self.objects[object_id]["disappeared"] > self.max_disappeared:
                    remove.append(object_id)

            for object_id in remove:
                del self.objects[object_id]

            return []

        tracked = []
        matched_ids = set()

        # -----------------------------
        # Match detections
        # -----------------------------

        for detection in detections:

            cx, cy = detection["center"]

            best_id = None
            best_distance = float("inf")

            for object_id, obj in self.objects.items():

                if object_id in matched_ids:
                    continue

                ox, oy = obj["center"]

                distance = math.hypot(cx - ox, cy - oy)

                if distance < best_distance:
                    best_distance = distance
                    best_id = object_id

            # Existing object
            if best_id is not None and best_distance < self.max_distance:

                self.objects[best_id]["center"] = (cx, cy)
                self.objects[best_id]["disappeared"] = 0
                self.objects[best_id]["history"].append((cx, cy))

                if len(self.objects[best_id]["history"]) > 50:
                    self.objects[best_id]["history"].pop(0)

                detection["id"] = best_id
                detection["history"] = self.objects[best_id]["history"]

                matched_ids.add(best_id)

            # New object
            else:

                self.objects[self.next_id] = {
                    "center": (cx, cy),
                    "history": [(cx, cy)],
                    "disappeared": 0
                }

                detection["id"] = self.next_id
                detection["history"] = self.objects[self.next_id]["history"]

                matched_ids.add(self.next_id)

                self.next_id += 1

            tracked.append(detection)

        # -----------------------------
        # Update disappeared objects
        # -----------------------------

        remove = []

        for object_id in self.objects:

            if object_id not in matched_ids:

                self.objects[object_id]["disappeared"] += 1

                if self.objects[object_id]["disappeared"] > self.max_disappeared:
                    remove.append(object_id)

        for object_id in remove:
            del self.objects[object_id]

        return tracked