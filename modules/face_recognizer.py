import math


class FaceRecognizer:

    def __init__(self, database):

        self.database = database

        # Initial threshold for our current
        # landmark-based representation.
        self.match_threshold = 0.055

        # Require several consecutive matching
        # frames before confirming identity.
        self.required_confirmations = 4

        # face_id -> recognition state
        self.states = {}

    def calculate_distance(
        self,
        features_a,
        features_b
    ):

        if not features_a or not features_b:
            return float("inf")

        if len(features_a) != len(features_b):
            return float("inf")

        squared_sum = 0.0

        for a, b in zip(
            features_a,
            features_b
        ):

            difference = a - b

            squared_sum += (
                difference * difference
            )

        mean_squared_error = (
            squared_sum
            / len(features_a)
        )

        return math.sqrt(
            mean_squared_error
        )

    def find_best_match(self, features):

        people = self.database.get_people()

        if not people:

            return {
                "name": "Unknown",
                "distance": None,
                "matched": False,
                "score": 0.0
            }

        best_person = None
        best_distance = float("inf")

        for person in people:

            # New database format.
            samples = person.get(
                "samples",
                []
            )

            # Backwards compatibility with
            # the previous single-sample format.
            if not samples:

                old_features = person.get(
                    "features"
                )

                if old_features:

                    samples = [
                        old_features
                    ]

            for stored_features in samples:

                distance = (
                    self.calculate_distance(
                        features,
                        stored_features
                    )
                )

                if distance < best_distance:

                    best_distance = distance
                    best_person = person

        if best_person is None:

            return {
                "name": "Unknown",
                "distance": None,
                "matched": False,
                "score": 0.0
            }

        matched = (
            best_distance
            <= self.match_threshold
        )

        if matched:

            score = max(
                0.0,
                1.0 - (
                    best_distance
                    / self.match_threshold
                )
            )

            name = best_person.get(
                "name",
                "Unknown"
            )

        else:

            score = 0.0
            name = "Unknown"

        return {
            "name": name,
            "distance": best_distance,
            "matched": matched,
            "score": score
        }

    def recognize(
        self,
        face,
        face_id=None,
        features=None
    ):

        if features is None:

            return {
                "name": "Unknown",
                "distance": None,
                "matched": False,
                "score": 0.0
            }

        result = self.find_best_match(
            features
        )

        if face_id is None:

            return result

        if face_id not in self.states:

            self.states[face_id] = {
                "candidate": None,
                "confirmations": 0,
                "identity": "Unknown"
            }

        state = self.states[face_id]

        # -------------------------
        # No valid match
        # -------------------------

        if not result["matched"]:

            state["candidate"] = None
            state["confirmations"] = 0
            state["identity"] = "Unknown"

            return {
                "name": "Unknown",
                "distance": result["distance"],
                "matched": False,
                "score": 0.0
            }

        # -------------------------
        # Valid match
        # -------------------------

        candidate = result["name"]

        if state["candidate"] == candidate:

            state["confirmations"] += 1

        else:

            state["candidate"] = candidate
            state["confirmations"] = 1

        if (
            state["confirmations"]
            >= self.required_confirmations
        ):

            state["identity"] = candidate

        return {
            "name": state["identity"],
            "distance": result["distance"],
            "matched": (
                state["identity"]
                != "Unknown"
            ),
            "score": result["score"]
        }

    def cleanup(self, active_face_ids):

        active_face_ids = set(
            active_face_ids
        )

        stale_ids = [
            face_id
            for face_id in self.states
            if face_id not in active_face_ids
        ]

        for face_id in stale_ids:

            del self.states[face_id]