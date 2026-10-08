import json
from pathlib import Path


class FaceDatabase:

    def __init__(self):

        project_root = (
            Path(__file__).resolve().parent.parent
        )

        self.data_directory = (
            project_root / "data"
        )

        self.database_path = (
            self.data_directory / "faces.json"
        )

        self.data_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.database = {
            "people": []
        }

        self.load()

    def load(self):

        if not self.database_path.exists():

            self.save()

            return

        try:

            with open(
                self.database_path,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

            if (
                isinstance(data, dict)
                and isinstance(
                    data.get("people"),
                    list
                )
            ):

                self.database = data

            else:

                self.database = {
                    "people": []
                }

        except (
            json.JSONDecodeError,
            OSError
        ):

            self.database = {
                "people": []
            }

    def save(self):

        with open(
            self.database_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.database,
                file,
                indent=4
            )

    def add_person(
        self,
        name,
        samples
    ):

        name = name.strip()

        if not name:
            return False

        if not samples:
            return False

        person_data = {
            "name": name,
            "samples": samples,

            # Keep the first sample in the old
            # field as well for compatibility.
            "features": samples[0]
        }

        # Replace an existing person with the same name.
        for index, person in enumerate(
            self.database["people"]
        ):

            if (
                person.get("name", "")
                .lower()
                == name.lower()
            ):

                self.database["people"][index] = (
                    person_data
                )

                self.save()

                return True

        self.database["people"].append(
            person_data
        )

        self.save()

        return True

    def get_people(self):

        return self.database.get(
            "people",
            []
        )

    def count(self):

        return len(
            self.database.get(
                "people",
                []
            )
        )