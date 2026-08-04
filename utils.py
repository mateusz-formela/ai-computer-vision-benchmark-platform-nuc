"""
YOLO Benchmark Tool
utils.py
"""

import uuid
import psutil

from pathlib import Path
from datetime import datetime
from config import LIVE_STREAMS


# ==========================================================
# RUN ID
# ==========================================================

def create_run_id():

    return (

        datetime.now().strftime("%Y%m%d_%H%M%S")

        + "_"

        + uuid.uuid4().hex[:6]

    )


# ==========================================================
# CPU TEMPERATURE
# ==========================================================

def get_cpu_temp():

    try:

        temps = psutil.sensors_temperatures()

        if "coretemp" in temps:

            for entry in temps["coretemp"]:

                if "Package" in entry.label:

                    return entry.current

    except Exception:

        pass

    return 0.0


# ==========================================================
# FILE SEARCH
# ==========================================================

def get_models(models_dir):

    models = sorted(

        Path(models_dir).glob("*.pt")

    )

    return models


def get_videos(videos_dir):

    extensions = ("*.mp4", "*.avi", "*.mkv", "*.mov")

    videos = ["__ESP32__"]

    # Publiczne streamy
    videos.extend(LIVE_STREAMS.keys())

    # Lokalne filmy
    for pattern in extensions:
        videos.extend(Path(videos_dir).glob(pattern))

    return videos


# ==========================================================
# MENU
# ==========================================================

def choose_from_list(title, items):

    print()

    print("=" * 60)

    print(title)

    print("=" * 60)

    print()


    for index, item in enumerate(items, start=1):

        print(f"{index}. {item.name}")


    while True:

        try:

            value = int(input("\nChoose: "))

            if 1 <= value <= len(items):

                return items[value - 1]

        except Exception:

            pass

        print("Invalid choice.")


# ==========================================================
# SIMPLE MENU
# ==========================================================

def choose_dict(title, dictionary):

    print()

    print("=" * 60)

    print(title)

    print("=" * 60)

    print()


    for key, value in dictionary.items():

        if isinstance(value, dict):

            print(f"{key}. {value['name']}")

        else:

            print(f"{key}. {value}")


    while True:

        try:

            value = int(input("\nChoose: "))

            if value in dictionary:

                return dictionary[value]

        except Exception:

            pass

        print("Invalid choice.")


# ==========================================================
# IMAGE SIZE
# ==========================================================

def choose_imgsz(values):

    print()

    print("=" * 60)

    print("Image Size")

    print("=" * 60)

    print()

    for index, value in enumerate(values, start=1):

        print(f"{index}. {value}")


    while True:

        try:

            choice = int(input("\nChoose: "))

            if 1 <= choice <= len(values):

                return values[choice - 1]

        except Exception:

            pass

        print("Invalid choice.")


# ==========================================================
# CONFIDENCE
# ==========================================================

def choose_confidence(values):

    print()

    print("=" * 60)

    print("Confidence")

    print("=" * 60)

    print()

    for index, value in enumerate(values, start=1):

        print(f"{index}. {value}")


    while True:

        try:

            choice = int(input("\nChoose: "))

            if 1 <= choice <= len(values):

                return values[choice - 1]

        except Exception:

            pass

        print("Invalid choice.")


# ==========================================================
# YES / NO
# ==========================================================

def choose_yes_no():

    print()

    print("=" * 60)

    print("Preview")

    print("=" * 60)

    print()

    print("1. YES")

    print("2. NO")

    while True:

        try:

            value = int(input("\nChoose: "))

            if value == 1:

                return True

            if value == 2:

                return False

        except Exception:

            pass

        print("Invalid choice.")