"""
YOLO Benchmark Tool
config.py
"""

from pathlib import Path

# ==========================================================
# PROJECT
# ==========================================================

PROJECT_NAME = "YOLO Benchmark Tool"
VERSION = "2.0"

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).parent

MODELS_DIR = BASE_DIR

VIDEOS_DIR = BASE_DIR

RESULTS_DIR = BASE_DIR / "results"

# automatycznie utwórz katalog wyników
RESULTS_DIR.mkdir(exist_ok=True)

# ==========================================================
# INFLUXDB
# ==========================================================

INFLUX_HOST = "localhost"

INFLUX_PORT = 8086

INFLUX_DATABASE = "yolo"

# ==========================================================
# AVAILABLE DEVICES
# ==========================================================

DEVICES = {

    1: {
        "name": "Intel NUC",
        "tag": "nuc"
    },

    2: {
        "name": "Intel Joule",
        "tag": "joule"
    }

}

# ==========================================================
# BENCHMARK TYPES
# ==========================================================

BENCHMARKS = {

    1: "models",

    2: "confidence",

    3: "imgsz",

    4: "video"

}

# ==========================================================
# IMAGE SIZE
# ==========================================================

IMAGE_SIZES = [

    320,

    640,

    960,

    1280

]

# ==========================================================
# CONFIDENCE
# ==========================================================

CONFIDENCE_VALUES = [

    0.10,

    0.15,

    0.25,

    0.35,

    0.50,

    0.75

]

# ==========================================================
# PREVIEW
# ==========================================================

PREVIEW = {

    1: True,

    2: False

}

LIVE_STREAMS = {

    "__TIMES__": {
        "name": "🇺🇸 Times Square (Live)",
        "youtube": "https://www.youtube.com/live/z-jYdOIKcTQ",
    },

    "__TOKYO__": {
        "name": "🇯🇵 Tokyo Crossing (Live)",
        "youtube": "https://www.youtube.com/live/DjdUEyjx8GM",
    },

    "__SHIBUYA__": {
        "name": "🚆 Shibuya Crossing (Live)",
        "youtube": "https://www.youtube.com/live/dfVK7ld38Ys",
    },

    "__VENICE__": {
        "name": "🇮🇹 Venice (Live)",
        "youtube": "https://www.youtube.com/live/mt7uE-n0YPI",
    },

    "__GDANSK__": {
        "name": "⚓ Gdańsk (Live)",
        "youtube": "https://www.youtube.com/watch?v=WAm5hsBFQa0",
    },

    "__DAVAO__": {
        "name": "🇵🇭 Davao City (Live)",
        "youtube": "https://www.youtube.com/live/mt7uE-n0YPI",
    },

}