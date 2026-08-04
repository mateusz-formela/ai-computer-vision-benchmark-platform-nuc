from threading import Lock

STATUS = {
    "running": False,

    # Informacje o benchmarku
    "run_id": None,
    "device": None,
    "benchmark": None,
    "model": None,
    "video": None,

    # Czas
    "start_time": None,
    "end_time": None,
    "duration": None,

    # Dane live
    "frame": 0,
    "fps": 0,
    "inference": 0,
    "cpu": 0,
    "ram": 0,
    "temp": 0,
    "detections": 0,
    "persons": 0,
    "confidence": None,
    "image_size": None,
    "preview": None,

    # Podsumowanie
    "frames": 0,
    "avg_fps": 0,
    "avg_inf": 0,
    "avg_cpu": 0,
    "avg_ram": 0,
    "avg_temp": 0,
    "avg_conf": 0,
}

PREVIEW_FRAMES = {
    "nuc": None,
    "joule": None,
}

PREVIEW_LOCK = Lock()