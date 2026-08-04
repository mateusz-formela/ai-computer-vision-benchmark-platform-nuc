"""
YOLO Benchmark Tool
main.py
"""

from config import (
    PROJECT_NAME,
    VERSION,
    DEVICES,
    BENCHMARKS,
    IMAGE_SIZES,
    CONFIDENCE_VALUES,
    MODELS_DIR,
    VIDEOS_DIR
)

from utils import (
    get_models,
    get_videos,
    choose_dict,
    choose_from_list,
    choose_imgsz,
    choose_confidence,
    choose_yes_no
)

from benchmark import run_benchmark


def main():

    print()

    print("=" * 60)
    print(PROJECT_NAME)
    print(f"Version {VERSION}")
    print("=" * 60)

    # =====================================================
    # DEVICE
    # =====================================================

    device = choose_dict(

        "Select device",

        DEVICES

    )

    # =====================================================
    # BENCHMARK
    # =====================================================

    benchmark = choose_dict(

        "Select benchmark",

        BENCHMARKS

    )

    # =====================================================
    # MODEL
    # =====================================================

    models = get_models(MODELS_DIR)

    if not models:

        print("No models found.")

        return

    model = choose_from_list(

        "Available models",

        models

    )

    # =====================================================
    # VIDEO
    # =====================================================

    videos = get_videos(VIDEOS_DIR)

    if not videos:

        print("No videos found.")

        return

    video = choose_from_list(

        "Available videos",

        videos

    )

    # =====================================================
    # IMGSZ
    # =====================================================

    imgsz = choose_imgsz(

        IMAGE_SIZES

    )

    # =====================================================
    # CONFIDENCE
    # =====================================================

    confidence = choose_confidence(

        CONFIDENCE_VALUES

    )

    # =====================================================
    # PREVIEW
    # =====================================================

    preview = choose_yes_no()

    # =====================================================
    # SUMMARY
    # =====================================================

    print()

    print("=" * 60)

    print("CONFIGURATION")

    print("=" * 60)

    print(f"Device      : {device['name']}")

    print(f"Benchmark   : {benchmark}")

    print(f"Model       : {model.name}")

    print(f"Video       : {video.name}")

    print(f"Image Size  : {imgsz}")

    print(f"Confidence  : {confidence}")

    print(f"Preview     : {preview}")

    print("=" * 60)

    start = input(

        "\nStart benchmark? [y/n]: "

    ).lower()

    if start != "y":

        print("Cancelled.")

        return

    run_id = run_benchmark(

        device_name=device["tag"],

        benchmark_name=benchmark,

        model_path=model,

        video_path=video,

        imgsz=imgsz,

        confidence=confidence,

        preview=preview

    )

    print()

    print("=" * 60)

    print("Benchmark finished.")

    print(f"Run ID: {run_id}")

    print("=" * 60)


if __name__ == "__main__":

    main()