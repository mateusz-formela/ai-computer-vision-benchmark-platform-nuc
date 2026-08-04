"""
YOLO Benchmark Tool
benchmark.py
"""

import time
import signal
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

import cv2
import psutil
import torch
from camera_reader import CameraReader
from ultralytics import YOLO

from utils import (
    create_run_id,
    get_cpu_temp,
)

from config import LIVE_STREAMS

from youtube_stream import get_stream_url

from influx_writer import (
    save_metrics,
    save_summary,
    save_class_metrics,
)
from web.status import (
    STATUS,
    PREVIEW_FRAMES,
    PREVIEW_LOCK,
)

def stop_benchmark():
    print("\nStopping benchmark...")
    STATUS["running"] = False


def run_benchmark(
    device_name,
    benchmark_name,
    model_path,
    video_path,
    imgsz,
    confidence,
    preview=True,
):

    run_id = create_run_id()

    start_datetime = datetime.now()
    start_ts = int(start_datetime.timestamp() * 1000)

    model_name = model_path.stem
    video_name = video_path.stem

    device = "cuda" if torch.cuda.is_available() else "cpu"

    STATUS.update({
    "start_ts": start_ts,
    "running": True,
    "run_id": run_id,

    "device": device_name,
    "benchmark": benchmark_name,

    # Parametry benchmarku
    "selected_model": model_name,
    "selected_video": video_name,
    "selected_image_size": imgsz,
    "selected_confidence": confidence,
    "selected_preview": preview,

    # Zachowujemy zgodność z obecnym UI
    "model": model_name,
    "video": video_name,
    "image_size": imgsz,
    "confidence": confidence,
    "preview": preview,
        "start_time": start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
"start_time_iso": start_datetime.isoformat(timespec="milliseconds"),
        "end_time": None,
        "duration": None,
        "frames": 0,
        "avg_fps": 0,
        "avg_inf": 0,
        "avg_cpu": 0,
        "avg_ram": 0,
        "avg_temp": 0,
        "avg_conf": 0,
    })

    print("=" * 60)

    print("YOLO Benchmark")
    print("=" * 60)
    print(f"Run ID      : {run_id}")
    print(f"Device Tag  : {device_name}")
    print(f"Benchmark   : {benchmark_name}")
    print(f"Model       : {model_name}")
    print(f"Video       : {video_name}")
    print(f"ImgSz       : {imgsz}")
    print(f"Confidence  : {confidence}")
    print(f"Torch Device: {device}")
    print("=" * 60)

    model = YOLO(str(model_path))
    model.to(device)

    if str(video_path) == "__ESP32__":

        video_name = "ESP32-CAM (Live)"

        STATUS["video"] = video_name
        STATUS["selected_video"] = video_name

        cap = cv2.VideoCapture("http://192.168.1.5:81/stream")

    elif str(video_path) in LIVE_STREAMS:

        video_name = LIVE_STREAMS[str(video_path)]["name"]

        STATUS["video"] = video_name
        STATUS["selected_video"] = video_name

        stream_url = get_stream_url(str(video_path))

        print("=" * 60)
        print("YouTube stream:")
        print(stream_url)
        print("=" * 60)

        cap = CameraReader(stream_url)

    else:

        video_name = video_path.stem


        print("VIDEO_PATH =", video_path)
        print("VIDEO_EXISTS =", Path(video_path).exists())
        print("VIDEO_STR =", str(video_path))
        cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {video_path}")

    STATUS["selected_video"] = video_name
    STATUS["video"] = video_name

    print("BENCHMARK STATUS VIDEO =", STATUS["video"])

    frame_count = 0
    start_total = time.time()

    fps_history = []
    inf_history = []
    cpu_history = []
    ram_history = []
    temp_history = []
    conf_history = []

    total_detections = 0
    total_persons = 0

    class_totals = Counter()

    # ==========================================================
    # MAIN LOOP
    # ==========================================================

    status = "aborted"

    last_preview_update = 0.0

    while STATUS["running"]:

        t_cap0 = time.perf_counter()
        ret, frame = cap.read()
        t_cap1 = time.perf_counter()

        if not ret or frame is None:
            status = "finished"
            break

        if not STATUS["running"]:
            status = "stopped"
            break

        start = time.time()

        display = frame.copy()

        t_inf0 = time.perf_counter()

        results = model(
            display,
            device=device,
            verbose=False,
            imgsz=imgsz,
            conf=confidence,
        )[0]

        t_inf1 = time.perf_counter()

        if not STATUS["running"]:
            status = "stopped"
            break

        labels = []
        confidences = []

        if results.boxes is not None:

            for box in results.boxes:

                cls = int(box.cls[0])

                label = results.names[cls]

                conf = float(box.conf[0])

                labels.append(label)

                confidences.append(conf)

                if preview:

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    cv2.rectangle(
                        display,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        display,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

        counts = Counter(labels)

        class_totals.update(counts)

        if frame_count % 30 == 0:
            print("\nDetected objects:")
            for name, cnt in sorted(counts.items()):
                print(f"  {name:20} {cnt}")

        detections = len(labels)

        persons = counts.get("person", 0)

        avg_conf = (

            sum(confidences) / len(confidences)

            if confidences else 0

        )

        inference_time = time.time() - start

        fps = 1.0 / inference_time

        cpu = psutil.cpu_percent(None)

        ram = psutil.virtual_memory().percent

        temp = get_cpu_temp()
        current_duration = time.time() - start_total
        STATUS.update({
            "duration": round(current_duration, 2),
            "frame": frame_count,
            "fps": round(fps, 2),
            "inference": round(inference_time * 1000, 1),
            "cpu": round(cpu, 1),
            "ram": round(ram, 1),
            "temp": round(temp, 1),
            "detections": detections,
            "persons": persons,
	    "classes": dict(counts),
	    "duration": round(current_duration, 2),
            "current_confidence": round(avg_conf, 3),
        })

        fps_history.append(fps)

        inf_history.append(inference_time)

        cpu_history.append(cpu)

        ram_history.append(ram)

        temp_history.append(temp)

        conf_history.append(avg_conf)

        STATUS.update({
            "frames": frame_count + 1,
            "avg_fps": round(sum(fps_history) / len(fps_history), 2),
            "avg_inf": round((sum(inf_history) / len(inf_history)) * 1000, 2),
            "avg_cpu": round(sum(cpu_history) / len(cpu_history), 2),
            "avg_ram": round(sum(ram_history) / len(ram_history), 2),
            "avg_temp": round(sum(temp_history) / len(temp_history), 2),
            "avg_conf": round(sum(conf_history) / len(conf_history), 3),
        })

        total_detections += detections

        total_persons += persons

        save_metrics(

            run_id=run_id,

            device=device_name,

            benchmark=benchmark_name,

            model=model_name,

            video=video_name,

            imgsz=imgsz,

            confidence=confidence,

            frame=frame_count,

            detections=detections,

            persons=persons,

            avg_conf=avg_conf,

            fps=fps,

            inference_time=inference_time,

            cpu=cpu,

            ram=ram,

            temp=temp,

	    class_counts=counts,

        )

        save_class_metrics(

            run_id=run_id,

            device=device_name,

            benchmark=benchmark_name,

            model=model_name,

            video=video_name,

            imgsz=imgsz,

            confidence=confidence,

            frame=frame_count,

            class_counts=counts,

            )

        print(

            f"{model_name.replace('.pt','')} | "

            f"Frame:{frame_count:05d} | "

            f"DET:{detections} | "

            f"CONF:{avg_conf:.2f} | "

            f"FPS:{fps:.2f} | "

            f"INF:{inference_time*1000:.1f}ms | "

            f"CPU:{cpu:.1f}% | "

            f"RAM:{ram:.1f}% | "

            f"TEMP:{temp:.1f}°C"

        )

        frame_count += 1

        if preview:

            now = time.time()

            if now - last_preview_update >= 0.01:

                t0 = time.perf_counter()
                t_jpg0 = time.perf_counter()
                ok, jpeg = cv2.imencode(".jpg", display)
                t_jpg1 = time.perf_counter()
                t1 = time.perf_counter()

                if frame_count % 100 == 0:
                    print(f"JPEG encode: {(t1 - t0) * 1000:.1f} ms")

                if ok:
                    data = jpeg.tobytes()

                    print("JPEG:", len(data), time.time())

                    with PREVIEW_LOCK:
                        PREVIEW_FRAMES["nuc"] = data

                last_preview_update = now
                if frame_count % 30 == 0:
                    print(
                        f"READ={(t_cap1 - t_cap0) * 1000:.1f} ms | "
                        f"INFER={(t_inf1 - t_inf0) * 1000:.1f} ms | "
                        f"JPEG={(t_jpg1 - t_jpg0) * 1000:.1f} ms"
                    )

            # cv2.imshow("YOLO Benchmark", display)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

    # ==========================================================
    # END BENCHMARK
    # ==========================================================

    cap.release()

    with PREVIEW_LOCK:
        PREVIEW_FRAMES["nuc"] = None

    if preview:
        cv2.destroyAllWindows()

    duration = time.time() - start_total

    end_datetime = datetime.now()
    end_ts = int(end_datetime.timestamp() * 1000)

    if frame_count == 0:

        print("No frames processed.")

        return None

    save_summary(

    run_id=run_id,
    device=device_name,
    benchmark=benchmark_name,
    model=model_name,
    video=video_name,
    imgsz=imgsz,
    confidence=confidence,
    preview=preview,
    frames=frame_count,
    duration=duration,
    fps_history=fps_history,
    inf_history=inf_history,
    cpu_history=cpu_history,
    ram_history=ram_history,
    temp_history=temp_history,
    conf_history=conf_history,
    total_detections=total_detections,
    total_persons=total_persons,
    status=status,
    start_time=start_datetime.isoformat(timespec="seconds"),
    end_time=end_datetime.isoformat(timespec="seconds"),
    start_ts=int(start_datetime.timestamp() * 1000),
    end_ts=int(end_datetime.timestamp() * 1000),
)

    STATUS.update({
        "running": False,
        "classes": {},
        "end_time": end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time_iso": end_datetime.isoformat(timespec="milliseconds"),
        "end_ts": end_ts,
        "duration": round(duration, 2),
    })

    print()
    print("=" * 60)
    print("YOLO BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"Run ID         : {run_id}")
    print(f"Device         : {device_name}")
    print(f"Benchmark      : {benchmark_name}")
    print(f"Model          : {model_name}")
    print(f"Video          : {video_name}")
    print(f"Start time     : {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time       : {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"Frames         : {frame_count}")
    print(f"Duration       : {duration:.2f} s")

    print(f"Average FPS    : {sum(fps_history)/len(fps_history):.2f}")
    print(f"Minimum FPS    : {min(fps_history):.2f}")
    print(f"Maximum FPS    : {max(fps_history):.2f}")

    print(f"Average INF    : {(sum(inf_history)/len(inf_history))*1000:.2f} ms")

    print(f"Average CPU    : {sum(cpu_history)/len(cpu_history):.2f} %")
    print(f"Average RAM    : {sum(ram_history)/len(ram_history):.2f} %")
    print(f"Average TEMP   : {sum(temp_history)/len(temp_history):.2f} °C")

    print(f"Detections     : {total_detections}")
    print(f"Persons        : {total_persons}")

    print()
    print("Detected classes:")

    for name, cnt in sorted(class_totals.items()):
        print(f"{name:20} {cnt}")

    if conf_history:

        print(
            f"Average CONF   : "
            f"{sum(conf_history)/len(conf_history):.3f}"
        )

    print("=" * 60)
    STATUS.update({
        "duration": round(duration, 2),
        "frames": frame_count,
        "avg_fps": round(sum(fps_history) / len(fps_history), 2),
        "avg_inf": round((sum(inf_history) / len(inf_history)) * 1000, 2),
        "avg_cpu": round(sum(cpu_history) / len(cpu_history), 2),
        "avg_ram": round(sum(ram_history) / len(ram_history), 2),
        "avg_temp": round(sum(temp_history) / len(temp_history), 2),
        "avg_conf": round(sum(conf_history) / len(conf_history), 3) if conf_history else 0,
	"classes": dict(class_totals),
    })
    return run_id

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", required=True)
    parser.add_argument("--benchmark", default="web")
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--confidence", type=float, required=True)
    parser.add_argument("--preview", action="store_true")

    args = parser.parse_args()

    run_benchmark(
        device_name=args.device,
        benchmark_name=args.benchmark,
        model_path=Path(args.model),
        video_path=Path(args.video),
        imgsz=args.imgsz,
        confidence=args.confidence,
        preview=args.preview,
    )