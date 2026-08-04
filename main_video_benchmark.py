import cv2
import time
import psutil
import signal
import torch
from ultralytics import YOLO
from influxdb import InfluxDBClient
from collections import Counter

VIDEO_PATH = "./12208077_720_1280_60fps.mp4"
MODEL_NAME = "yolo26n.pt"
DEVICE_NAME = "nuc"
SHOW_PREVIEW = True

INFLUX_HOST = "localhost"
INFLUX_PORT = 8086
INFLUX_DB = "yolo"

client = InfluxDBClient(host=INFLUX_HOST, port=INFLUX_PORT)
client.switch_database(INFLUX_DB)

running = True

def signal_handler(sig, frame):
    global running
    print("\n🛑 Stop (CTRL+C)")
    running = False

signal.signal(signal.SIGINT, signal_handler)

def get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        if "coretemp" in temps:
            for entry in temps["coretemp"]:
                if "Package" in entry.label:
                    return entry.current
    except Exception:
        pass
    return 0

def save_metrics(frame, detections, person, confidence,
                 fps, inference_time, cpu, ram, temp):
    json_body = [{
        "measurement": "yolo_metrics",
        "tags": {
            "device": DEVICE_NAME,
            "model": MODEL_NAME.replace(".pt", "")
        },
        "fields": {
            "frame": int(frame),
            "detections": int(detections),
            "person": int(person),
            "confidence": float(confidence),
            "fps": float(fps),
            "inference_time": float(inference_time),
            "cpu": float(cpu),
            "ram": float(ram),
            "temp": float(temp)
        }
    }]
    try:
        client.write_points(json_body)
    except Exception as e:
        print("❌ Influx error:", e)

def main():
    global running

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {device}")
    print(f"🤖 Model : {MODEL_NAME}")
    print(f"📹 Video : {VIDEO_PATH}")

    model = YOLO(MODEL_NAME)
    model.to(device)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"❌ Nie można otworzyć pliku: {VIDEO_PATH}")
        return

    frame_count = 0
    total_start = time.time()

    while running:
        ret, frame = cap.read()
        if not ret:
            print("\n✅ Koniec filmu")
            break

        start = time.time()

        display = frame.copy()

        results = model(
            display,
            device=device,
            verbose=False,
            classes=[0],
            conf=0.15,
            imgsz=1280
        )[0]

        labels = []
        confidences = []

        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                label = results.names[cls]
                conf = float(box.conf[0])

                labels.append(label)
                confidences.append(conf)

                if SHOW_PREVIEW:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(display, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(
                        display,
                        f"{label} {conf:.2f}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,0),
                        2
                    )

        counts = Counter(labels)

        detections = len(labels)
        person_count = counts.get("person", 0)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        inference_time = time.time() - start
        fps = 1.0 / inference_time

        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        temp = get_cpu_temp()

        print(
            f"{MODEL_NAME.replace('.pt','')} | "
            f"Frame:{frame_count:05d} | "
            f"DET:{detections} | "
            f"CONF:{avg_confidence:.2f} | "
            f"FPS:{fps:.2f} | "
            f"INF:{inference_time*1000:.1f} ms | "
            f"CPU:{cpu:.1f}% | "
            f"RAM:{ram:.1f}% | "
            f"TEMP:{temp:.1f}°C"
        )

        save_metrics(
            frame_count,
            detections,
            person_count,
            avg_confidence,
            fps,
            inference_time,
            cpu,
            ram,
            temp
        )

        frame_count += 1

        if SHOW_PREVIEW:
            cv2.imshow("YOLO Benchmark", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    total_time = time.time() - total_start

    print("\n==============================")
    print("      PODSUMOWANIE")
    print("==============================")
    print(f"Model                : {MODEL_NAME}")
    print(f"Urządzenie           : {DEVICE_NAME}")
    print(f"Przetworzonych klatek: {frame_count}")
    print(f"Czas                 : {total_time:.2f} s")
    print(f"Średni FPS           : {frame_count / total_time:.2f}")
    print("==============================")

    cap.release()

    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
