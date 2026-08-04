import cv2
import time
import psutil
import signal
import torch
from ultralytics import YOLO
from influxdb import InfluxDBClient
from collections import Counter

# =========================
# KONFIG
# =========================
VIDEO_PATH = "test.mp4"

INFLUX_HOST = "localhost"
INFLUX_PORT = 8086
INFLUX_DB = "yolo"

# =========================
# INFLUX
# =========================
client = InfluxDBClient(host=INFLUX_HOST, port=INFLUX_PORT)
client.switch_database(INFLUX_DB)

# =========================
# CTRL+C
# =========================
running = True

def signal_handler(sig, frame):
    global running
    print("\n🛑 Stop (CTRL+C)")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# =========================
# TEMP
# =========================
def get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        if "coretemp" in temps:
            for entry in temps["coretemp"]:
                if "Package" in entry.label:
                    return entry.current
    except:
        pass
    return 0

# =========================
# INFLUX WRITE
# =========================
def save_metrics(detections, person, confidence, fps, cpu, ram, temp):
    json_body = [{
        "measurement": "yolo_metrics",
        "tags": {
            "device": "nuc"
        },
        "fields": {
            "detections": int(detections),
            "person": int(person),
            "confidence": float(confidence),
            "fps": float(fps),
            "cpu": float(cpu),
            "ram": float(ram),
            "temp": float(temp)
        }
    }]

    try:
        client.write_points(json_body)
    except Exception as e:
        print("❌ Influx error:", e)

# =========================
# MAIN
# =========================
def main():

    global running

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {device}")

    model = YOLO("yolov8n.pt")
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
            print("✅ Koniec filmu")
            break

        start = time.time()

        display = frame.copy()

        # =========================
        # YOLO
        # =========================
        results = model(
            display,
            device=device,
            verbose=False,
            classes=[0]
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

        avg_confidence = (
            sum(confidences) / len(confidences)
            if confidences else 0
        )

        inference_time = time.time() - start
        fps = 1.0 / inference_time

        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        temp = get_cpu_temp()

        print(
            f"Frame:{frame_count:05d} | "
            f"DET:{detections} | "
            f"CONF:{avg_confidence:.2f} | "
            f"FPS:{fps:.2f} | "
            f"CPU:{cpu}% | "
            f"RAM:{ram}% | "
            f"TEMP:{temp}°C"
        )

        save_metrics(
            detections,
            person_count,
            avg_confidence,
            fps,
            cpu,
            ram,
            temp
        )

        cv2.imshow("YOLO Benchmark", display)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    total_time = time.time() - total_start

    print("\n========== PODSUMOWANIE ==========")
    print(f"Przetworzonych klatek : {frame_count}")
    print(f"Czas                 : {total_time:.2f} s")
    print(f"Średni FPS           : {frame_count / total_time:.2f}")

    cap.release()
    cv2.destroyAllWindows()

# =========================
# START
# =========================
if __name__ == "__main__":
    main()