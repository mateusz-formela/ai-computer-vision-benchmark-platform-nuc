import cv2
import time
import psutil
import signal
import torch
import requests
from ultralytics import YOLO
from influxdb import InfluxDBClient
from collections import Counter

# =========================
# KONFIG
# =========================
CAMERA_URL = "http://192.168.1.5:81/stream"

INFLUX_HOST = "localhost"
INFLUX_PORT = 8086
INFLUX_DB = "yolo"

STATUS_URL = "http://127.0.0.1:5000/status"

RECONNECT_DELAY = 2

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
# STATUS JOULE
# =========================
def joule_active():
    try:
        r = requests.get(STATUS_URL, timeout=0.3)
        return r.json().get("joule") == True
    except:
        return False

# =========================
# INFLUX WRITE
# =========================
def save_metrics(detections, person, confidence, fps, cpu, ram, temp):
    json_body = [
        {
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
        }
    ]
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

    cap = None
    prev_time = 0

    while running:

        # 👉 jeśli Joule działa → oddaj kamerę
        if joule_active():
            if cap is not None:
                print("🟡 Joule aktywny → oddaję kamerę")
                cap.release()
                cap = None
                cv2.destroyAllWindows()
            time.sleep(1)
            continue

        # 👉 NUC przejmuje kamerę
        if cap is None:
            print("🟢 NUC przejmuje kamerę")
            cap = cv2.VideoCapture(CAMERA_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print("❌ Nie można otworzyć streama")
            cap = None
            time.sleep(RECONNECT_DELAY)
            continue

        ret, frame = cap.read()
        if not ret:
            print("🔄 Restart streama...")
            cap.release()
            cap = None
            time.sleep(RECONNECT_DELAY)
            continue

        # =========================
        # YOLO - TYLKO PERSON
        # =========================
        results = model(
            frame,
            device=device,
            verbose=False,
            classes=[0]  # person only
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

                # rysowanie boxów
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                cv2.putText(
                    frame,
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

        # średnia precyzja rozpoznania
        avg_confidence = (
            sum(confidences) / len(confidences)
            if confidences else 0
        )

        # =========================
        # FPS
        # =========================
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        # =========================
        # SYSTEM
        # =========================
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        temp = get_cpu_temp()

        # =========================
        # LOG
        # =========================
        print(
            f"NUC | DET:{detections} | {dict(counts)} "
            f"| CONF:{avg_confidence:.2f} "
            f"| FPS:{fps:.2f} | CPU:{cpu}% "
            f"| RAM:{ram}% | TEMP:{temp}°C"
        )

        # =========================
        # INFLUX
        # =========================
        save_metrics(
            detections,
            person_count,
            avg_confidence,
            fps,
            cpu,
            ram,
            temp
        )

        # =========================
        # PODGLĄD
        # =========================
        cv2.imshow("NUC CAMERA", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

    # =========================
    # CLEAN EXIT
    # =========================
    if cap:
        cap.release()

    cv2.destroyAllWindows()
    print("✅ Koniec")


# =========================
# START
# =========================
if __name__ == "__main__":
    main()
