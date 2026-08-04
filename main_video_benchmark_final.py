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
VIDEO_PATH = "./street_720p60.mp4"

MODEL_NAME = "yolo26n.pt"      # np. yolov8n.pt, yolo11n.pt, yolo26n.pt
DEVICE_NAME = "nuc"            # później zmień na "joule"
BENCHMARK_NAME = "models"

VIDEO_NAME = "street_720p60"
IMGSZ = 640                 # benchmark modeli = 640
CONFIDENCE = 0.75              # benchmark modeli = 0.25

SHOW_PREVIEW = True

INFLUX_HOST = "localhost"
INFLUX_PORT = 8086
INFLUX_DB = "yolo"

client = InfluxDBClient(host=INFLUX_HOST, port=INFLUX_PORT)
client.switch_database(INFLUX_DB)

running = True

def signal_handler(sig, frame):
    global running
    print("\n🛑 Stop")
    running = False

signal.signal(signal.SIGINT, signal_handler)

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

def save_metrics(frame,detections,person,confidence,fps,inference_time,cpu,ram,temp):
    json_body=[{
        "measurement":"yolo_metrics",
        "tags":{
            "device":DEVICE_NAME,
            "model":MODEL_NAME.replace(".pt",""),
            "benchmark": BENCHMARK_NAME,

            "video": VIDEO_NAME,
            "imgsz": str(IMGSZ),
            "confidence": str(CONFIDENCE),
        },
        "fields":{
            "frame":int(frame),
            "detections":int(detections),
            "person":int(person),
            "confidence":float(confidence),
            "fps":float(fps),
            "inference_time":float(inference_time),
            "cpu":float(cpu),
            "ram":float(ram),
            "temp":float(temp)
        }
    }]
    try:
        client.write_points(json_body)
    except Exception as e:
        print("Influx:",e)

device="cuda" if torch.cuda.is_available() else "cpu"
print("Device:",device)
print("Model :",MODEL_NAME)

model=YOLO(MODEL_NAME)
model.to(device)

cap=cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit("Nie można otworzyć filmu")

frame_count=0
t0=time.time()

while running:
    ret,frame=cap.read()
    if not ret:
        break

    start=time.time()
    display=frame.copy()

    results=model(
        display,
        device=device,
        verbose=False,
        classes=[0],
        imgsz=IMGSZ,
        conf=CONFIDENCE
    )[0]

    labels=[]
    confs=[]

    if results.boxes is not None:
        for box in results.boxes:
            cls=int(box.cls[0])
            label=results.names[cls]
            c=float(box.conf[0])

            labels.append(label)
            confs.append(c)

            if SHOW_PREVIEW:
                x1,y1,x2,y2=map(int,box.xyxy[0])
                cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(display,f"{label} {c:.2f}",(x1,y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    counts=Counter(labels)
    detections=len(labels)
    person_count=counts.get("person",0)
    avg_conf=sum(confs)/len(confs) if confs else 0

    inference_time=time.time()-start
    fps=1.0/inference_time

    cpu=psutil.cpu_percent(None)
    ram=psutil.virtual_memory().percent
    temp=get_cpu_temp()

    print(f"{MODEL_NAME.replace('.pt','')} | Frame:{frame_count:05d} | DET:{detections} | "
          f"CONF:{avg_conf:.2f} | FPS:{fps:.2f} | INF:{inference_time*1000:.1f}ms | "
          f"CPU:{cpu:.1f}% | RAM:{ram:.1f}% | TEMP:{temp:.1f}°C")

    save_metrics(frame_count,detections,person_count,avg_conf,
                 fps,inference_time,cpu,ram,temp)

    frame_count+=1

    if SHOW_PREVIEW:
        cv2.imshow("YOLO Benchmark",display)
        if cv2.waitKey(1)&0xFF==ord("q"):
            break

total=time.time()-t0
print(f"\nFrames: {frame_count}")
print(f"Czas: {total:.2f}s")
print(f"Średni FPS: {frame_count/total:.2f}")

cap.release()
if SHOW_PREVIEW:
    cv2.destroyAllWindows()
