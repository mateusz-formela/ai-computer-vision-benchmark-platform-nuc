import cv2
import time
import psutil
import signal
import torch
import uuid

from datetime import datetime
from collections import Counter

from ultralytics import YOLO
from influxdb import InfluxDBClient


# ==========================================================
# KONFIGURACJA
# ==========================================================

# DEVICE
DEVICE_NAME = "nuc"

# BENCHMARK
BENCHMARK_NAME = "models"

# MODEL
MODEL_NAME = "yolo26n.pt"

# VIDEO
VIDEO_PATH = "./street_720p60.mp4"
VIDEO_NAME = "street_720p60"

# YOLO PARAMS
IMGSZ = 640
CONFIDENCE = 0.35

# PREVIEW
SHOW_PREVIEW = True


# ==========================================================
# RUN ID
# ==========================================================

RUN_ID = (
    datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_"
    + uuid.uuid4().hex[:6]
)


# ==========================================================
# INFLUXDB
# ==========================================================

INFLUX_HOST = "localhost"
INFLUX_PORT = 8086
INFLUX_DB = "yolo"


client = InfluxDBClient(
    host=INFLUX_HOST,
    port=INFLUX_PORT
)

client.switch_database(INFLUX_DB)


# ==========================================================
# GLOBAL
# ==========================================================

running = True


def signal_handler(sig, frame):
    global running
    print("\n🛑 Stop")
    running = False


signal.signal(signal.SIGINT, signal_handler)



# ==========================================================
# SYSTEM
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
# INFLUX FRAME METRICS
# ==========================================================

def save_metrics(
        frame,
        detections,
        persons,
        confidence,
        fps,
        inference_time,
        cpu,
        ram,
        temp
):

    json_body = [

        {
            "measurement": "yolo_metrics",

            "tags": {

                "run_id": RUN_ID,

                "device": DEVICE_NAME,

                "model":
                    MODEL_NAME.replace(".pt", ""),

                "benchmark":
                    BENCHMARK_NAME,

                "video":
                    VIDEO_NAME,

                "imgsz":
                    str(IMGSZ),

                "confidence":
                    str(CONFIDENCE)
            },


            "fields": {

                "frame":
                    int(frame),

                "detections":
                    int(detections),

                "person":
                    int(persons),

                "confidence":
                    float(confidence),

                "fps":
                    float(fps),

                "inference_time":
                    float(inference_time),

                "cpu":
                    float(cpu),

                "ram":
                    float(ram),

                "temp":
                    float(temp)

            }
        }

    ]


    try:

        client.write_points(json_body)

    except Exception as e:

        print(
            "Influx error:",
            e
        )



# ==========================================================
# SUMMARY VARIABLES
# ==========================================================

fps_history = []

inf_history = []

cpu_history = []

ram_history = []

temp_history = []

conf_history = []


total_detections = 0

total_persons = 0

# ==========================================================
# START
# ==========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("======================================")
print("YOLO BENCHMARK v1.0")
print("======================================")
print("RUN ID :", RUN_ID)
print("DEVICE :", DEVICE_NAME)
print("MODEL  :", MODEL_NAME)
print("VIDEO  :", VIDEO_NAME)
print("IMG    :", IMGSZ)
print("CONF   :", CONFIDENCE)
print("======================================")


model = YOLO(MODEL_NAME)

model.to(device)



cap = cv2.VideoCapture(VIDEO_PATH)


if not cap.isOpened():

    raise SystemExit(
        "Nie można otworzyć filmu"
    )



frame_count = 0

start_total = time.time()



# ==========================================================
# MAIN LOOP
# ==========================================================

while running:


    ret, frame = cap.read()


    if not ret:

        break



    start = time.time()


    display = frame.copy()



    results = model(

        display,

        device=device,

        verbose=False,

        classes=[0],

        imgsz=IMGSZ,

        conf=CONFIDENCE

    )[0]



    labels = []

    confs = []



    if results.boxes is not None:


        for box in results.boxes:


            cls = int(box.cls[0])

            label = results.names[cls]

            conf = float(box.conf[0])


            labels.append(label)

            confs.append(conf)



            if SHOW_PREVIEW:


                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )


                cv2.rectangle(

                    display,

                    (x1, y1),

                    (x2, y2),

                    (0,255,0),

                    2

                )


                cv2.putText(

                    display,

                    f"{label} {conf:.2f}",

                    (x1, y1-8),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (0,255,0),

                    2

                )



    counts = Counter(labels)



    detections = len(labels)


    persons = counts.get(
        "person",
        0
    )


    avg_conf = (

        sum(confs) / len(confs)

        if confs

        else 0

    )



    inference_time = time.time() - start


    fps = 1.0 / inference_time



    cpu = psutil.cpu_percent(None)


    ram = psutil.virtual_memory().percent


    temp = get_cpu_temp()



    # ==========================
    # STATISTICS
    # ==========================

    fps_history.append(fps)

    inf_history.append(inference_time)

    cpu_history.append(cpu)

    ram_history.append(ram)

    temp_history.append(temp)

    conf_history.append(avg_conf)



    total_detections += detections

    total_persons += persons



    save_metrics(

        frame_count,

        detections,

        persons,

        avg_conf,

        fps,

        inference_time,

        cpu,

        ram,

        temp

    )



    print(

        f"{MODEL_NAME.replace('.pt','')} | "

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



    if SHOW_PREVIEW:


        cv2.imshow(

            "YOLO Benchmark",

            display

        )


        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

# ==========================================================
# SUMMARY SAVE
# ==========================================================

def save_summary():

    duration = time.time() - start_total


    json_body = [

        {
            "measurement": "yolo_summary",

            "tags": {

                "run_id": RUN_ID,

                "device": DEVICE_NAME,

                "model":
                    MODEL_NAME.replace(".pt",""),

                "benchmark":
                    BENCHMARK_NAME,

                "video":
                    VIDEO_NAME,

                "imgsz":
                    str(IMGSZ),

                "confidence":
                    str(CONFIDENCE)
            },


            "fields": {


                "frames":
                    int(frame_count),


                "duration":
                    float(duration),


                "avg_fps":
                    float(
                        sum(fps_history)
                        /
                        len(fps_history)
                    )
                    if fps_history else 0,


                "min_fps":
                    float(min(fps_history))
                    if fps_history else 0,


                "max_fps":
                    float(max(fps_history))
                    if fps_history else 0,


                "avg_inf":
                    float(
                        sum(inf_history)
                        /
                        len(inf_history)
                    )
                    if inf_history else 0,


                "avg_cpu":
                    float(
                        sum(cpu_history)
                        /
                        len(cpu_history)
                    )
                    if cpu_history else 0,


                "avg_ram":
                    float(
                        sum(ram_history)
                        /
                        len(ram_history)
                    )
                    if ram_history else 0,


                "avg_temp":
                    float(
                        sum(temp_history)
                        /
                        len(temp_history)
                    )
                    if temp_history else 0,


                "avg_conf":
                    float(
                        sum(conf_history)
                        /
                        len(conf_history)
                    )
                    if conf_history else 0,


                "detections":
                    int(total_detections),


                "persons":
                    int(total_persons)

            }

        }

    ]


    try:

        client.write_points(json_body)


    except Exception as e:

        print(
            "Summary Influx error:",
            e
        )





# ==========================================================
# FINISH
# ==========================================================

save_summary()



total_time = time.time() - start_total



print("\n")
print("=" * 50)
print("YOLO BENCHMARK SUMMARY")
print("=" * 50)

print(
    "RUN ID       :",
    RUN_ID
)

print(
    "DEVICE       :",
    DEVICE_NAME
)

print(
    "MODEL        :",
    MODEL_NAME
)

print(
    "BENCHMARK    :",
    BENCHMARK_NAME
)

print(
    "VIDEO        :",
    VIDEO_NAME
)

print(
    "IMG SIZE     :",
    IMGSZ
)

print(
    "CONFIDENCE   :",
    CONFIDENCE
)

print(
    "FRAMES       :",
    frame_count
)

print(
    "TIME         :",
    round(total_time,2),
    "s"
)

if fps_history:

    print(
        "AVG FPS      :",
        round(
            sum(fps_history)
            /
            len(fps_history),
            2
        )
    )


    print(
        "MIN FPS      :",
        round(
            min(fps_history),
            2
        )
    )


    print(
        "MAX FPS      :",
        round(
            max(fps_history),
            2
        )
    )


print(
    "DETECTIONS   :",
    total_detections
)

print(
    "PERSONS      :",
    total_persons
)

print("=" * 50)



# ==========================================================
# CLEANUP
# ==========================================================

cap.release()


if SHOW_PREVIEW:

    cv2.destroyAllWindows()