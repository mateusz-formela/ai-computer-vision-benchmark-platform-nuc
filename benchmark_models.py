import cv2,time,psutil,signal,torch
from ultralytics import YOLO
from influxdb import InfluxDBClient
from datetime import datetime

VIDEO_PATH="./12208077_720_1280_60fps.mp4"
MODEL_NAME="yolo26n.pt"
DEVICE_NAME="nuc"
IMG_SIZE=640
CONFIDENCE=0.25
SHOW_PREVIEW=False

client=InfluxDBClient(host="localhost",port=8086)
client.switch_database("yolo")

running=True
signal.signal(signal.SIGINT, lambda s,f: globals().__setitem__("running",False))

def temp():
    try:
        t=psutil.sensors_temperatures()
        if "coretemp" in t:
            for e in t["coretemp"]:
                if "Package" in e.label:
                    return e.current
    except:
        pass
    return 0

# ============================================
# START BENCHMARK
# ============================================

start_datetime = datetime.now()

print("="*60)
print("YOLO BENCHMARK")
print("="*60)
print("Start :", start_datetime.strftime("%Y-%m-%d %H:%M:%S"))
print("="*60)

m=YOLO(MODEL_NAME)
m.to("cuda" if torch.cuda.is_available() else "cpu")

cap=cv2.VideoCapture(VIDEO_PATH)

f=0
t0=time.time()

while running:

    ok,img=cap.read()

    if not ok:
        break

    s=time.time()

    r=m(
        img,
        classes=[0],
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
        verbose=False
    )[0]

    det=len(r.boxes) if r.boxes is not None else 0
    person=det

    conf=sum(float(b.conf[0]) for b in r.boxes)/det if det else 0

    inf=time.time()-s
    fps=1/inf

    client.write_points([{

      "measurement":"yolo_metrics",

      "tags":{
          "device":DEVICE_NAME,
          "model":MODEL_NAME.replace(".pt","")
      },

      "fields":{

          "frame":f,
          "detections":det,
          "person":person,
          "confidence":conf,

          "fps":fps,
          "inference_time":inf,

          "cpu":psutil.cpu_percent(None),
          "ram":psutil.virtual_memory().percent,
          "temp":temp()

      }

    }])

    print(
        f"{MODEL_NAME} | "
        f"Frame:{f:05d} | "
        f"DET:{det} | "
        f"FPS:{fps:.2f}"
    )

    if SHOW_PREVIEW:

        cv2.imshow("YOLO",r.plot())

        if cv2.waitKey(1)&0xFF==ord("q"):
            break

    f+=1

# ============================================
# END BENCHMARK
# ============================================

end_datetime = datetime.now()

duration = time.time()-t0

print("\n"+"="*60)
print("BENCHMARK SUMMARY")
print("="*60)

print("Start    :", start_datetime.strftime("%Y-%m-%d %H:%M:%S"))
print("End      :", end_datetime.strftime("%Y-%m-%d %H:%M:%S"))
print(f"Duration : {duration:.2f} s")

print(f"Frames   : {f}")
print(f"Avg FPS  : {f/duration:.2f}")

print("="*60)

cap.release()

if SHOW_PREVIEW:
    cv2.destroyAllWindows()