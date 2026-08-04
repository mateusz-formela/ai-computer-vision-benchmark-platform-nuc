"""
YOLO Benchmark Tool
influx_writer.py
"""

from influxdb import InfluxDBClient

from config import (
    INFLUX_HOST,
    INFLUX_PORT,
    INFLUX_DATABASE
)


# ==========================================================
# CLIENT
# ==========================================================

client = InfluxDBClient(
    host=INFLUX_HOST,
    port=INFLUX_PORT
)

client.switch_database(INFLUX_DATABASE)


# ==========================================================
# FRAME METRICS
# ==========================================================

def save_metrics(
    run_id,
    device,
    benchmark,
    model,
    video,
    imgsz,
    confidence,
    frame,
    detections,
    persons,
    avg_conf,
    fps,
    inference_time,
    cpu,
    ram,
    temp,
    class_counts,
):

    fields = {

        "frame": int(frame),

        "detections": int(detections),

        "person": int(persons),

        "confidence": float(avg_conf),

        "fps": float(fps),

        "inference_time": float(inference_time),

        "cpu": float(cpu),

        "ram": float(ram),

        "temp": float(temp),

    }

    for cls, cnt in class_counts.items():

        safe = cls.replace(" ", "_").replace("-", "_")

        fields[f"class_{safe}"] = int(cnt)

    json_body = [{

        "measurement": "yolo_metrics",

        "tags": {

            "run_id": run_id,

            "device": device,

            "benchmark": benchmark,

            "model": model.replace(".pt", ""),

            "video": video,

            "imgsz": str(imgsz),

            "confidence": str(confidence),

        },

        "fields": fields,

    }]


    print("=" * 80)
    print(fields)
    print("=" * 80)
    client.write_points(json_body)


# ==========================================================
# SUMMARY
# ==========================================================

def save_summary(
    run_id,
    device,
    benchmark,
    model,
    video,
    imgsz,
    confidence,
    preview,
    frames,
    duration,
    fps_history,
    inf_history,
    cpu_history,
    ram_history,
    temp_history,
    conf_history,
    total_detections,
    total_persons,
    status,
    start_time,
    end_time,
    start_ts,
    end_ts
):

    json_body = [{

        "measurement": "yolo_summary",

        "tags": {

            "run_id": run_id,

            "device": device,

            "benchmark": benchmark,

            "model": model.replace(".pt", ""),

            "video": video,

            "imgsz": str(imgsz),

            "confidence": str(confidence),

            "status": status

        },

        "fields": {

            "start_time": start_time,

            "end_time": end_time,

	    "start_ts": start_ts,

	    "end_ts": end_ts,

            "frames": frames,

            "duration": duration,

            "avg_fps": sum(fps_history) / len(fps_history),

            "min_fps": min(fps_history),

            "max_fps": max(fps_history),

            "avg_inf": sum(inf_history) / len(inf_history),

            "avg_cpu": sum(cpu_history) / len(cpu_history),

            "avg_ram": sum(ram_history) / len(ram_history),

            "avg_temp": sum(temp_history) / len(temp_history),

            "avg_conf": (
                sum(conf_history) / len(conf_history)
                if conf_history else 0
            ),

            "detections": total_detections,

            "persons": total_persons,

            "preview": preview

        }

    }]

    client.write_points(json_body)

# ==========================================================
# CLASS METRICS
# ==========================================================

def save_class_metrics(
    run_id,
    device,
    benchmark,
    model,
    video,
    imgsz,
    confidence,
    frame,
    class_counts,
):

    json_body = []

    for cls, cnt in class_counts.items():

        json_body.append({

            "measurement": "yolo_classes",

            "tags": {

                "run_id": run_id,

                "device": device,

                "benchmark": benchmark,

                "model": model.replace(".pt", ""),

                "video": video,

                "imgsz": str(imgsz),

                "confidence": str(confidence),

                "class": cls,

            },

            "fields": {

                "count": int(cnt),

                "frame": int(frame),

            }

        })

    if json_body:
        client.write_points(json_body)