from flask import Flask, request, jsonify
from influxdb import InfluxDBClient
import time

app = Flask(__name__)

# ===== INFLUX =====
INFLUX_HOST = "localhost"
INFLUX_PORT = 8086
INFLUX_DB = "yolo"

client = InfluxDBClient(
    host=INFLUX_HOST,
    port=INFLUX_PORT
)
client.switch_database(INFLUX_DB)

# ===== HEARTBEAT =====
last_heartbeat = 0

LIVE_STATUS = {
    "running": False
}


# ===== HEARTBEAT ENDPOINT =====
@app.route("/heartbeat", methods=["POST"])
def heartbeat():
    global last_heartbeat
    last_heartbeat = time.time()
    return jsonify({"status": "ok"})


# ===== METRICS ENDPOINT =====
@app.route("/metrics", methods=["POST"])
def metrics():

    data = request.json
    print("📡 JOULE:", data)

    device = data.get("device", "unknown")

    json_body = [
        {
            "measurement": "yolo_metrics",

            "tags": {
                "device": device
            },

            "fields": {
                "cpu": float(data.get("cpu", 0)),
                "ram": float(data.get("ram", 0)),
                "temp": float(data.get("temp", 0)),
                "fps": float(data.get("fps", 0)),
                "detections": int(data.get("detections", 0)),
                "confidence": float(data.get("confidence", 0.0)),
                "person_detected": int(
                    data.get("detections", 0) > 0
                )
            }
        }
    ]

    try:
        client.write_points(json_body)

    except Exception as e:
        print("❌ Influx error:", e)

    return jsonify({"status": "ok"})
# ===== LIVE STATUS =====
@app.route("/live_status", methods=["GET", "POST"])
def live_status():
    global LIVE_STATUS

    if request.method == "POST":

        data = request.json or {}

        LIVE_STATUS.clear()
        LIVE_STATUS.update(data)
        LIVE_STATUS["last_update"] = time.time()

        return jsonify({"status": "ok"})

    return jsonify(LIVE_STATUS)


# ===== STATUS (czy Joule żyje) =====
@app.route("/status", methods=["GET"])
def status():

    active = (
        time.time() - last_heartbeat
    ) < 5

    return jsonify({
        "joule": active
    })


# ===== START =====
if __name__ == "__main__":
    print("🚀 Flask + Influx start")

    app.run(
        host="0.0.0.0",
        port=5000
    )
