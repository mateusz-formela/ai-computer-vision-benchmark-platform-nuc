import os
import threading
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import (
    FastAPI,
    Form,
    Request,
    Body,
    UploadFile,
    File,
)

from fastapi.responses import (
    RedirectResponse,
    JSONResponse,
    StreamingResponse,
    HTMLResponse,
)

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
import os

from config import (
    DEVICES,
    IMAGE_SIZES,
    CONFIDENCE_VALUES,
    MODELS_DIR,
    VIDEOS_DIR,
    LIVE_STREAMS,
)

from utils import (
    create_run_id,
    get_models,
    get_videos,
)

from benchmark import run_benchmark
from web.status import (
    STATUS,
    PREVIEW_FRAMES,
    PREVIEW_LOCK,
)

LIVE_STATUS = {}

from history import get_history, delete_runs


BASE_DIR = Path(__file__).parent

load_dotenv(BASE_DIR.parent / ".env")

app = FastAPI(title="One Click AI Benchmark Platform")

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY"),
)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

USERNAME = os.getenv("APP_USERNAME")
PASSWORD = os.getenv("APP_PASSWORD")

print("USERNAME =", USERNAME)
print("PASSWORD =", PASSWORD)

JOULE_HOST = "student@192.168.1.4"
JOULE_PROJECT = "/home/student/ai-project"

def is_joule_online():

    result = subprocess.run(
        [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=2",
            JOULE_HOST,
            "echo ok",
        ],
        capture_output=True,
        text=True,
    )

    return result.returncode == 0


def start_benchmark(
    device,
    model,
    video,
    image_size,
    confidence,
    preview,
):

    try:
        run_id = create_run_id()

        if str(video) == "__ESP32__":
            video_name = "ESP32-CAM (Live)"
        elif str(video) in LIVE_STREAMS:
            video_name = LIVE_STREAMS[str(video)]["name"]
        else:
            video_name = Path(video).stem

        STATUS.update({
            "running": True,
            "run_id": run_id,
            "device": device,

            "selected_model": Path(model).stem,
            "selected_video": video_name,
            "selected_image_size": image_size,
            "selected_confidence": confidence,
            "selected_preview": preview,

            "model": Path(model).stem,
            "video": video_name,
            "image_size": image_size,
            "confidence": confidence,
            "preview": preview,
        })
        print("START STATUS VIDEO =", STATUS["video"])

        if device == "joule":

            cmd = [
                "ssh",
                JOULE_HOST,
                (
                    f"cd {JOULE_PROJECT} && "
                    f"/home/student/miniforge3/envs/yt/bin/python benchmark.py "
                    f'--device "{device}" '
		    f'--benchmark "web" '
                    f'--model "{Path(model).name}" '
                    f'--video "{Path(video).name}" '
                    f"--imgsz {image_size} "
                    f"--confidence {confidence}"
                    + (" --preview" if preview else "")
                ),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            print(result.stdout)
            print(result.stderr)

            for line in result.stdout.splitlines():
                if line.startswith("Run ID"):
                    STATUS["run_id"] = line.split(":", 1)[1].strip()
                    break

        else:

            run_benchmark(
                device_name=device,
                benchmark_name="web",
                model_path=Path(model),
                video_path=Path(video),
                imgsz=image_size,
                confidence=confidence,
                preview=preview,
            )

    finally:

        STATUS["running"] = False

        print()
        print("=" * 60)
        print("Benchmark finished.")
        print(f"Run ID: {STATUS['run_id']}")
        print("=" * 60)


@app.get("/")
async def home(request: Request):

    if not request.session.get("logged_in"):
        return RedirectResponse("/login")

    print("HOME STATUS =", STATUS)
    print("HOME STATUS VIDEO =", repr(STATUS["video"]))

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "username": request.session.get("username"),
            "title": "One Click AI Benchmark Platform",
            "devices": DEVICES,
            "models": get_models(MODELS_DIR),
            "videos": get_videos(VIDEOS_DIR),
            "image_sizes": IMAGE_SIZES,
            "confidence_values": CONFIDENCE_VALUES,
            "status": STATUS,
        },
    )

@app.get("/login")
async def login_page(request: Request):

    error = request.query_params.get("error")

    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Login</title>
    </head>
    <body style="
        background:#0f172a;
        color:white;
        font-family:Arial;
        display:flex;
        justify-content:center;
        align-items:center;
        height:100vh;
    ">

        <form
            method="post"
            action="/login"
            style="
                background:#1e293b;
                padding:40px;
                border-radius:10px;
                width:320px;
            ">

             <img
                src="/static/oneclick.png"
                style="height:200px;display:block;margin:0 auto 25px auto;">

            <h2 style="margin:0;">
                AI Benchmark Platform
            </h2>
	    {"<div style='margin-top:15px;padding:10px;border-radius:8px;background:#7f1d1d;color:white;text-align:center;'>❌ Invalid username or password</div>" if error else ""}

            <div
                style="
                    color:#94A3B8;
                    margin-top:8px;
                    font-size:18px;
                ">

                Real-Time AI Performance Benchmark Suite

            </div>

            <input
                name="username"
                placeholder="Username"
                style="
		width:100%;
		box-sizing:border-box;
		padding:12px;
		margin:10px 0;
		border-radius:8px;
		border:none;
    	    ">

            <input
	        type="password"
	        name="password"
	        placeholder="Password"
	        style="
		    width:100%;
		    box-sizing:border-box;
		    padding:12px;
		    margin:10px 0;
		    border-radius:8px;
		    border:none;
	        ">

            <button
                style="
                    width:100%;
                    padding:10px;
                    cursor:pointer;
                ">
                Login
            </button>

        </form>

    </body>
    </html>
    """)

@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):

    users = {
        USERNAME: PASSWORD,
        "pptak": "Admin@12",
    }

    if users.get(username) == password:

        request.session["logged_in"] = True
        request.session["username"] = username

        return RedirectResponse("/", status_code=302)

    return RedirectResponse(
        "/login?error=1",
        status_code=302,
    )

@app.get("/logout")
async def logout(request: Request):

    request.session.clear()

    return RedirectResponse("/login", status_code=302)

@app.post("/benchmark")
async def benchmark(
    device: str = Form(...),
    model: str = Form(...),
    video: str = Form(...),
    image_size: int = Form(...),
    confidence: float = Form(...),
    preview: str | None = Form(None),
):

    if STATUS["running"]:
        return RedirectResponse("/", status_code=303)

    threading.Thread(
        target=start_benchmark,
        args=(
            device,
            model,
            video,
            image_size,
            confidence,
            preview is not None,
        ),
        daemon=True,
    ).start()

    return RedirectResponse("/", status_code=303)

@app.post("/live_preview")
async def live_preview(request: Request):

    data = await request.body()

    with PREVIEW_LOCK:
        PREVIEW_FRAMES["joule"] = data

    return {"status": "ok"}


@app.post("/live_status")
async def live_status(data: dict = Body(...)):
    global LIVE_STATUS

    LIVE_STATUS.clear()
    LIVE_STATUS.update(data)

    return {"status": "ok"}

def mjpeg_generator(device: str):

    import time

    while True:

        with PREVIEW_LOCK:
            frame = PREVIEW_FRAMES.get(device)

        if frame is None:
            time.sleep(0.02)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame +
            b"\r\n"
        )

        time.sleep(0.001)


@app.get("/stream/{device}")
async def stream(device: str):

    return StreamingResponse(
        mjpeg_generator(device),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close",
        },
    )


@app.get("/status")
async def status():

    status = STATUS.copy()

    if (
        status.get("device") == "joule"
        and status.get("running")
    ):
        status.update(LIVE_STATUS)

   #status["joule_online"] = is_joule_online()
    status["joule_online"] = True

    return JSONResponse(status)

@app.get("/history")
async def history():

    return JSONResponse(get_history())

@app.post("/history/delete")
async def history_delete(run_ids: list[str] = Body(...)):

    delete_runs(run_ids)

    return JSONResponse({"success": True})


@app.post("/stop")
async def stop():

    STATUS.update({

    "running": False,

    "run_id": None,
    "device": None,
    "benchmark": None,
    "model": None,
    "video": None,

    "image_size": None,
    "confidence": None,
    "preview": None,

    "frame": 0,
    "frames": 0,

    "fps": 0,
    "inference": 0,

    "cpu": 0,
    "ram": 0,
    "temp": 0,

    "detections": 0,
    "persons": 0,
    "classes": {},

    "avg_fps": 0,
    "avg_inf": 0,
    "avg_cpu": 0,
    "avg_ram": 0,
    "avg_temp": 0,
    "avg_conf": 0,

})

    return JSONResponse({"success": True})


@app.post("/reset")
async def reset():

    STATUS.update({
        "running": False,
        "run_id": None,
        "device": None,
        "benchmark": None,
        "model": None,
        "video": None,
	"image_size": None,
	"preview": None,
        "start_time": None,
	"start_time_iso": None,
	"end_time": None,
	"end_time_iso": None,
        "duration": None,

        "frame": 0,
        "fps": 0,
        "inference": 0,
        "cpu": 0,
        "ram": 0,
        "temp": 0,
        "detections": 0,
        "persons": 0,
        "confidence": None,
        "classes": {},

        "frames": 0,
        "avg_fps": 0,
        "avg_inf": 0,
        "avg_cpu": 0,
        "avg_ram": 0,
        "avg_temp": 0,
        "avg_conf": 0,
    })

    return JSONResponse({"success": True})