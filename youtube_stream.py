import subprocess

from config import LIVE_STREAMS


def get_stream_url(source: str):

    if source not in LIVE_STREAMS:
        return source

    youtube_url = LIVE_STREAMS[source]["youtube"]

    result = subprocess.run(
        [
            "yt-dlp",
            "-g",
            youtube_url,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout.strip()
