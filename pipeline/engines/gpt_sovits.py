"""GPT-SoVITS voice cloning engine.

Calls GPT-SoVITS API server (default port 9880).
The GPT-SoVITS server runs as a separate process.

Setup: https://github.com/RVC-Boss/GPT-SoVITS
Start: python api_v2.py (in GPT-SoVITS directory)
"""

import urllib.request
import urllib.error
import json
from pathlib import Path


DEFAULT_URL = "http://127.0.0.1:9880"


def synthesize(
    text: str,
    output_path: Path,
    speaker_wav: str,
    base_url: str = DEFAULT_URL,
    language: str = "zh",
    speed: float = 1.0,
    ref_text: str = "",
) -> Path:
    """Call GPT-SoVITS API to generate speech."""
    payload = json.dumps({
        "text": text,
        "text_lang": language,
        "ref_audio_path": speaker_wav,
        "prompt_text": ref_text,
        "prompt_lang": language,
        "media_type": "wav",
        "streaming_mode": False,
        "speed_factor": speed,
        "batch_size": 1,
        "temperature": 0.6,
        "top_k": 15,
        "top_p": 1.0,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/tts",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GPT-SoVITS {e.code}: {body}")

    return output_path


def health_check(base_url: str = DEFAULT_URL) -> bool:
    """Check if GPT-SoVITS API is reachable via TCP connect."""
    import socket
    try:
        url = base_url.rstrip("/").replace("http://", "").replace("https://", "")
        host, port = url.split(":")
        sock = socket.create_connection((host, int(port)), timeout=3)
        sock.close()
        return True
    except Exception:
        return False
