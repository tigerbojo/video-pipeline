"""Standalone transcription: video -> Traditional Chinese transcript + SRT."""

from pathlib import Path
from .base import cmd_exists, run_cmd


def transcribe_video(
    video_path: str,
    workspace: str,
    language: str = "zh",
    on_log=None,
) -> dict:
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    video = Path(video_path)

    def log(msg):
        if on_log:
            on_log(msg)

    # Step 1: Extract full audio
    audio_file = ws / "audio.wav"
    log("擷取完整音軌中...")
    if not cmd_exists("ffmpeg"):
        return {"error": "需要安裝 ffmpeg"}

    try:
        run_cmd([
            "ffmpeg", "-y", "-i", str(video),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_file)
        ], timeout=300)
    except Exception as e:
        return {"error": f"音軌擷取失敗：{e}"}

    if not audio_file.exists() or audio_file.stat().st_size < 1000:
        return {"error": "影片可能沒有音軌"}

    file_size_mb = audio_file.stat().st_size / 1024 / 1024
    log(f"音軌大小：{file_size_mb:.1f} MB")

    # Step 2: Transcribe
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return {"error": "需要安裝：pip install faster-whisper"}

    # Use medium model on CPU for speed, large-v3 if GPU available
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False

    model_name = "large-v3" if has_gpu else "medium"
    compute = "float16" if has_gpu else "int8"
    log(f"載入 Whisper {model_name} 模型（{'GPU' if has_gpu else 'CPU'}, {compute}）...")

    try:
        model = WhisperModel(model_name, compute_type=compute)
    except Exception as e:
        log(f"{model_name} 載入失敗，嘗試 base 模型...")
        model = WhisperModel("base", compute_type="int8")
        model_name = "base"

    log(f"開始辨識（語言：{language}）...")

    segments_gen, info = model.transcribe(
        str(audio_file),
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=800,
            speech_pad_ms=400,
            threshold=0.4,
        ),
        condition_on_previous_text=True,
    )

    # Consume all segments
    raw_segments = []
    for seg in segments_gen:
        raw_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })

    log(f"原始辨識：{len(raw_segments)} 段")

    if not raw_segments:
        return {"error": "未偵測到語音內容"}

    # Step 3: Merge short segments (< 2 seconds or < 5 chars)
    merged = _merge_short_segments(raw_segments, min_duration=3.0, min_chars=8)
    log(f"合併後：{len(merged)} 段")

    # Step 4: OpenCC simplified -> traditional Chinese (s2twp = 簡體到台灣繁體+台灣用語)
    try:
        import opencc
        cc = opencc.OpenCC("s2twp")
        for seg in merged:
            seg["text"] = cc.convert(seg["text"])
        log("OpenCC 簡體→繁體轉換完成")
    except ImportError:
        log("OpenCC 未安裝，跳過繁體轉換（pip install opencc-python-reimplemented）")

    # Step 5: Build output
    transcript_lines = []
    srt_lines = []
    for i, seg in enumerate(merged, 1):
        start = _fmt(seg["start"])
        end = _fmt(seg["end"])
        transcript_lines.append(f"[{start}] {seg['text']}")
        srt_lines.append(f"{i}\n{start} --> {end}\n{seg['text']}\n")

    transcript_text = "\n".join(transcript_lines)
    srt_text = "\n".join(srt_lines)

    transcript_file = ws / "transcript.txt"
    srt_file = ws / "transcript.srt"
    transcript_file.write_text(transcript_text, encoding="utf-8")
    srt_file.write_text(srt_text, encoding="utf-8")

    log(f"逐字稿已儲存（{len(merged)} 段，模型：{model_name}）")

    return {
        "transcript": transcript_text,
        "srt": srt_text,
        "transcript_file": str(transcript_file),
        "srt_file": str(srt_file),
    }


def _merge_short_segments(segments: list[dict], min_duration: float = 3.0, min_chars: int = 8) -> list[dict]:
    """Merge segments that are too short into neighbors."""
    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        prev = merged[-1]
        prev_duration = prev["end"] - prev["start"]
        gap = seg["start"] - prev["end"]

        # Merge if: previous is short, or gap is tiny, or text is very short
        should_merge = (
            prev_duration < min_duration
            or len(prev["text"]) < min_chars
            or gap < 0.5
        )

        if should_merge:
            prev["end"] = seg["end"]
            prev["text"] = prev["text"] + seg["text"]
        else:
            merged.append(seg.copy())

    # Second pass: merge any remaining short trailing segments
    if len(merged) > 1 and (merged[-1]["end"] - merged[-1]["start"]) < 1.0:
        last = merged.pop()
        merged[-1]["end"] = last["end"]
        merged[-1]["text"] += last["text"]

    return merged


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
