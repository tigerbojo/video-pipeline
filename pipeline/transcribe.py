"""Standalone transcription: video -> text transcript + SRT."""

from pathlib import Path
from .base import cmd_exists, run_cmd


def transcribe_video(
    video_path: str,
    workspace: str,
    language: str = "zh",
    on_log=None,
) -> dict:
    """Transcribe a video file and return transcript + SRT content."""
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    video = Path(video_path)

    def log(msg):
        if on_log:
            on_log(msg)

    # Step 1: Extract full audio
    audio_file = ws / "transcribe_audio.wav"
    log("擷取完整音軌中...")
    if not cmd_exists("ffmpeg"):
        return {"error": "需要安裝 ffmpeg"}

    try:
        run_cmd([
            "ffmpeg", "-y", "-i", str(video),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-map", "0:a:0",
            str(audio_file)
        ], timeout=300)
    except Exception as e:
        return {"error": f"音軌擷取失敗：{e}"}

    # Verify audio was extracted
    if not audio_file.exists() or audio_file.stat().st_size < 1000:
        return {"error": "影片可能沒有音軌，或音軌擷取失敗"}

    # Step 2: Transcribe with faster-whisper or whisper
    transcript_lines = []
    srt_lines = []

    try:
        from faster_whisper import WhisperModel
        log("載入 faster-whisper large-v3 模型...")
        model = WhisperModel("large-v3", compute_type="auto")
        log(f"開始辨識（語言：{language}）...")

        # Consume all segments first (generator is lazy)
        all_segments = list(model.transcribe(
            str(audio_file),
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=300,
            ),
            condition_on_previous_text=True,
            word_timestamps=False,
        )[0])

        log(f"辨識中...共偵測到 {len(all_segments)} 段語音")

        for i, seg in enumerate(all_segments, 1):
            start = _fmt(seg.start)
            end = _fmt(seg.end)
            text = seg.text.strip()
            if text:
                transcript_lines.append(f"[{start}] {text}")
                srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")

        log(f"辨識完成：{len(transcript_lines)} 段")

    except ImportError:
        # Fallback: use whisper CLI
        if cmd_exists("whisper"):
            log("使用 whisper CLI 辨識...")
            run_cmd([
                "whisper", str(audio_file),
                "--model", "large-v3",
                "--language", language,
                "--output_format", "srt",
                "--output_dir", str(ws),
            ])
            srt_file = ws / (audio_file.stem + ".srt")
            if srt_file.exists():
                srt_content = srt_file.read_text(encoding="utf-8")
                # Parse SRT to transcript
                import re
                blocks = re.split(r'\n\n+', srt_content.strip())
                for block in blocks:
                    lines = block.strip().split('\n')
                    if len(lines) >= 3:
                        time_line = lines[1]
                        text = ' '.join(lines[2:])
                        start = time_line.split(' --> ')[0].strip()
                        transcript_lines.append(f"[{start}] {text}")
                        srt_lines.append(block + "\n")
                log(f"辨識完成：{len(transcript_lines)} 段")
            else:
                return {"error": "whisper 未產生輸出檔案"}
        else:
            return {
                "error": "需要安裝語音辨識引擎：\npip install faster-whisper\n或\npip install openai-whisper"
            }

    # Step 3: Save outputs
    transcript_text = "\n".join(transcript_lines)
    srt_text = "\n".join(srt_lines)

    transcript_file = ws / "transcript.txt"
    srt_file = ws / "transcript.srt"
    transcript_file.write_text(transcript_text, encoding="utf-8")
    srt_file.write_text(srt_text, encoding="utf-8")

    log(f"逐字稿已儲存：{transcript_file.name}")
    log(f"SRT 字幕已儲存：{srt_file.name}")

    return {
        "transcript": transcript_text,
        "srt": srt_text,
        "transcript_file": str(transcript_file),
        "srt_file": str(srt_file),
    }


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
