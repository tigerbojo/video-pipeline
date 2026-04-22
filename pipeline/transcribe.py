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

    # Get audio duration
    try:
        dur_str = run_cmd([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_file)
        ]).strip()
        audio_duration = float(dur_str)
        log(f"音軌長度：{audio_duration:.1f} 秒")
    except Exception:
        audio_duration = 0

    log(f"開始辨識（語言：{language}）...")

    # No VAD filter - let Whisper handle all audio to avoid missing speech
    segments_gen, info = model.transcribe(
        str(audio_file),
        language=language,
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=True,
        no_speech_threshold=0.5,
        log_prob_threshold=-0.5,
    )

    raw_segments = []
    for seg in segments_gen:
        text = seg.text.strip()
        if text and seg.no_speech_prob < 0.7:
            raw_segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": text,
            })

    log(f"原始辨識：{len(raw_segments)} 段（音軌 {audio_duration:.0f} 秒）")

    if not raw_segments:
        return {"error": "未偵測到語音內容"}

    # Step 3: Merge short segments (< 2 seconds or < 5 chars)
    merged = _merge_short_segments(raw_segments)
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


def _merge_short_segments(segments: list[dict], min_duration: float = 2.0, min_chars: int = 6, max_duration: float = 15.0) -> list[dict]:
    """Merge segments that are too short (by time or text length)."""
    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        prev = merged[-1]
        prev_dur = prev["end"] - prev["start"]
        gap = seg["start"] - prev["end"]
        merged_dur = seg["end"] - prev["start"]

        # Merge if previous is short (time or text) and won't exceed max
        too_short = prev_dur < min_duration or len(prev["text"]) < min_chars
        can_merge = gap < 1.5 and merged_dur <= max_duration

        if too_short and can_merge:
            prev["end"] = seg["end"]
            prev["text"] = prev["text"] + "，" + seg["text"]
        else:
            merged.append(seg.copy())

    return merged


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _estimate_num_speakers(affinity, max_speakers=6):
    """Estimate number of speakers using eigengap heuristic."""
    import numpy as np
    from scipy import linalg

    # Compute normalized Laplacian eigenvalues
    D = np.diag(affinity.sum(axis=1))
    L = D - affinity
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(affinity.sum(axis=1), 1e-8)))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    eigenvalues = np.sort(np.real(linalg.eigvalsh(L_norm)))
    # Find largest gap between consecutive eigenvalues
    gaps = np.diff(eigenvalues[:max_speakers + 1])
    n_speakers = np.argmax(gaps) + 1
    return max(2, min(n_speakers, max_speakers))


# ─── AI 潤詞 ─────────────────────────────────────────────

def polish_transcript(
    transcript: str,
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "gemma4:26b",
) -> str:
    """Use local LLM to clean up filler words and polish transcript."""
    import json
    import urllib.request

    prompt = (
        "你是逐字稿編輯助手。請清理以下逐字稿：\n\n"
        "1. 刪除所有贅詞和語助詞：嗯、啊、呃、那個、就是、然後、對啊、對對對、欸、蛤、噢、哦、好\n"
        "2. 刪除嘆氣聲、笑聲描述、重複的字詞\n"
        "3. 修正明顯的語音辨識錯字\n"
        "4. 加上適當的標點符號（逗號、句號、問號）\n"
        "5. 保持 [HH:MM:SS,mmm] 時間戳格式不變\n"
        "6. 保持原始語意，不要改寫或摘要\n"
        "7. 只輸出清理後的逐字稿，不要加任何說明\n\n"
        f"原始逐字稿：\n{transcript}"
    )

    payload = json.dumps({
        "model": ollama_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 4096},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    return result.get("message", {}).get("content", transcript).strip()


# ─── 說話者辨識（SpeechBrain，免 token）─────────────────────

def diarize_and_transcribe(
    video_path: str,
    workspace: str,
    language: str = "zh",
    hf_token: str = "",
    on_log=None,
) -> dict:
    """Transcribe with speaker diarization using SpeechBrain + Whisper."""
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    video = Path(video_path)

    def log(msg):
        if on_log:
            on_log(msg)

    # Extract audio
    audio_file = ws / "audio.wav"
    log("擷取音軌中...")
    run_cmd([
        "ffmpeg", "-y", "-i", str(video),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_file)
    ], timeout=300)

    # Step 1: Speaker diarization with SpeechBrain
    try:
        import torchaudio
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        return {"error": "需要安裝：pip install speechbrain torchaudio"}

    log("載入 SpeechBrain 說話者嵌入模型（免 token）...")
    try:
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(ws / "sb_model"),
        )
    except Exception as e:
        return {"error": f"SpeechBrain 模型載入失敗：{e}"}

    # Load audio and split into windows for embedding extraction
    log("分析說話者特徵中...")
    signal, sr = torchaudio.load(str(audio_file))
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)
        sr = 16000

    # Extract embeddings per 1.5s window with 0.75s overlap
    window_sec = 1.5
    hop_sec = 0.75
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    total_samples = signal.shape[1]

    embeddings = []
    timestamps = []
    for start_sample in range(0, total_samples - window_samples, hop_samples):
        chunk = signal[:, start_sample:start_sample + window_samples]
        emb = classifier.encode_batch(chunk).squeeze().detach().cpu()
        embeddings.append(emb)
        timestamps.append(start_sample / sr)

    if not embeddings:
        return {"error": "音軌太短，無法辨識說話者"}

    emb_tensor = torch.stack(embeddings)
    log(f"擷取 {len(embeddings)} 個語音片段的特徵向量")

    # Spectral clustering to identify speakers
    from sklearn.cluster import SpectralClustering
    import numpy as np

    emb_np = emb_tensor.numpy()
    # Cosine similarity matrix
    norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
    emb_normed = emb_np / (norms + 1e-8)
    similarity = emb_normed @ emb_normed.T
    # Convert to affinity (0 to 1)
    affinity = (similarity + 1) / 2
    np.fill_diagonal(affinity, 1.0)

    # Auto-detect number of speakers (2-6 range)
    n_speakers = _estimate_num_speakers(affinity, max_speakers=6)
    log(f"預估 {n_speakers} 位說話者")

    clustering = SpectralClustering(
        n_clusters=n_speakers, affinity="precomputed", random_state=42,
    ).fit(affinity)

    # Build speaker timeline from clustering
    speaker_segments = []
    for i, label in enumerate(clustering.labels_):
        speaker_segments.append({
            "start": timestamps[i],
            "end": timestamps[i] + window_sec,
            "speaker": f"SPEAKER_{label}",
        })
    log(f"偵測到 {len(set(s['speaker'] for s in speaker_segments))} 位說話者")

    # Step 2: Whisper transcription
    from faster_whisper import WhisperModel
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False

    model_name = "large-v3" if has_gpu else "medium"
    compute = "float16" if has_gpu else "int8"
    log(f"載入 Whisper {model_name}...")
    model = WhisperModel(model_name, compute_type=compute)

    log("語音辨識中...")
    segments_gen, _ = model.transcribe(
        str(audio_file), language=language,
        beam_size=5, vad_filter=False,
        condition_on_previous_text=True,
        no_speech_threshold=0.5,
    )

    whisper_segments = []
    for seg in segments_gen:
        text = seg.text.strip()
        if text and seg.no_speech_prob < 0.7:
            whisper_segments.append({
                "start": seg.start, "end": seg.end, "text": text,
            })

    # Step 3: Assign speakers to Whisper segments
    def find_speaker(seg_start, seg_end):
        best_speaker = "?"
        best_overlap = 0
        for sp in speaker_segments:
            overlap = min(seg_end, sp["end"]) - max(seg_start, sp["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sp["speaker"]
        return best_speaker

    # Map speaker IDs to friendly names
    speaker_ids = sorted(set(s["speaker"] for s in speaker_segments))
    speaker_names = {sid: f"說話者{chr(65 + i)}" for i, sid in enumerate(speaker_ids)}

    for seg in whisper_segments:
        raw_speaker = find_speaker(seg["start"], seg["end"])
        seg["speaker"] = speaker_names.get(raw_speaker, "?")

    # OpenCC
    try:
        import opencc
        cc = opencc.OpenCC("s2twp")
        for seg in whisper_segments:
            seg["text"] = cc.convert(seg["text"])
        log("OpenCC 簡體→繁體完成")
    except ImportError:
        pass

    # Build output
    transcript_lines = []
    srt_lines = []
    for i, seg in enumerate(whisper_segments, 1):
        start = _fmt(seg["start"])
        end = _fmt(seg["end"])
        transcript_lines.append(f"[{start}] [{seg['speaker']}] {seg['text']}")
        srt_lines.append(f"{i}\n{start} --> {end}\n[{seg['speaker']}] {seg['text']}\n")

    transcript_text = "\n".join(transcript_lines)
    srt_text = "\n".join(srt_lines)

    transcript_file = ws / "transcript_diarized.txt"
    srt_file = ws / "transcript_diarized.srt"
    transcript_file.write_text(transcript_text, encoding="utf-8")
    srt_file.write_text(srt_text, encoding="utf-8")

    log(f"完成：{len(whisper_segments)} 段，{len(speaker_ids)} 位說話者")

    return {
        "transcript": transcript_text,
        "srt": srt_text,
        "transcript_file": str(transcript_file),
        "srt_file": str(srt_file),
        "speakers": list(speaker_names.values()),
    }
