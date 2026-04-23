"""AI Video Post-Production Pipeline - Step-based UI."""

import os
os.environ["PYTHONIOENCODING"] = "utf-8"

# Force UTF-8 for all subprocess calls on Windows
import subprocess
_orig_run = subprocess.run
def _utf8_run(*args, **kwargs):
    if kwargs.get("text") or kwargs.get("universal_newlines"):
        kwargs.setdefault("encoding", "utf-8")
        kwargs.setdefault("errors", "replace")
    return _orig_run(*args, **kwargs)
subprocess.run = _utf8_run

import gradio as gr
from pathlib import Path
from datetime import datetime
from pipeline import run_phase1, run_phase2, create_pipeline
from pipeline.base import Status, run_cmd
from pipeline.transcribe import transcribe_video, polish_transcript, diarize_and_transcribe

WORKSPACE = Path(__file__).parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)

# ─── Options ──────────────────────────────────────────────

WORK_MODES = ["AI 旁白後製", "原音上字幕"]

VOICE_OPTIONS = {
    "曉臻 (zh-TW, 女聲)": "zh-TW-HsiaoChenNeural",
    "曉雨 (zh-TW, 女聲)": "zh-TW-HsiaoYuNeural",
    "雲哲 (zh-TW, 男聲)": "zh-TW-YunJheNeural",
    "曉曉 (zh-CN, 女聲)": "zh-CN-XiaoxiaoNeural",
    "雲希 (zh-CN, 男聲)": "zh-CN-YunxiNeural",
}

TTS_ENGINES = {
    "edge-tts (免費，不需 GPU)": "edge-tts",
    "GPT-SoVITS (聲音複製)": "gpt-sovits",
}

MUSIC_ENGINES = {
    "靜音佔位（配樂引擎尚未安裝）": "silence",
    "ACE-Step 1.5 (本地開源，需安裝)": "ace-step",
}

NARRATION_MODES = ["AI 自動生成", "手動輸入", "略過旁白"]

LLM_PROVIDERS = {
    "Ollama 本地模型（免費，離線）": "ollama",
    "Gemini 雲端（需 API Key）": "gemini",
}

LANGUAGES = ["zh (中文)", "en (英文)", "ja (日文)", "ko (韓文)"]

DEFAULT_OLLAMA_URL = "http://localhost:11434"


# ─── Tool status ──────────────────────────────────────────

def _check_all_deps() -> dict:
    import shutil
    checks = {}
    checks["ffmpeg"] = (shutil.which("ffmpeg") is not None, "影片處理")
    try:
        import edge_tts  # noqa: F401
        checks["edge-tts"] = (True, "語音合成")
    except ImportError:
        checks["edge-tts"] = (False, "pip install edge-tts")
    try:
        import demucs  # noqa: F401
        checks["demucs"] = (True, "去人聲")
    except ImportError:
        checks["demucs"] = (False, "選裝")
    try:
        import faster_whisper  # noqa: F401
        checks["Whisper"] = (True, "語音辨識")
    except ImportError:
        checks["Whisper"] = (False, "pip install faster-whisper")
    try:
        import urllib.request, json as _j
        with urllib.request.urlopen(f"{DEFAULT_OLLAMA_URL}/api/tags", timeout=2) as r:
            models = [m["name"] for m in _j.loads(r.read()).get("models", [])]
            checks["Ollama"] = (True, ", ".join(models))
    except Exception:
        checks["Ollama"] = (False, "ollama serve")
    return checks


def get_tools_html() -> str:
    html = '<div style="display:flex; gap:12px; flex-wrap:wrap; padding:8px 0;">'
    for tool, (ok, msg) in _check_all_deps().items():
        color = "#4caf50" if ok else "#888"
        dot = "●" if ok else "○"
        html += f'<span style="font-size:12px; color:{color};">{dot} {tool}: {msg}</span>'
    html += '</div>'
    return html


def get_pipeline_html(steps_data: list[dict]) -> str:
    html = '<div style="display:flex; flex-wrap:wrap; gap:4px; align-items:center; padding:8px 0;">'
    for i, s in enumerate(steps_data):
        st = s.get("status", "pending")
        colors = {"pending": "#555", "running": "#4da6ff", "done": "#4caf50",
                  "error": "#f44336", "skipped": "#ffc107"}
        icons = {"pending": "○", "running": "►", "done": "●", "error": "✕", "skipped": "◎"}
        c = colors.get(st, "#555")
        ico = icons.get(st, "○")
        dur = f" ({s['duration']})" if s.get("duration") else ""
        html += f'<span style="font-size:12px; color:{c}; white-space:nowrap;">{ico} {s["name"]}{dur}</span>'
        if i < len(steps_data) - 1:
            html += '<span style="color:#444; font-size:10px;">→</span>'
    html += '</div>'
    return html


def make_steps_data() -> list[dict]:
    return [{"name": s.name, "desc": s.description, "status": "pending", "duration": ""}
            for s in create_pipeline()]


def make_subtitle_steps() -> list[dict]:
    return [
        {"name": "語音辨識", "desc": "Whisper ASR + OpenCC", "status": "pending", "duration": ""},
        {"name": "燒入字幕", "desc": "FFmpeg 合成", "status": "pending", "duration": ""},
    ]


# ─── AI Narration Mode: Phase handlers ────────────────────

def run_p1(video_file, remove_vocals, narration_mode, narration_text,
           llm_prov_label, ollama_model, ollama_url, llm_api_key, narration_style):

    if video_file is None:
        return get_pipeline_html(make_steps_data()), "[!] 請先上傳影片", "", None, make_steps_data()

    llm_provider = LLM_PROVIDERS.get(llm_prov_label, "ollama")
    mode_map = {"AI 自動生成": "ai", "手動輸入": "manual", "略過旁白": "skip"}

    steps_data = make_steps_data()
    log_lines = []

    def on_start(i, step):
        steps_data[i]["status"] = "running"
        log_lines.append(f"\n{'='*40}\n{step.name}\n{'='*40}")

    def on_done(i, step, result):
        m = {Status.DONE: "done", Status.ERROR: "error", Status.SKIPPED: "skipped"}
        steps_data[i]["status"] = m.get(result.status, "error")
        steps_data[i]["duration"] = f"{result.duration:.1f}s"
        for l in step.log_lines:
            log_lines.append(f"  {l}")
        log_lines.append(f"  >> {result.message}")

    steps, ctx = run_phase1(
        source_video=video_file, workspace=str(WORKSPACE),
        remove_vocals=remove_vocals,
        narration_mode=mode_map.get(narration_mode, "skip"),
        narration_script=narration_text or "",
        narration_style=narration_style or "",
        llm_provider=llm_provider,
        llm_api_key=llm_api_key or "",
        ollama_url=ollama_url or DEFAULT_OLLAMA_URL,
        ollama_model=ollama_model or "gemma4:26b",
        on_step_start=on_start, on_step_done=on_done,
    )

    has_error = any(s["status"] == "error" for s in steps_data)
    if has_error:
        log_lines.append("\n[!] Phase 1 有步驟失敗")
    else:
        log_lines.append("\n--- Phase 1 完成，請到 Step 2 審閱旁白 ---")

    return (get_pipeline_html(steps_data), "\n".join(log_lines),
            ctx.get("narration_text", ""), ctx, steps_data)


def run_p2(ctx, edited_narration, tts_engine_label, voice_label, voice_sample,
           sovits_url, music_engine_label, music_prompt, music_duration, bgm_volume,
           prev_steps_data):

    if ctx is None:
        return get_pipeline_html(make_steps_data()), "[!] 請先執行 Step 1", None

    ctx["narration_text"] = edited_narration or ""
    ctx["tts_engine"] = TTS_ENGINES.get(tts_engine_label, "edge-tts")
    ctx["tts_voice"] = VOICE_OPTIONS.get(voice_label, "zh-TW-HsiaoChenNeural")
    ctx["voice_sample"] = voice_sample
    ctx["sovits_url"] = sovits_url or "http://127.0.0.1:9880"
    ctx["music_engine"] = MUSIC_ENGINES.get(music_engine_label, "silence")
    ctx["music_prompt"] = music_prompt or "溫柔的戶外自然環境背景音樂"
    ctx["music_duration"] = int(music_duration)
    ctx["bgm_volume"] = float(bgm_volume)

    steps_data = prev_steps_data if prev_steps_data else make_steps_data()
    log_lines = []

    def on_start(i, step):
        steps_data[i]["status"] = "running"
        log_lines.append(f"\n{'='*40}\n{step.name}\n{'='*40}")

    def on_done(i, step, result):
        m = {Status.DONE: "done", Status.ERROR: "error", Status.SKIPPED: "skipped"}
        steps_data[i]["status"] = m.get(result.status, "error")
        steps_data[i]["duration"] = f"{result.duration:.1f}s"
        for l in step.log_lines:
            log_lines.append(f"  {l}")
        log_lines.append(f"  >> {result.message}")

    steps, ctx = run_phase2(ctx, on_step_start=on_start, on_step_done=on_done)

    final = ctx.get("final_output")
    final_path = str(final) if final and Path(final).exists() else None
    done = sum(1 for s in steps_data if s["status"] == "done")
    log_lines.append(f"\n--- 完成：{done}/{len(steps_data)} 步驟 ---")

    return get_pipeline_html(steps_data), "\n".join(log_lines), final_path


# ─── Subtitle Mode: Transcribe + Burn ─────────────────────

def run_subtitle_p1(video_file, sub_lang, whisper_model_label="large-v3 (最精準)"):
    """Phase 1 for subtitle mode: transcribe original audio."""
    if video_file is None:
        return get_pipeline_html(make_subtitle_steps()), "[!] 請先上傳影片", "", None, None, None

    lang_code = sub_lang.split(" ")[0] if sub_lang else "zh"
    # Parse model name from label
    whisper_model = whisper_model_label.split(" ")[0] if whisper_model_label else "large-v3"
    logs = []
    ws = str(WORKSPACE / f"sub_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    steps = make_subtitle_steps()
    steps[0]["status"] = "running"

    import time
    t0 = time.time()
    result = transcribe_video(video_path=video_file, workspace=ws,
                              language=lang_code, whisper_model=whisper_model,
                              on_log=lambda m: logs.append(m))
    dur = time.time() - t0

    if "error" in result:
        steps[0]["status"] = "error"
        steps[0]["duration"] = f"{dur:.1f}s"
        logs.append(f"[ERROR] {result['error']}")
        return (get_pipeline_html(steps), "\n".join(logs), "", None, None, None)

    steps[0]["status"] = "done"
    steps[0]["duration"] = f"{dur:.1f}s"
    logs.append("\n--- 辨識完成，請審閱字幕後點「燒入字幕」 ---")

    # Store context for phase 2
    sub_ctx = {
        "source_video": video_file,
        "srt_file": result["srt_file"],
        "workspace": ws,
    }

    return (get_pipeline_html(steps), "\n".join(logs),
            result["transcript"], sub_ctx, steps,
            result["srt_file"])


def run_polish(transcript, ollama_model, ollama_url):
    """Send transcript to local LLM for cleanup."""
    if not transcript.strip():
        return transcript, "沒有內容可以潤詞"
    try:
        cleaned = polish_transcript(
            transcript,
            ollama_url=ollama_url or DEFAULT_OLLAMA_URL,
            ollama_model=ollama_model or "gemma4:26b",
        )
        return cleaned, "AI 潤詞完成"
    except Exception as e:
        return transcript, f"潤詞失敗：{e}"


def run_diarize_p1(video_file, sub_lang, hf_token):
    """Phase 1 with speaker diarization."""
    if video_file is None:
        return get_pipeline_html(make_subtitle_steps()), "[!] 請先上傳影片", "", None, None, None

    lang_code = sub_lang.split(" ")[0] if sub_lang else "zh"
    logs = []
    ws = str(WORKSPACE / f"dia_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    steps = make_subtitle_steps()
    steps[0]["status"] = "running"

    import time
    t0 = time.time()
    result = diarize_and_transcribe(
        video_path=video_file, workspace=ws,
        language=lang_code, hf_token=hf_token or "",
        on_log=lambda m: logs.append(m),
    )
    dur = time.time() - t0

    if "error" in result:
        steps[0]["status"] = "error"
        steps[0]["duration"] = f"{dur:.1f}s"
        logs.append(f"[ERROR] {result['error']}")
        return (get_pipeline_html(steps), "\n".join(logs), "", None, None, None)

    steps[0]["status"] = "done"
    steps[0]["duration"] = f"{dur:.1f}s"
    speakers = result.get("speakers", [])
    logs.append(f"\n--- 辨識完成（{', '.join(speakers)}），請審閱後點「燒入字幕」 ---")

    sub_ctx = {
        "source_video": video_file,
        "srt_file": result["srt_file"],
        "workspace": ws,
    }
    return (get_pipeline_html(steps), "\n".join(logs),
            result["transcript"], sub_ctx, steps, result["srt_file"])


def run_subtitle_p2(sub_ctx, edited_transcript, prev_steps):
    """Phase 2 for subtitle mode: burn SRT into video."""
    if sub_ctx is None:
        return get_pipeline_html(make_subtitle_steps()), "[!] 請先執行辨識", None, None

    steps = prev_steps if prev_steps else make_subtitle_steps()
    steps[1]["status"] = "running"
    logs = []

    import time
    t0 = time.time()

    src = sub_ctx["source_video"]
    ws = Path(sub_ctx["workspace"])

    # Rebuild SRT from edited transcript (preserves timestamps, updates text)
    orig_srt = sub_ctx.get("srt_file")
    srt_file = ws / "final_subtitle.srt"

    if orig_srt and Path(orig_srt).exists() and edited_transcript.strip():
        import re
        orig_lines = Path(orig_srt).read_text(encoding="utf-8")
        edited_lines = [l.strip() for l in edited_transcript.strip().split("\n") if l.strip()]

        # Parse original SRT blocks (keep timestamps, replace text)
        blocks = re.split(r'\n\n+', orig_lines.strip())
        new_srt = []
        for i, block in enumerate(blocks):
            parts = block.strip().split("\n")
            if len(parts) >= 3:
                idx = parts[0]
                timestamp = parts[1]
                # Use edited text if available, else keep original
                if i < len(edited_lines):
                    # Extract text after timestamp prefix [HH:MM:SS,mmm]
                    edited = edited_lines[i]
                    edited = re.sub(r'^\[[\d:,]+\]\s*', '', edited)
                    new_srt.append(f"{idx}\n{timestamp}\n{edited}")
                else:
                    new_srt.append(block)

        srt_file.write_text("\n\n".join(new_srt) + "\n", encoding="utf-8")
        logs.append(f"已套用編輯後的字幕（{len(new_srt)} 段）")
    elif orig_srt and Path(orig_srt).exists():
        import shutil
        shutil.copy2(orig_srt, srt_file)
    else:
        logs.append("[!] 找不到 SRT 檔案")
        return get_pipeline_html(make_subtitle_steps()), "\n".join(logs), None, None

    out = ws / "output_with_subtitles.mp4"
    sub_path = str(srt_file).replace("\\", "/").replace(":", "\\:")

    logs.append("燒入字幕中...")
    try:
        run_cmd([
            "ffmpeg", "-y", "-i", str(src),
            "-vf", f"subtitles='{sub_path}'",
            "-c:a", "copy",
            str(out)
        ], timeout=600)
        steps[1]["status"] = "done"
        logs.append(f"完成：{out.name}")
    except Exception as e:
        steps[1]["status"] = "error"
        logs.append(f"[ERROR] {e}")
        out = None

    dur = time.time() - t0
    steps[1]["duration"] = f"{dur:.1f}s"

    final_path = str(out) if out and out.exists() else None
    srt_dl = str(srt_file) if srt_file.exists() else None

    return get_pipeline_html(steps), "\n".join(logs), final_path, srt_dl


# ─── Gradio UI ────────────────────────────────────────────

CUSTOM_CSS = """
.step-btn { min-height: 48px !important; }
.log-box textarea { font-family: monospace !important; font-size: 11px !important; }
"""

with gr.Blocks(title="AI 影片自動後製") as app:

    pipeline_ctx = gr.State(None)
    steps_state = gr.State(None)
    sub_ctx = gr.State(None)
    sub_steps_state = gr.State(None)

    gr.Markdown("# AI 影片自動後製工作流")
    tools_bar = gr.HTML(value=get_tools_html())

    # ═══ Work Mode Selector ═══
    work_mode = gr.Radio(choices=WORK_MODES, value="AI 旁白後製",
                         label="工作模式", info="AI 旁白後製 = 全自動剪片配音｜原音上字幕 = 辨識原音燒入字幕")

    # ═══════════════════════════════════════════════════════
    # MODE A: AI 旁白後製
    # ═══════════════════════════════════════════════════════
    with gr.Group(visible=True) as mode_narration:
        pipeline_bar = gr.HTML(value=get_pipeline_html(make_steps_data()))

        with gr.Tabs():
            with gr.Tab("Step 1: 上傳與設定"):
                with gr.Row():
                    with gr.Column(scale=1):
                        a_video = gr.File(label="上傳影片", file_types=["video"])
                        a_remove_vocals = gr.Checkbox(label="移除人聲（保留環境音）", value=False)

                    with gr.Column(scale=1):
                        narration_mode = gr.Radio(
                            choices=NARRATION_MODES, value="AI 自動生成", label="旁白模式")

                        with gr.Group(visible=True) as ai_group:
                            llm_provider = gr.Dropdown(
                                choices=list(LLM_PROVIDERS.keys()),
                                value=list(LLM_PROVIDERS.keys())[0], label="AI 模型")
                            with gr.Row():
                                ollama_model = gr.Dropdown(
                                    choices=["gemma4:26b", "gemma4:e4b"],
                                    value="gemma4:26b", label="模型",
                                    allow_custom_value=True, scale=1)
                                ollama_url = gr.Textbox(
                                    value=DEFAULT_OLLAMA_URL, label="位址", scale=1)
                            llm_api_key = gr.Textbox(
                                label="Gemini API Key", type="password", visible=False)
                            narration_style = gr.Textbox(
                                label="旁白風格（選填）",
                                placeholder="Discovery 紀錄片 / 輕鬆 Vlog / 詩意文藝")

                        manual_input = gr.Textbox(
                            label="旁白文字", lines=4, visible=False,
                            placeholder="在這裡輸入旁白...")

                a1_btn = gr.Button("生成旁白稿 →", variant="primary", size="lg", elem_classes=["step-btn"])
                a1_log = gr.Textbox(label="執行紀錄", lines=6, interactive=False, elem_classes=["log-box"])

            with gr.Tab("Step 2: 審閱旁白"):
                gr.Markdown("**審閱/編輯旁白稿，確認後點「完成製作」**")
                narration_editor = gr.Textbox(
                    label="旁白稿（可編輯）", lines=10, interactive=True,
                    placeholder="Step 1 完成後旁白會出現在這裡...")
                with gr.Row():
                    with gr.Column(scale=1):
                        tts_engine = gr.Dropdown(
                            choices=list(TTS_ENGINES.keys()),
                            value=list(TTS_ENGINES.keys())[0], label="語音引擎")
                        voice_select = gr.Dropdown(
                            choices=list(VOICE_OPTIONS.keys()),
                            value=list(VOICE_OPTIONS.keys())[0], label="語音角色")
                        voice_sample = gr.Audio(
                            label="聲音樣本（錄製 10-30 秒，用於聲音複製）",
                            sources=["microphone", "upload"], type="filepath", visible=False)
                        sovits_url = gr.Textbox(
                            label="GPT-SoVITS 伺服器位址",
                            value="http://127.0.0.1:9880", visible=False)
                    with gr.Column(scale=1):
                        music_engine = gr.Dropdown(
                            choices=list(MUSIC_ENGINES.keys()),
                            value=list(MUSIC_ENGINES.keys())[0], label="配樂引擎")
                        music_prompt = gr.Textbox(label="配樂風格", value="溫柔的戶外自然環境背景音樂")
                        with gr.Row():
                            music_duration = gr.Slider(30, 600, 120, step=10, label="配樂秒數")
                            bgm_volume = gr.Slider(0.0, 1.0, 0.15, step=0.05, label="音量")

                a2_btn = gr.Button("完成製作 →", variant="primary", size="lg", elem_classes=["step-btn"])
                a2_log = gr.Textbox(label="執行紀錄", lines=6, interactive=False, elem_classes=["log-box"])

            with gr.Tab("Step 3: 輸出"):
                a_output = gr.Video(label="最終輸出影片")

    # ═══════════════════════════════════════════════════════
    # MODE B: 原音上字幕
    # ═══════════════════════════════════════════════════════
    with gr.Group(visible=False) as mode_subtitle:
        sub_bar = gr.HTML(value=get_pipeline_html(make_subtitle_steps()))

        with gr.Tabs():
            with gr.Tab("Step 1: 上傳與辨識"):
                with gr.Row():
                    with gr.Column(scale=1):
                        b_video = gr.File(label="上傳影片", file_types=["video"])
                        with gr.Row():
                            b_lang = gr.Dropdown(choices=LANGUAGES, value="zh (中文)", label="語言", scale=1)
                            b_whisper_model = gr.Dropdown(
                                choices=["large-v3 (最精準)", "medium (快 5 倍)", "base (最快)"],
                                value="large-v3 (最精準)", label="Whisper 模型", scale=1)

                        b_diarize = gr.Checkbox(
                            label="辨識說話者（SpeechBrain，免 token）", value=False)

                        with gr.Row():
                            b1_btn = gr.Button("辨識語音 →", variant="primary", size="lg", elem_classes=["step-btn"])

                        b1_log = gr.Textbox(label="辨識紀錄", lines=6, interactive=False, elem_classes=["log-box"])

                    with gr.Column(scale=1):
                        b_transcript = gr.Textbox(label="逐字稿（可編輯）", lines=12, interactive=True)
                        with gr.Row():
                            b_polish_model = gr.Dropdown(
                                choices=["gemma4:26b", "gemma4:e4b"],
                                value="gemma4:26b", label="潤詞模型", scale=1,
                                allow_custom_value=True)
                            b_polish_btn = gr.Button("AI 潤詞", variant="secondary", scale=1)
                        b_polish_log = gr.Textbox(label="", lines=1, interactive=False)
                        b_dl_srt = gr.File(label="下載 SRT")

            with gr.Tab("Step 2: 燒入字幕"):
                b2_btn = gr.Button("燒入字幕 →", variant="primary", size="lg", elem_classes=["step-btn"])
                b2_log = gr.Textbox(label="執行紀錄", lines=6, interactive=False, elem_classes=["log-box"])
                b_output = gr.Video(label="上字幕後的影片")

    # ─── Event wiring ────────────────────────────────────

    # Work mode toggle
    def toggle_mode(mode):
        if mode == "AI 旁白後製":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    work_mode.change(fn=toggle_mode, inputs=[work_mode],
                     outputs=[mode_narration, mode_subtitle])

    # Narration mode toggle
    def toggle_narr(mode):
        if mode == "AI 自動生成":
            return gr.update(visible=True), gr.update(visible=False)
        elif mode == "手動輸入":
            return gr.update(visible=False), gr.update(visible=True)
        return gr.update(visible=False), gr.update(visible=False)

    narration_mode.change(fn=toggle_narr, inputs=[narration_mode],
                          outputs=[ai_group, manual_input])

    # LLM provider toggle
    def toggle_llm(label):
        is_ollama = LLM_PROVIDERS.get(label) == "ollama"
        return (gr.update(visible=is_ollama), gr.update(visible=is_ollama),
                gr.update(visible=not is_ollama))

    llm_provider.change(fn=toggle_llm, inputs=[llm_provider],
                        outputs=[ollama_model, ollama_url, llm_api_key])

    # Voice clone toggle (sample + server URL)
    def toggle_clone(eng):
        is_sovits = "GPT-SoVITS" in eng
        is_clone = "複製" in eng
        return gr.update(visible=is_clone), gr.update(visible=is_sovits)

    tts_engine.change(fn=toggle_clone,
                      inputs=[tts_engine], outputs=[voice_sample, sovits_url])

    # Mode A: Phase 1
    a1_btn.click(
        fn=run_p1,
        inputs=[a_video, a_remove_vocals, narration_mode, manual_input,
                llm_provider, ollama_model, ollama_url, llm_api_key, narration_style],
        outputs=[pipeline_bar, a1_log, narration_editor, pipeline_ctx, steps_state],
    )

    # Mode A: Phase 2
    a2_btn.click(
        fn=run_p2,
        inputs=[pipeline_ctx, narration_editor, tts_engine, voice_select, voice_sample,
                sovits_url, music_engine, music_prompt, music_duration, bgm_volume, steps_state],
        outputs=[pipeline_bar, a2_log, a_output],
    )

    # Mode B: Phase 1 (transcribe, with optional diarization)
    def handle_b1(video, lang, whisper_model, do_diarize):
        if do_diarize:
            return run_diarize_p1(video, lang, "")
        return run_subtitle_p1(video, lang, whisper_model)

    b1_btn.click(
        fn=handle_b1,
        inputs=[b_video, b_lang, b_whisper_model, b_diarize],
        outputs=[sub_bar, b1_log, b_transcript, sub_ctx, sub_steps_state, b_dl_srt],
    )

    # Mode B: AI 潤詞
    b_polish_btn.click(
        fn=run_polish,
        inputs=[b_transcript, b_polish_model, ollama_url],
        outputs=[b_transcript, b_polish_log],
    )

    # Mode B: Phase 2 (burn subtitles)
    b2_btn.click(
        fn=run_subtitle_p2,
        inputs=[sub_ctx, b_transcript, sub_steps_state],
        outputs=[sub_bar, b2_log, b_output, b_dl_srt],
    )


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860,
               max_file_size=5 * 1024 * 1024 * 1024,
               theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
               css=CUSTOM_CSS)
