"""AI Video Post-Production Pipeline - Step-based UI."""

import gradio as gr
from pathlib import Path
from pipeline import run_phase1, run_phase2, create_pipeline
from pipeline.base import Status
from pipeline.transcribe import transcribe_video

WORKSPACE = Path(__file__).parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)

# ─── Options ──────────────────────────────────────────────

VOICE_OPTIONS = {
    "曉臻 (zh-TW, 女聲)": "zh-TW-HsiaoChenNeural",
    "曉雨 (zh-TW, 女聲)": "zh-TW-HsiaoYuNeural",
    "雲哲 (zh-TW, 男聲)": "zh-TW-YunJheNeural",
    "曉曉 (zh-CN, 女聲)": "zh-CN-XiaoxiaoNeural",
    "雲希 (zh-CN, 男聲)": "zh-CN-YunxiNeural",
}

TTS_ENGINES = {
    "edge-tts (免費，不需 GPU)": "edge-tts",
    "Fish-Speech S2 (聲音複製)": "fish-speech",
    "GPT-SoVITS (聲音複製)": "gpt-sovits",
    "CosyVoice (聲音複製)": "cosyvoice",
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


# ─── Pipeline status ─────────────────────────────────────

def make_steps_data() -> list[dict]:
    return [
        {"name": s.name, "desc": s.description, "status": "pending", "duration": ""}
        for s in create_pipeline()
    ]


def get_pipeline_html(steps_data: list[dict]) -> str:
    html = '<div style="display:flex; flex-wrap:wrap; gap:4px; align-items:center; padding:8px 0;">'
    for i, s in enumerate(steps_data):
        st = s.get("status", "pending")
        colors = {
            "pending": "#555", "running": "#4da6ff",
            "done": "#4caf50", "error": "#f44336", "skipped": "#ffc107",
        }
        c = colors.get(st, "#555")
        icons = {"pending": "○", "running": "►", "done": "●", "error": "✕", "skipped": "◎"}
        ico = icons.get(st, "○")
        dur = f" ({s['duration']})" if s.get("duration") else ""

        html += f'<span style="font-size:12px; color:{c}; white-space:nowrap;">{ico} {s["name"]}{dur}</span>'
        if i < len(steps_data) - 1:
            html += '<span style="color:#444; font-size:10px;">→</span>'
    html += '</div>'
    return html


# ─── Phase handlers ───────────────────────────────────────

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
        log_lines.append("\n--- Phase 1 完成，請到「Step 2」審閱旁白 ---")

    return (
        get_pipeline_html(steps_data),
        "\n".join(log_lines),
        ctx.get("narration_text", ""),
        ctx,
        steps_data,
    )


def run_p2(ctx, edited_narration, tts_engine_label, voice_label, voice_sample,
           music_engine_label, music_prompt, music_duration, bgm_volume, prev_steps_data):

    if ctx is None:
        return get_pipeline_html(make_steps_data()), "[!] 請先執行 Step 1", None

    ctx["narration_text"] = edited_narration or ""
    ctx["tts_engine"] = TTS_ENGINES.get(tts_engine_label, "edge-tts")
    ctx["tts_voice"] = VOICE_OPTIONS.get(voice_label, "zh-TW-HsiaoChenNeural")
    ctx["voice_sample"] = voice_sample
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


def run_transcribe(video_file, lang_label):
    if video_file is None:
        return "[!] 請先上傳影片", "", gr.update(visible=False), gr.update(visible=False)

    lang_code = lang_label.split(" ")[0] if lang_label else "zh"
    logs = []
    from datetime import datetime
    ws = str(WORKSPACE / f"tr_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    result = transcribe_video(video_path=video_file, workspace=ws,
                              language=lang_code, on_log=lambda m: logs.append(m))
    log_text = "\n".join(logs)
    if "error" in result:
        return f"{log_text}\n[ERROR] {result['error']}", "", \
               gr.update(visible=False), gr.update(visible=False)

    return (log_text, result["transcript"],
            gr.update(value=result["transcript_file"], visible=True),
            gr.update(value=result["srt_file"], visible=True))


# ─── Gradio UI ────────────────────────────────────────────

CUSTOM_CSS = """
.step-btn { min-height: 48px !important; }
.log-box textarea { font-family: monospace !important; font-size: 11px !important; }
"""

with gr.Blocks(title="AI 影片自動後製") as app:

    # State
    pipeline_ctx = gr.State(None)
    steps_state = gr.State(None)

    gr.Markdown("# AI 影片自動後製工作流")

    # Pipeline status bar (always visible)
    pipeline_bar = gr.HTML(value=get_pipeline_html(make_steps_data()))
    tools_bar = gr.HTML(value=get_tools_html())

    # Shared log (always visible at bottom via render order)
    with gr.Tabs() as main_tabs:

        # ═══ Tab 1: Upload & Configure ═══
        with gr.Tab("Step 1: 上傳與設定"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.File(label="上傳影片（MP4/MOV/MKV/AVI）", file_types=["video"])
                    remove_vocals = gr.Checkbox(label="移除人聲（保留環境音）", value=False)

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

            p1_btn = gr.Button("生成旁白稿 →", variant="primary", size="lg", elem_classes=["step-btn"])
            p1_log = gr.Textbox(label="執行紀錄", lines=8, interactive=False, elem_classes=["log-box"])

        # ═══ Tab 2: Review Narration ═══
        with gr.Tab("Step 2: 審閱旁白"):
            gr.Markdown("**審閱並編輯 AI 生成的旁白稿，確認後進入下一步。**")
            narration_editor = gr.Textbox(
                label="旁白稿（可直接編輯）", lines=12, interactive=True,
                placeholder="Step 1 完成後旁白內容會出現在這裡...")

            with gr.Row():
                with gr.Column(scale=1):
                    tts_engine = gr.Dropdown(
                        choices=list(TTS_ENGINES.keys()),
                        value=list(TTS_ENGINES.keys())[0], label="語音引擎")
                    voice_select = gr.Dropdown(
                        choices=list(VOICE_OPTIONS.keys()),
                        value=list(VOICE_OPTIONS.keys())[0], label="語音角色")
                    voice_sample = gr.Audio(
                        label="聲音樣本（錄製或上傳，用於聲音複製）",
                        sources=["microphone", "upload"], type="filepath", visible=False)

                with gr.Column(scale=1):
                    music_engine = gr.Dropdown(
                        choices=list(MUSIC_ENGINES.keys()),
                        value=list(MUSIC_ENGINES.keys())[0], label="配樂引擎")
                    music_prompt = gr.Textbox(
                        label="配樂風格", value="溫柔的戶外自然環境背景音樂")
                    with gr.Row():
                        music_duration = gr.Slider(30, 600, 120, step=10, label="配樂秒數")
                        bgm_volume = gr.Slider(0.0, 1.0, 0.15, step=0.05, label="音量")

            p2_btn = gr.Button("完成製作 →", variant="primary", size="lg", elem_classes=["step-btn"])
            p2_log = gr.Textbox(label="執行紀錄", lines=8, interactive=False, elem_classes=["log-box"])

        # ═══ Tab 3: Output ═══
        with gr.Tab("Step 3: 輸出"):
            output_video = gr.Video(label="最終輸出影片")

        # ═══ Tab 4: Transcription ═══
        with gr.Tab("逐字稿"):
            gr.Markdown("上傳訪談/會議影片 → AI 辨識語音 → 帶時間戳逐字稿 + SRT")
            with gr.Row():
                with gr.Column(scale=1):
                    tr_video = gr.File(label="上傳影片", file_types=["video"])
                    tr_lang = gr.Dropdown(
                        choices=["zh (中文)", "en (英文)", "ja (日文)", "ko (韓文)"],
                        value="zh (中文)", label="語言")
                    tr_btn = gr.Button("開始辨識", variant="primary", size="lg")
                    tr_log = gr.Textbox(label="辨識紀錄", lines=5, interactive=False)
                with gr.Column(scale=1):
                    tr_output = gr.Textbox(label="逐字稿", lines=14, interactive=True)
                    with gr.Row():
                        tr_dl_txt = gr.File(label="下載 TXT", visible=False)
                        tr_dl_srt = gr.File(label="下載 SRT", visible=False)

    # ─── Event wiring ────────────────────────────────────

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

    # Voice sample toggle
    def toggle_vsample(eng):
        return gr.update(visible="複製" in eng)

    tts_engine.change(fn=toggle_vsample, inputs=[tts_engine], outputs=[voice_sample])

    # Phase 1
    p1_btn.click(
        fn=run_p1,
        inputs=[video_input, remove_vocals, narration_mode, manual_input,
                llm_provider, ollama_model, ollama_url, llm_api_key, narration_style],
        outputs=[pipeline_bar, p1_log, narration_editor, pipeline_ctx, steps_state],
    )

    # Phase 2
    p2_btn.click(
        fn=run_p2,
        inputs=[pipeline_ctx, narration_editor, tts_engine, voice_select, voice_sample,
                music_engine, music_prompt, music_duration, bgm_volume, steps_state],
        outputs=[pipeline_bar, p2_log, output_video],
    )

    # Transcription
    tr_btn.click(
        fn=run_transcribe,
        inputs=[tr_video, tr_lang],
        outputs=[tr_log, tr_output, tr_dl_txt, tr_dl_srt],
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860,
               theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
               css=CUSTOM_CSS)
