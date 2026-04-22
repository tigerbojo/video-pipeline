"""AI Video Post-Production Pipeline - Visual Workflow UI."""

import gradio as gr
from pathlib import Path
from pipeline import run_phase1, run_phase2, create_pipeline, ALL_STEPS
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
    "ACE-Step 1.5 (本地開源)": "ace-step",
    "靜音佔位（尚未安裝配樂引擎）": "silence",
}

NARRATION_MODES = ["AI 自動生成", "手動輸入", "略過旁白"]

LLM_PROVIDERS = {
    "Ollama 本地模型（免費，離線）": "ollama",
    "Gemini 雲端（需 API Key）": "gemini",
}

DEFAULT_OLLAMA_URL = "http://localhost:11434"


# ─── Status HTML ──────────────────────────────────────────

def get_status_html(steps_data: list[dict]) -> str:
    html = '<div style="display:flex; flex-direction:column; gap:8px; padding:16px;">'
    html += '<div style="font-size:18px; font-weight:700; margin-bottom:8px; color:#e0e0e0;">工作流程</div>'
    html += '<div style="display:flex; flex-wrap:wrap; gap:6px; align-items:center;">'

    for i, s in enumerate(steps_data):
        status = s.get("status", "pending")
        name = s.get("name", f"步驟 {i+1}")
        desc = s.get("desc", "")
        duration = s.get("duration", "")

        colors = {
            "pending":  ("#3a3a4a", "#888", "---"),
            "running":  ("#1a3a5c", "#4da6ff", ">>>"),
            "done":     ("#1a3c1a", "#4caf50", "[v]"),
            "error":    ("#3c1a1a", "#f44336", "[x]"),
            "skipped":  ("#3a3a2a", "#ffc107", "[~]"),
        }
        bg, border_color, icon = colors.get(status, colors["pending"])
        time_str = f'<div style="font-size:10px; color:#999;">{duration}</div>' if duration else ""

        # Highlight phase separator
        phase_border = ""
        if i == 2:  # After narration = phase boundary
            phase_border = "border-right: 3px dashed #555;"

        html += f'''
        <div style="
            background:{bg}; border:2px solid {border_color}; border-radius:10px;
            padding:10px 14px; min-width:115px; text-align:center; {phase_border}
        ">
            <div style="font-size:12px; color:{border_color}; font-weight:700;">{icon}</div>
            <div style="font-size:13px; font-weight:600; color:#e0e0e0; margin:4px 0;">{name}</div>
            <div style="font-size:10px; color:#aaa;">{desc}</div>
            {time_str}
        </div>'''
        if i < len(steps_data) - 1:
            sep = '<span style="font-size:9px;color:#666;">review</span>' if i == 2 else ''
            html += f'<div style="font-size:20px; color:#555; padding:0 2px; text-align:center;">->{sep}</div>'

    html += '</div>'

    # Tool status
    html += '<div style="margin-top:16px; padding:12px; background:#1a1a2e; border-radius:8px;">'
    html += '<div style="font-size:13px; font-weight:600; color:#e0e0e0; margin-bottom:8px;">工具狀態</div>'
    for tool, (ok, msg) in _check_all_deps().items():
        color = "#4caf50" if ok else "#f44336"
        icon = "[v]" if ok else "[x]"
        html += f'<div style="font-size:11px; color:{color}; margin:2px 0;">{icon} {tool}: {msg}</div>'
    html += '</div></div>'
    return html


def _check_all_deps() -> dict:
    import shutil
    checks = {}
    checks["ffmpeg"] = (shutil.which("ffmpeg") is not None, "所有影片處理必備")
    try:
        import edge_tts  # noqa: F401
        checks["edge-tts"] = (True, "免費語音合成")
    except ImportError:
        checks["edge-tts"] = (False, "pip install edge-tts")
    try:
        import demucs  # noqa: F401
        checks["demucs"] = (True, "AI 去人聲")
    except ImportError:
        checks["demucs"] = (False, "pip install demucs（選裝）")
    checks["auto-editor"] = (
        shutil.which("auto-editor") is not None,
        "靜音移除" if shutil.which("auto-editor") else "pip install auto-editor（選裝）"
    )
    try:
        import faster_whisper  # noqa: F401
        checks["faster-whisper"] = (True, "語音辨識字幕")
    except ImportError:
        checks["faster-whisper"] = (False, "pip install faster-whisper（選裝）")
    try:
        import urllib.request, json as _json
        with urllib.request.urlopen(f"{DEFAULT_OLLAMA_URL}/api/tags", timeout=2) as r:
            models = [m["name"] for m in _json.loads(r.read()).get("models", [])]
            checks["Ollama"] = (True, f"本地 AI（{', '.join(models)}）")
    except Exception:
        checks["Ollama"] = (False, "未執行（ollama serve）")
    return checks


def make_initial_steps_data() -> list[dict]:
    return [
        {"name": s.name, "desc": s.description, "status": "pending", "duration": ""}
        for s in create_pipeline()
    ]


# ─── Phase 1: Generate narration ──────────────────────────

def do_phase1(
    video_file, remove_vocals,
    narration_mode, narration_text,
    llm_provider_label, ollama_model, ollama_url, llm_api_key, narration_style,
):
    if video_file is None:
        return (
            get_status_html(make_initial_steps_data()),
            "[!] 請先上傳影片檔案。",
            "", None,
        )

    llm_provider = LLM_PROVIDERS.get(llm_provider_label, "ollama")
    mode_map = {"AI 自動生成": "ai", "手動輸入": "manual", "略過旁白": "skip"}
    narr_mode = mode_map.get(narration_mode, "skip")

    steps_data = make_initial_steps_data()
    log_text = ""

    def on_start(i, step):
        nonlocal log_text
        steps_data[i]["status"] = "running"
        log_text += f"\n{'='*50}\n{step.name}\n{'='*50}\n"

    def on_done(i, step, result):
        nonlocal log_text
        status_map = {Status.DONE: "done", Status.ERROR: "error", Status.SKIPPED: "skipped"}
        steps_data[i]["status"] = status_map.get(result.status, "error")
        steps_data[i]["duration"] = f"{result.duration:.1f}s"
        for line in step.log_lines:
            log_text += f"  {line}\n"
        log_text += f"  >> {result.message} ({result.duration:.1f}s)\n"

    steps, ctx = run_phase1(
        source_video=video_file,
        workspace=str(WORKSPACE),
        remove_vocals=remove_vocals,
        narration_mode=narr_mode,
        narration_script=narration_text or "",
        narration_style=narration_style or "",
        llm_provider=llm_provider,
        llm_api_key=llm_api_key or "",
        ollama_url=ollama_url or DEFAULT_OLLAMA_URL,
        ollama_model=ollama_model or "gemma4:26b",
        on_step_start=on_start,
        on_step_done=on_done,
    )

    narration = ctx.get("narration_text", "")
    log_text += f"\n{'='*50}\n--- Phase 1 完成，請審閱旁白稿後點「完成製作」 ---\n"

    return (
        get_status_html(steps_data),
        log_text,
        narration,
        ctx,  # Store in Gradio State for phase 2
    )


# ─── Phase 2: Produce final video ─────────────────────────

def do_phase2(
    ctx, edited_narration,
    tts_engine_label, voice_label, voice_sample,
    music_engine_label, music_prompt, music_duration, bgm_volume,
    steps_data_json,
):
    if ctx is None:
        return (
            get_status_html(make_initial_steps_data()),
            "[!] 請先執行「生成旁白稿」。",
            None,
        )

    # Update context with edited narration and phase 2 settings
    ctx["narration_text"] = edited_narration or ""
    ctx["tts_engine"] = TTS_ENGINES.get(tts_engine_label, "edge-tts")
    ctx["tts_voice"] = VOICE_OPTIONS.get(voice_label, "zh-TW-HsiaoChenNeural")
    ctx["voice_sample"] = voice_sample
    ctx["music_engine"] = MUSIC_ENGINES.get(music_engine_label, "auto")
    ctx["music_prompt"] = music_prompt or "溫柔的戶外自然環境背景音樂"
    ctx["music_duration"] = int(music_duration)
    ctx["bgm_volume"] = float(bgm_volume)

    steps_data = make_initial_steps_data()
    # Restore phase 1 statuses
    if steps_data_json:
        for i in range(min(3, len(steps_data_json), len(steps_data))):
            steps_data[i] = steps_data_json[i]

    log_text = ""

    def on_start(i, step):
        nonlocal log_text
        steps_data[i]["status"] = "running"
        log_text += f"\n{'='*50}\n{step.name}\n{'='*50}\n"

    def on_done(i, step, result):
        nonlocal log_text
        status_map = {Status.DONE: "done", Status.ERROR: "error", Status.SKIPPED: "skipped"}
        steps_data[i]["status"] = status_map.get(result.status, "error")
        steps_data[i]["duration"] = f"{result.duration:.1f}s"
        for line in step.log_lines:
            log_text += f"  {line}\n"
        log_text += f"  >> {result.message} ({result.duration:.1f}s)\n"

    steps, ctx = run_phase2(ctx, on_step_start=on_start, on_step_done=on_done)

    final_video = ctx.get("final_output")
    final_path = str(final_video) if final_video and Path(final_video).exists() else None

    done_count = sum(1 for s in steps_data if s["status"] == "done")
    total = len(steps_data)
    log_text += f"\n{'='*50}\n流程完成：{done_count}/{total} 個步驟已完成\n"

    return (
        get_status_html(steps_data),
        log_text,
        final_path,
    )


# ─── Gradio UI ────────────────────────────────────────────

HEADER_MD = """
# AI 影片自動後製工作流
"""

CUSTOM_CSS = """
.pipeline-flow { min-height: 120px; }
.log-box textarea { font-family: monospace !important; font-size: 12px !important; }
"""

CUSTOM_THEME = gr.themes.Base(primary_hue="blue", neutral_hue="slate")

with gr.Blocks(title="AI 影片自動後製") as app:

    gr.Markdown(HEADER_MD)

    with gr.Tabs():

        # ═══════════════════════════════════════════════
        # TAB 1: 影片後製
        # ═══════════════════════════════════════════════
        with gr.Tab("影片後製"):

            # Hidden state
            pipeline_ctx = gr.State(value=None)
            steps_data_state = gr.State(value=None)

            gr.Markdown("**Phase 1**：上傳素材 -> 去人聲(選) -> 自動粗剪 -> AI 旁白稿 -> **審閱編輯** | **Phase 2**：AI 配音 -> 字幕生成 -> AI 配樂 -> 合成輸出")

            pipeline_html = gr.HTML(
                value=get_status_html(make_initial_steps_data()),
                elem_classes=["pipeline-flow"],
            )

            with gr.Row():
                # ── Left column ──
                with gr.Column(scale=1):

                    # === PHASE 1 settings ===
                    gr.Markdown("### Phase 1 - 素材處理 + 旁白生成")
            video_input = gr.File(
                label="上傳影片（支援 MP4/MOV/MKV/AVI）",
                file_types=["video"],
            )
            remove_vocals = gr.Checkbox(label="移除人聲（保留環境音）", value=False)

            with gr.Accordion("旁白設定", open=True):
                narration_mode = gr.Radio(
                    choices=NARRATION_MODES, value="AI 自動生成", label="旁白模式",
                )
                with gr.Group(visible=True) as ai_narration_group:
                    llm_provider = gr.Dropdown(
                        choices=list(LLM_PROVIDERS.keys()),
                        value=list(LLM_PROVIDERS.keys())[0],
                        label="AI 模型來源",
                    )
                    with gr.Group(visible=True) as ollama_group:
                        ollama_model = gr.Dropdown(
                            choices=["gemma4:26b", "gemma4:e4b"],
                            value="gemma4:26b", label="Ollama 模型",
                            allow_custom_value=True,
                        )
                        ollama_url = gr.Textbox(
                            label="Ollama 位址", value=DEFAULT_OLLAMA_URL, lines=1,
                        )
                    with gr.Group(visible=False) as gemini_group:
                        llm_api_key = gr.Textbox(
                            label="Gemini API Key", type="password", lines=1,
                        )
                    narration_style = gr.Textbox(
                        label="旁白風格（選填）",
                        placeholder="例如：Discovery 紀錄片 / 輕鬆 Vlog / 詩意文藝",
                        lines=1,
                    )
                narration_input = gr.Textbox(
                    label="旁白文字", lines=5, visible=False,
                    placeholder="在這裡輸入旁白內容...",
                )

            phase1_btn = gr.Button("Phase 1: 生成旁白稿", variant="primary", size="lg")

            # === PHASE 2 settings ===
            gr.Markdown("### Phase 2 - 配音製作")

            with gr.Accordion("語音設定", open=True):
                tts_engine = gr.Dropdown(
                    choices=list(TTS_ENGINES.keys()),
                    value=list(TTS_ENGINES.keys())[0],
                    label="語音引擎",
                )
                voice_select = gr.Dropdown(
                    choices=list(VOICE_OPTIONS.keys()),
                    value=list(VOICE_OPTIONS.keys())[0],
                    label="語音角色",
                )
                voice_sample = gr.Audio(
                    label="聲音樣本（錄製或上傳，用於聲音複製）",
                    sources=["microphone", "upload"],
                    type="filepath",
                    visible=False,
                )

            with gr.Accordion("配樂設定", open=False):
                music_engine = gr.Dropdown(
                    choices=list(MUSIC_ENGINES.keys()),
                    value=list(MUSIC_ENGINES.keys())[0],
                    label="配樂引擎",
                )
                music_prompt = gr.Textbox(
                    label="配樂風格描述", value="溫柔的戶外自然環境背景音樂",
                )
                music_duration = gr.Slider(
                    minimum=30, maximum=600, value=120, step=10, label="配樂時長（秒）",
                )
                bgm_volume = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.15, step=0.05, label="背景音樂音量",
                )

            phase2_btn = gr.Button("Phase 2: 完成製作", variant="primary", size="lg")

        # ── Right column ──
        with gr.Column(scale=1):
            gr.Markdown("### 輸出結果")
            output_video = gr.Video(label="最終輸出影片")

            gr.Markdown("### 旁白稿（可編輯）")
            generated_script = gr.Textbox(
                label="AI 生成的旁白稿 - Phase 1 完成後在此審閱/編輯，再執行 Phase 2",
                lines=10,
                interactive=True,
            )

            log_output = gr.Textbox(
                label="執行紀錄", lines=10, interactive=False,
                elem_classes=["log-box"],
            )

        # ── Tab 1 event handlers ──

        def toggle_narration_mode(mode):
            if mode == "AI 自動生成":
                return gr.update(visible=True), gr.update(visible=False)
            elif mode == "手動輸入":
                return gr.update(visible=False), gr.update(visible=True)
            else:
                return gr.update(visible=False), gr.update(visible=False)

        narration_mode.change(
            fn=toggle_narration_mode,
            inputs=[narration_mode],
            outputs=[ai_narration_group, narration_input],
        )

        def toggle_llm_provider(label):
            is_ollama = LLM_PROVIDERS.get(label) == "ollama"
            return gr.update(visible=is_ollama), gr.update(visible=not is_ollama)

        llm_provider.change(
            fn=toggle_llm_provider,
            inputs=[llm_provider],
            outputs=[ollama_group, gemini_group],
        )

        def toggle_voice_sample(engine_label):
            needs_sample = any(k in engine_label for k in ["複製"])
            return gr.update(visible=needs_sample)

        tts_engine.change(
            fn=toggle_voice_sample,
            inputs=[tts_engine],
            outputs=[voice_sample],
        )

        def handle_phase1(video, remove_voc, narr_mode, narr_text,
                          llm_prov, ol_model, ol_url, api_key, style):
            html, log, narration, ctx = do_phase1(
                video, remove_voc, narr_mode, narr_text,
                llm_prov, ol_model, ol_url, api_key, style,
            )
            steps_data = make_initial_steps_data()
            return html, log, narration, ctx, steps_data

        phase1_btn.click(
            fn=handle_phase1,
            inputs=[
                video_input, remove_vocals,
                narration_mode, narration_input,
                llm_provider, ollama_model, ollama_url, llm_api_key, narration_style,
            ],
            outputs=[pipeline_html, log_output, generated_script, pipeline_ctx, steps_data_state],
        )

        def handle_phase2(ctx, edited_narr, tts_eng, voice_sel, v_sample,
                          mus_eng, mus_prompt, mus_dur, bgm_vol, steps_data):
            html, log, video = do_phase2(
                ctx, edited_narr, tts_eng, voice_sel, v_sample,
                mus_eng, mus_prompt, mus_dur, bgm_vol, steps_data,
            )
            return html, log, video

        phase2_btn.click(
            fn=handle_phase2,
            inputs=[
                pipeline_ctx, generated_script,
                tts_engine, voice_select, voice_sample,
                music_engine, music_prompt, music_duration, bgm_volume,
                steps_data_state,
            ],
            outputs=[pipeline_html, log_output, output_video],
        )

        # ═══════════════════════════════════════════════
        # TAB 2: 逐字稿
        # ═══════════════════════════════════════════════
        with gr.Tab("逐字稿（訪談/會議）"):
            gr.Markdown(
                "### 影片逐字稿\n"
                "上傳訪談或會議影片，AI 自動辨識語音並輸出帶時間戳的逐字稿 + SRT 字幕檔。"
            )

            with gr.Row():
                with gr.Column(scale=1):
                    tr_video = gr.File(
                        label="上傳影片（支援 MP4/MOV/MKV/AVI）",
                        file_types=["video"],
                    )
                    tr_language = gr.Dropdown(
                        choices=["zh (中文)", "en (英文)", "ja (日文)", "ko (韓文)"],
                        value="zh (中文)",
                        label="語言",
                    )
                    tr_btn = gr.Button("開始辨識", variant="primary", size="lg")
                    tr_log = gr.Textbox(label="辨識紀錄", lines=6, interactive=False)

                with gr.Column(scale=1):
                    tr_output = gr.Textbox(
                        label="逐字稿（帶時間戳）",
                        lines=18, interactive=True,
                        placeholder="辨識結果將顯示在這裡...",
                    )
                    with gr.Row():
                        tr_download_txt = gr.File(label="下載 TXT", visible=False)
                        tr_download_srt = gr.File(label="下載 SRT", visible=False)

            def handle_transcribe(video_file, lang_label):
                if video_file is None:
                    return "[!] 請先上傳影片。", "", gr.update(visible=False), gr.update(visible=False)

                lang_code = lang_label.split(" ")[0] if lang_label else "zh"
                log_lines = []

                from datetime import datetime
                ws = str(WORKSPACE / f"transcribe_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

                result = transcribe_video(
                    video_path=video_file,
                    workspace=ws,
                    language=lang_code,
                    on_log=lambda msg: log_lines.append(msg),
                )

                log_text = "\n".join(log_lines)

                if "error" in result:
                    return log_text + f"\n[ERROR] {result['error']}", "", \
                           gr.update(visible=False), gr.update(visible=False)

                return (
                    log_text,
                    result["transcript"],
                    gr.update(value=result["transcript_file"], visible=True),
                    gr.update(value=result["srt_file"], visible=True),
                )

            tr_btn.click(
                fn=handle_transcribe,
                inputs=[tr_video, tr_language],
                outputs=[tr_log, tr_output, tr_download_txt, tr_download_srt],
            )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, theme=CUSTOM_THEME, css=CUSTOM_CSS)
