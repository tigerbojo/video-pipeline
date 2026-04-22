"""Step 3: AI voiceover - TTS / Voice Clone."""

import asyncio
from pathlib import Path
from .base import PipelineStep, StepResult, Status


class VoiceoverStep(PipelineStep):
    id = "voiceover"
    name = "3. AI 配音"
    description = "語音合成 / 聲音複製"
    icon = "[VOICE]"

    def check_deps(self) -> tuple[bool, str]:
        try:
            import edge_tts  # noqa: F401
            return True, "edge-tts 就緒（免費備援語音）"
        except ImportError:
            return False, "pip install edge-tts"

    def run(self, ctx: dict) -> StepResult:
        ws: Path = ctx["workspace"]
        out = ws / "03_voiceover.mp3"
        srt_out = ws / "03_voiceover.srt"
        text = ctx.get("narration_text", "")
        engine = ctx.get("tts_engine", "edge-tts")
        voice = ctx.get("tts_voice", "zh-TW-HsiaoChenNeural")
        ref_audio = ctx.get("voice_sample")

        if not text.strip():
            self.log("無旁白文字，跳過配音")
            ctx["voiceover"] = None
            return StepResult(status=Status.SKIPPED, message="無旁白文字")

        if engine == "fish-speech" and ref_audio:
            self.log("Fish-Speech 聲音複製尚未整合，使用 edge-tts 備援")
            self._run_edge_tts(text, voice, out, srt_out)
        elif engine == "gpt-sovits" and ref_audio:
            self.log("GPT-SoVITS 聲音複製尚未整合，使用 edge-tts 備援")
            self._run_edge_tts(text, voice, out, srt_out)
        elif engine == "cosyvoice" and ref_audio:
            self.log("CosyVoice 聲音複製尚未整合，使用 edge-tts 備援")
            self._run_edge_tts(text, voice, out, srt_out)
        else:
            self.log(f"使用 edge-tts，語音角色：{voice}")
            self._run_edge_tts(text, voice, out, srt_out)

        ctx["voiceover"] = out
        # Pass the TTS-generated SRT to subtitle step for perfect sync
        if srt_out.exists():
            ctx["voiceover_srt"] = srt_out
            self.log("已產生配音對時字幕")

        return StepResult(
            status=Status.DONE, output_files=[out, srt_out],
            message=f"配音完成（{engine}）",
            metadata={"engine": engine, "voice": voice}
        )

    def _run_edge_tts(self, text: str, voice: str, audio_out: Path, srt_out: Path):
        """Generate audio AND synchronized subtitles using edge-tts WordBoundary."""
        import edge_tts

        async def _generate():
            communicate = edge_tts.Communicate(text, voice)
            submaker = edge_tts.SubMaker()

            with open(audio_out, "wb") as audio_file:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_file.write(chunk["data"])
                    elif chunk["type"] == "WordBoundary":
                        submaker.create_sub(
                            (chunk["offset"], chunk["duration"]),
                            chunk["text"],
                        )

            # Generate SRT from WordBoundary data
            srt_content = submaker.generate_subs()
            if srt_content:
                srt_out.write_text(srt_content, encoding="utf-8")

        asyncio.run(_generate())
        self.log(f"edge-tts 輸出：{audio_out.name}")
