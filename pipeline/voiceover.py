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
            return True, "edge-tts 就緒"
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
        sovits_url = ctx.get("sovits_url", "http://127.0.0.1:9880")

        if not text.strip():
            self.log("無旁白文字，跳過配音")
            ctx["voiceover"] = None
            return StepResult(status=Status.SKIPPED, message="無旁白文字")

        # Validate voice sample path
        if ref_audio:
            ref_path = Path(ref_audio) if isinstance(ref_audio, str) else None
            if ref_path and ref_path.exists() and ref_path.stat().st_size > 1000:
                self.log(f"聲音樣本：{ref_path.name} ({ref_path.stat().st_size // 1024} KB)")
            else:
                self.log(f"聲音樣本無效或太小，改用 edge-tts")
                ref_audio = None

        # GPT-SoVITS voice clone
        if engine == "gpt-sovits":
            if not ref_audio:
                self.log("未提供聲音樣本，GPT-SoVITS 需要聲音樣本才能複製")
                self.log("改用 edge-tts 備援")
                engine = "edge-tts"
            else:
                return self._run_gpt_sovits(text, str(ref_audio), sovits_url, out, srt_out, ctx)

        # Default: edge-tts
        self.log(f"使用 edge-tts，語音角色：{voice}")
        self._run_edge_tts(text, voice, out, srt_out)

        ctx["voiceover"] = out
        if srt_out.exists():
            ctx["voiceover_srt"] = srt_out
            self.log("已產生配音對時字幕")

        return StepResult(
            status=Status.DONE, output_files=[out, srt_out],
            message=f"配音完成（{engine}）",
            metadata={"engine": engine, "voice": voice}
        )

    def _run_gpt_sovits(self, text: str, ref_audio: str, base_url: str,
                        audio_out: Path, srt_out: Path, ctx: dict) -> StepResult:
        """Voice clone via GPT-SoVITS API."""
        from .engines.gpt_sovits import synthesize, health_check
        from .engines.text_splitter import split_text
        from .engines.audio_utils import concat_audio_files, convert_to_mp3

        # Resolve to absolute path (GPT-SoVITS needs absolute path)
        ref_audio = str(Path(ref_audio).resolve())
        self.log(f"聲音樣本路徑：{ref_audio}")

        # Check server
        self.log(f"檢查 GPT-SoVITS 伺服器（{base_url}）...")
        if not health_check(base_url):
            self.log("GPT-SoVITS 不可用，改用 edge-tts 備援")
            voice = ctx.get("tts_voice", "zh-TW-HsiaoChenNeural")
            self._run_edge_tts(text, voice, audio_out, srt_out)
            ctx["voiceover"] = audio_out
            if srt_out.exists():
                ctx["voiceover_srt"] = srt_out
            return StepResult(
                status=Status.DONE, output_files=[audio_out],
                message="GPT-SoVITS 不可用，已用 edge-tts 備援",
                metadata={"engine": "edge-tts", "fallback": True}
            )

        # Split text into segments
        segments = split_text(text, max_length=300)
        self.log(f"GPT-SoVITS 聲音複製：{len(segments)} 段文字")

        # Synthesize each segment
        ws = audio_out.parent
        seg_files = []
        for i, seg_text in enumerate(segments):
            seg_out = ws / f"sovits_seg_{i:03d}.wav"
            self.log(f"  合成第 {i+1}/{len(segments)} 段...")
            try:
                synthesize(
                    text=seg_text,
                    output_path=seg_out,
                    speaker_wav=ref_audio,
                    base_url=base_url,
                )
                seg_files.append(seg_out)
            except Exception as e:
                self.log(f"  第 {i+1} 段合成失敗：{e}")
                return StepResult(status=Status.ERROR,
                                  message=f"GPT-SoVITS 合成失敗：{e}")

        # Concat all segments
        if not seg_files:
            return StepResult(status=Status.ERROR, message="無成功合成的音段")

        try:
            concat_wav = ws / "sovits_concat.wav"
            self.log("拼接音段...")
            concat_audio_files(seg_files, concat_wav)

            # Convert to MP3
            convert_to_mp3(concat_wav, audio_out)
            self.log(f"GPT-SoVITS 輸出：{audio_out.name}")
        except Exception as e:
            return StepResult(status=Status.ERROR,
                              message=f"音段拼接/轉檔失敗：{e}")

        ctx["voiceover"] = audio_out
        # No WordBoundary for GPT-SoVITS, subtitle step will use ASR or text fallback
        return StepResult(
            status=Status.DONE, output_files=[audio_out],
            message=f"聲音複製完成（GPT-SoVITS，{len(segments)} 段）",
            metadata={"engine": "gpt-sovits", "segments": len(segments)}
        )

    def _run_edge_tts(self, text: str, voice: str, audio_out: Path, srt_out: Path):
        """Generate audio AND synchronized subtitles using edge-tts."""
        import edge_tts

        async def _generate():
            communicate = edge_tts.Communicate(text, voice)
            submaker = edge_tts.SubMaker()

            with open(audio_out, "wb") as audio_file:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_file.write(chunk["data"])
                    else:
                        submaker.feed(chunk)

            srt_content = submaker.get_srt()
            if srt_content:
                srt_out.write_text(srt_content, encoding="utf-8")

        asyncio.run(_generate())
        self.log(f"edge-tts 輸出：{audio_out.name}")
