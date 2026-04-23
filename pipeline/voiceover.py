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

        # Validate voice sample path (handle both str and Path)
        if ref_audio:
            ref_audio = str(ref_audio)  # normalize Path to str
            ref_path = Path(ref_audio)
            if ref_path.exists() and ref_path.stat().st_size > 1000:
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

        # Pre-process ref audio: trim + normalize + denoise for better clone quality
        ref_src = Path(ref_audio).resolve()
        ref_copy = audio_out.parent / "ref_voice.wav"
        try:
            from .base import run_cmd
            dur_str = run_cmd([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(ref_src)
            ]).strip()
            dur = float(dur_str)
            if dur < 3:
                self.log(f"聲音樣本太短（{dur:.1f} 秒），需要至少 3 秒")
                return StepResult(status=Status.ERROR, message="聲音樣本太短，需要 3-10 秒")

            # FFmpeg pipeline: trim to 10s + remove silence + normalize loudness + 16kHz mono
            trim = min(dur, 10)
            self.log(f"聲音樣本前處理（{dur:.1f}s→{trim:.0f}s, 正規化, 降噪）")
            run_cmd([
                "ffmpeg", "-y", "-i", str(ref_src),
                "-t", str(trim),
                "-af", "silenceremove=start_periods=1:start_silence=0.3:start_threshold=-40dB,"
                       "loudnorm=I=-16:TP=-1.5:LRA=11,"
                       "highpass=f=80,lowpass=f=8000",
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                str(ref_copy)
            ])
        except Exception:
            import shutil
            shutil.copy2(ref_src, ref_copy)

        ref_audio = str(ref_copy)
        self.log(f"聲音樣本：{ref_copy.name}")

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

        # Auto-transcribe reference audio for prompt_text (critical for quality)
        ref_text = ""
        try:
            from .engines.whisper_pool import get_model
            self.log("辨識聲音樣本內容（作為 prompt_text）...")
            wm = get_model("medium")
            segs, _ = wm.transcribe(ref_audio, language="zh")
            ref_text = "".join(s.text for s in segs).strip()
            if ref_text:
                self.log(f"prompt_text：{ref_text[:60]}")
            else:
                self.log("警告：聲音樣本辨識為空，voice clone 品質可能不佳")
                self.log("建議：錄音時請清楚說一段話（非靜音或背景噪音）")
        except Exception as e:
            self.log(f"prompt_text 辨識失敗：{e}")
            self.log("警告：沒有 prompt_text，GPT-SoVITS 品質可能不穩定")

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
                    ref_text=ref_text,
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

            # Validate output duration — GPT-SoVITS sometimes produces garbage
            from .engines.audio_utils import get_audio_duration
            vo_dur = get_audio_duration(concat_wav)
            expected_min = len(text) * 0.05  # ~50ms per char minimum
            if vo_dur < 3 or vo_dur < expected_min:
                self.log(f"GPT-SoVITS 輸出異常短（{vo_dur:.1f}秒 / 預期>{expected_min:.0f}秒）")
                self.log("預訓練模型品質不足，改用 edge-tts 備援")
                self.log("（提示：用 GPT-SoVITS WebUI 訓練你的專屬聲音模型可大幅改善）")
                voice = ctx.get("tts_voice", "zh-TW-HsiaoChenNeural")
                self._run_edge_tts(text, voice, audio_out, srt_out)
                ctx["voiceover"] = audio_out
                if srt_out.exists():
                    ctx["voiceover_srt"] = srt_out
                return StepResult(
                    status=Status.DONE, output_files=[audio_out],
                    message=f"GPT-SoVITS 品質不足（{vo_dur:.1f}秒），已用 edge-tts 備援",
                    metadata={"engine": "edge-tts", "fallback": True}
                )

            convert_to_mp3(concat_wav, audio_out)
            self.log(f"GPT-SoVITS 輸出：{audio_out.name}（{vo_dur:.1f}秒）")
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
