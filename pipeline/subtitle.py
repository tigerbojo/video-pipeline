"""Step 4: Subtitle generation - use TTS timing or ASR."""

import re
from pathlib import Path
from .base import PipelineStep, StepResult, Status, cmd_exists, run_cmd


class SubtitleStep(PipelineStep):
    id = "subtitle"
    name = "4. 字幕生成"
    description = "配音對時字幕 / 語音辨識"
    icon = "[SUB]"

    def check_deps(self) -> tuple[bool, str]:
        return True, "就緒"

    def run(self, ctx: dict) -> StepResult:
        ws: Path = ctx["workspace"]
        srt_file = ws / "04_subtitles.srt"

        # Priority 1: Use TTS-generated synced subtitles (perfect timing)
        voiceover_srt = ctx.get("voiceover_srt")
        if voiceover_srt and Path(voiceover_srt).exists():
            self.log("使用配音對時字幕（WordBoundary 精準同步）")
            raw = Path(voiceover_srt).read_text(encoding="utf-8")
            # edge-tts generates VTT, convert to SRT if needed
            srt_content = self._vtt_to_srt(raw) if "WEBVTT" in raw else raw
            srt_file.write_text(srt_content, encoding="utf-8")
            ctx["subtitle"] = srt_file
            return StepResult(
                status=Status.DONE, output_files=[srt_file],
                message="字幕完成（配音精準對時）"
            )

        # Priority 2: Run ASR on voiceover audio
        voiceover = ctx.get("voiceover")
        if voiceover and Path(voiceover).exists():
            try:
                import faster_whisper  # noqa: F401
                self.log("使用 faster-whisper 辨識配音音軌...")
                self._run_faster_whisper(Path(voiceover), srt_file)
                ctx["subtitle"] = srt_file
                return StepResult(
                    status=Status.DONE, output_files=[srt_file],
                    message="字幕完成（faster-whisper ASR）"
                )
            except ImportError:
                self.log("faster-whisper 未安裝")

        # Priority 3: Generate from narration text with estimated timing
        narration_text = ctx.get("narration_text", "")
        if narration_text.strip():
            self.log("從旁白文字生成字幕（估算時間）")
            voiceover_path = ctx.get("voiceover")
            duration = self._get_audio_duration(voiceover_path) if voiceover_path else None
            self._text_to_srt(narration_text, srt_file, duration)
            ctx["subtitle"] = srt_file
            return StepResult(
                status=Status.DONE, output_files=[srt_file],
                message="字幕完成（文字估算對時）",
                metadata={"fallback": True}
            )

        ctx["subtitle"] = None
        return StepResult(status=Status.SKIPPED, message="無可用來源生成字幕")

    def _vtt_to_srt(self, vtt_content: str) -> str:
        """Convert WebVTT to SRT format."""
        lines = vtt_content.strip().split("\n")
        srt_lines = []
        counter = 0
        i = 0
        # Skip header
        while i < len(lines) and not re.match(r"\d{2}:\d{2}", lines[i]):
            i += 1

        while i < len(lines):
            line = lines[i].strip()
            # Match timestamp line
            match = re.match(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})", line)
            if match:
                counter += 1
                start = match.group(1).replace(".", ",")
                end = match.group(2).replace(".", ",")
                srt_lines.append(str(counter))
                srt_lines.append(f"{start} --> {end}")
                # Collect text lines
                i += 1
                text_lines = []
                while i < len(lines) and lines[i].strip():
                    text_lines.append(lines[i].strip())
                    i += 1
                srt_lines.append("\n".join(text_lines))
                srt_lines.append("")
            else:
                i += 1

        return "\n".join(srt_lines)

    def _run_faster_whisper(self, audio_path: Path, srt_out: Path):
        from faster_whisper import WhisperModel
        model = WhisperModel("large-v3", compute_type="auto")
        segments, _ = model.transcribe(str(audio_path), language="zh")
        lines = []
        for i, seg in enumerate(segments, 1):
            start = self._format_time(seg.start)
            end = self._format_time(seg.end)
            lines.append(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n")
        srt_out.write_text("\n".join(lines), encoding="utf-8")

    def _get_audio_duration(self, audio_path) -> float | None:
        """Get audio duration using ffprobe."""
        if not audio_path or not Path(audio_path).exists():
            return None
        try:
            out = run_cmd([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path)
            ]).strip()
            return float(out)
        except Exception:
            return None

    def _text_to_srt(self, text: str, srt_out: Path, total_duration: float | None = None):
        """Split text into sentences and assign proportional duration."""
        sentences = [s.strip() for s in re.split(r'[.!?\u3002\uff01\uff1f\n]+', text) if s.strip()]
        if not sentences:
            sentences = [text]

        # Estimate duration per character if we have total duration
        total_chars = sum(len(s) for s in sentences)
        if total_duration and total_chars > 0:
            char_duration = total_duration / total_chars
        else:
            char_duration = 0.15  # ~150ms per character as fallback

        lines = []
        current_time = 0.0
        for i, sent in enumerate(sentences, 1):
            duration = len(sent) * char_duration
            start = self._format_time(current_time)
            end = self._format_time(current_time + duration)
            lines.append(f"{i}\n{start} --> {end}\n{sent}\n")
            current_time += duration

        srt_out.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _format_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
