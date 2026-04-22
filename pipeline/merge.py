"""Step 6: Final merge - combine video + voiceover + subtitles + BGM."""

from pathlib import Path
from .base import PipelineStep, StepResult, Status, cmd_exists, run_cmd


class MergeStep(PipelineStep):
    id = "merge"
    name = "6. 合成輸出"
    description = "FFmpeg 合併影片+配音+字幕+配樂"
    icon = "[MERGE]"

    def check_deps(self) -> tuple[bool, str]:
        if not cmd_exists("ffmpeg"):
            return False, "找不到 ffmpeg"
        return True, "ffmpeg 就緒"

    def _has_audio_stream(self, path: Path) -> bool:
        try:
            out = run_cmd([
                "ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=index", "-of", "csv=p=0",
                str(path)
            ]).strip()
            return len(out) > 0
        except Exception:
            return False

    def run(self, ctx: dict) -> StepResult:
        ws: Path = ctx["workspace"]
        video_src = ctx.get("rough_cut") or ctx.get("source_video")
        voiceover = ctx.get("voiceover")
        subtitle = ctx.get("subtitle")
        bgm = ctx.get("bgm")
        bgm_volume = ctx.get("bgm_volume", 0.15)
        out = ws / "final_output.mp4"

        if not video_src or not Path(video_src).exists():
            return StepResult(status=Status.ERROR, message="找不到影片來源")

        has_orig_audio = self._has_audio_stream(Path(video_src))
        has_vo = voiceover and Path(voiceover).exists()
        has_bgm = bgm and Path(bgm).exists()

        # Collect audio inputs
        inputs = ["-i", str(video_src)]
        audio_filters = []
        amix_labels = []
        idx = 1

        if has_orig_audio:
            audio_filters.append("[0:a]volume=0.3[oa]")
            amix_labels.append("[oa]")
            self.log("+ 原始音軌")

        if has_vo:
            inputs.extend(["-i", str(voiceover)])
            audio_filters.append(f"[{idx}:a]volume=1.0[vo]")
            amix_labels.append("[vo]")
            idx += 1
            self.log("+ 配音音軌")

        if has_bgm:
            inputs.extend(["-i", str(bgm)])
            audio_filters.append(f"[{idx}:a]volume={bgm_volume}[bg]")
            amix_labels.append("[bg]")
            idx += 1
            self.log("+ 背景音樂音軌")

        # Build filter_complex
        filters = []
        filters.extend(audio_filters)

        n = len(amix_labels)
        if n > 1:
            filters.append(f"{''.join(amix_labels)}amix=inputs={n}:duration=longest[aout]")
            audio_out = "[aout]"
        elif n == 1:
            # Rename single audio to [aout]
            old_label = amix_labels[0]
            tag = old_label.strip("[]")
            # Replace last filter's output label
            filters[-1] = filters[-1].rsplit("[", 1)[0] + "[aout]"
            audio_out = "[aout]"
        else:
            audio_out = None

        # Subtitle burn-in
        has_sub = subtitle and Path(subtitle).exists()
        if has_sub:
            sub_path = str(subtitle).replace("\\", "/").replace(":", "\\:")
            filters.append(f"[0:v]subtitles='{sub_path}'[vout]")
            video_out = "[vout]"
            self.log("+ 燒入字幕")
        else:
            video_out = None

        # Assemble command
        cmd = ["ffmpeg", "-y"] + inputs

        if filters:
            cmd.extend(["-filter_complex", "; ".join(filters)])

        # Map streams
        cmd.extend(["-map", video_out if video_out else "0:v"])
        if audio_out:
            cmd.extend(["-map", audio_out])

        cmd.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "23"])
        if audio_out:
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        cmd.extend(["-shortest", str(out)])

        self.log(f"FFmpeg 合併：{idx} 個輸入串流")
        run_cmd(cmd, timeout=1200)

        ctx["final_output"] = out
        return StepResult(
            status=Status.DONE, output_files=[out],
            message=f"最終影片：{out.name}"
        )
