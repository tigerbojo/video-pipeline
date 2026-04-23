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

        # Dynamic ducking: when voiceover plays, auto-lower original audio and BGM
        vo_idx = None
        if has_orig_audio:
            audio_filters.append("[0:a]volume=0.3[oa]")
            amix_labels.append("[oa]")
            self.log("+ 原始音軌")

        if has_vo:
            inputs.extend(["-i", str(voiceover)])
            vo_idx = idx
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

        # Build filter_complex with sidechained ducking
        filters = []

        if has_vo and (has_orig_audio or has_bgm):
            # Use sidechaincompress: voiceover ducks the other audio sources
            self.log("+ 動態 ducking（旁白播放時自動降低背景音量）")
            duck_targets = []
            if has_orig_audio:
                duck_targets.append("[oa]")
            if has_bgm:
                duck_targets.append("[bg]")

            # Build: mix bg sources → duck with voiceover → combine with voiceover
            if len(duck_targets) == 1:
                bg_label = duck_targets[0]
                filters.extend(audio_filters)
                # Sidechain compress: when voiceover is loud, reduce bg
                filters.append(f"{bg_label}[vo]sidechaincompress=threshold=0.02:ratio=6:attack=200:release=1000[ducked]")
                filters.append(f"[ducked][vo]amix=inputs=2:duration=longest[aout]")
            else:
                # Multiple bg sources: mix them first, then duck
                filters.extend(audio_filters)
                bg_labels = "".join(duck_targets)
                filters.append(f"{bg_labels}amix=inputs={len(duck_targets)}:duration=longest[bgmix]")
                filters.append(f"[bgmix][vo]sidechaincompress=threshold=0.02:ratio=6:attack=200:release=1000[ducked]")
                filters.append(f"[ducked][vo]amix=inputs=2:duration=longest[aout]")
        else:
            # No ducking needed, simple mix
            filters.extend(audio_filters)
            n = len(amix_labels)
            if n > 1:
                filters.append(f"{''.join(amix_labels)}amix=inputs={n}:duration=longest[aout]")
            elif n == 1:
                filters[-1] = filters[-1].rsplit("[", 1)[0] + "[aout]"

        # Determine audio output label
        n = len(amix_labels)
        if n >= 1:
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
        # Use original video duration as reference (not BGM or voiceover)
        try:
            dur_str = run_cmd([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video_src)
            ]).strip()
            cmd.extend(["-t", dur_str])
            self.log(f"輸出時長：{float(dur_str):.1f} 秒（跟隨原始影片）")
        except Exception:
            pass
        cmd.extend([str(out)])

        self.log(f"FFmpeg 合併：{idx} 個輸入串流")
        run_cmd(cmd, timeout=1200)

        ctx["final_output"] = out
        return StepResult(
            status=Status.DONE, output_files=[out],
            message=f"最終影片：{out.name}"
        )
