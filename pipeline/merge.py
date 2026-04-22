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

        # Build FFmpeg command
        inputs = ["-i", str(video_src)]
        filter_parts = []
        audio_mix = []
        stream_idx = 1

        # Original audio from video
        audio_mix.append(f"[0:a]volume=0.3[orig_a]")
        mix_inputs = "[orig_a]"

        # Add voiceover
        if voiceover and Path(voiceover).exists():
            inputs.extend(["-i", str(voiceover)])
            audio_mix.append(f"[{stream_idx}:a]volume=1.0[vo_a]")
            mix_inputs += "[vo_a]"
            stream_idx += 1
            self.log("+ 配音音軌")

        # Add BGM
        if bgm and Path(bgm).exists():
            inputs.extend(["-i", str(bgm)])
            audio_mix.append(f"[{stream_idx}:a]volume={bgm_volume}[bgm_a]")
            mix_inputs += "[bgm_a]"
            stream_idx += 1
            self.log("+ 背景音樂音軌")

        # Build audio mix filter
        n_audio = mix_inputs.count("[")
        if n_audio > 1:
            filter_complex = "; ".join(audio_mix)
            filter_complex += f"; {mix_inputs}amix=inputs={n_audio}:duration=longest[mixed]"
        else:
            filter_complex = f"[0:a]volume=1.0[mixed]"

        # Build command
        cmd = ["ffmpeg", "-y"] + inputs

        # Add subtitle filter if available
        if subtitle and Path(subtitle).exists():
            # Burn subtitles into video
            sub_path = str(subtitle).replace("\\", "/").replace(":", "\\:")
            filter_complex += f"; [0:v]subtitles='{sub_path}'[v_out]"
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[v_out]", "-map", "[mixed]",
            ])
            self.log("+ 燒入字幕")
        else:
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "0:v", "-map", "[mixed]",
            ])

        cmd.extend([
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(out)
        ])

        self.log(f"FFmpeg 合併：{len(inputs) // 2} 個輸入串流")
        run_cmd(cmd, timeout=1200)

        ctx["final_output"] = out
        return StepResult(
            status=Status.DONE, output_files=[out],
            message=f"最終影片：{out.name}"
        )
