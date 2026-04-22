"""Step 0 (optional): Remove vocals from original video."""

from pathlib import Path
from .base import PipelineStep, StepResult, Status, cmd_exists, run_cmd


class VocalRemoveStep(PipelineStep):
    id = "vocal_remove"
    name = "0. 去人聲"
    description = "音軌分離，保留環境音"
    icon = "[VOCAL]"
    required = False

    def check_deps(self) -> tuple[bool, str]:
        if not cmd_exists("ffmpeg"):
            return False, "需要 ffmpeg"
        return True, "ffmpeg 就緒"

    def run(self, ctx: dict) -> StepResult:
        if not ctx.get("remove_vocals"):
            return StepResult(status=Status.SKIPPED, message="未啟用去人聲")

        src: Path = ctx["source_video"]
        ws: Path = ctx["workspace"]
        out = ws / "00_no_vocals.mp4"

        # Try demucs first (best quality)
        try:
            self._run_demucs(src, ws, out)
            ctx["source_video"] = out
            return StepResult(
                status=Status.DONE, output_files=[out],
                message="人聲已移除（demucs AI 分離）"
            )
        except Exception as e:
            self.log(f"demucs 無法使用：{str(e)[:100]}，切換 FFmpeg 備援...")

        # Fallback: FFmpeg center-channel cancellation
        self.log("使用 FFmpeg 中央聲道消除法（適用大部分人聲在正中的影片）")
        try:
            run_cmd([
                "ffmpeg", "-y", "-i", str(src),
                "-af", "pan=stereo|c0=c0-0.5*c1|c1=c1-0.5*c0,bass=g=3",
                "-c:v", "copy",
                str(out)
            ])
            ctx["source_video"] = out
            return StepResult(
                status=Status.DONE, output_files=[out],
                message="人聲已減弱（FFmpeg 中央聲道消除）",
                metadata={"method": "ffmpeg_fallback"}
            )
        except Exception as e2:
            self.log(f"FFmpeg 備援也失敗：{e2}")
            # Non-blocking: continue with original audio
            self.log("繼續使用原始音軌")
            return StepResult(
                status=Status.SKIPPED,
                message="去人聲失敗，使用原始音軌繼續"
            )

    def _run_demucs(self, src: Path, ws: Path, out: Path):
        """Use demucs for high-quality vocal separation."""
        import demucs  # noqa: F401

        # Extract audio
        audio_file = ws / "original_audio.wav"
        self.log("擷取原始音軌...")
        run_cmd([
            "ffmpeg", "-y", "-i", str(src),
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
            str(audio_file)
        ])

        # Run demucs
        self.log("demucs 分離人聲中（可能需要 1-3 分鐘）...")
        import subprocess
        result = subprocess.run(
            ["python", "-m", "demucs", "-n", "htdemucs",
             "--two-stems", "vocals",
             "-o", str(ws / "demucs_out"),
             str(audio_file)],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-300:])

        # Find instrumental track
        no_vocals = ws / "demucs_out" / "htdemucs" / "original_audio" / "no_vocals.wav"
        if not no_vocals.exists():
            raise RuntimeError(f"找不到去人聲音軌：{no_vocals}")

        # Mux back with video
        self.log("替換音軌...")
        run_cmd([
            "ffmpeg", "-y",
            "-i", str(src), "-i", str(no_vocals),
            "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", str(out)
        ])
        self.log("demucs 去人聲完成")
