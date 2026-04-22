"""Step 1: Rough cut - silence removal + scene detection."""

from pathlib import Path
from .base import PipelineStep, StepResult, Status, cmd_exists, run_cmd


class RoughCutStep(PipelineStep):
    id = "rough_cut"
    name = "1. 自動粗剪"
    description = "去靜音 + 場景分段"
    icon = "[CUT]"

    def check_deps(self) -> tuple[bool, str]:
        if not cmd_exists("ffmpeg"):
            return False, "找不到 ffmpeg，請安裝：https://ffmpeg.org/download.html"
        return True, "ffmpeg 就緒"

    def run(self, ctx: dict) -> StepResult:
        src: Path = ctx["source_video"]
        ws: Path = ctx["workspace"]
        out = ws / "01_rough_cut.mp4"

        # Try auto-editor first (best quality)
        if cmd_exists("auto-editor"):
            self.log("使用 auto-editor 移除靜音段...")
            run_cmd([
                "auto-editor", str(src),
                "--margin", "0.3s",
                "--output", str(out),
                "--no-open",
            ])
            ctx["rough_cut"] = out
            return StepResult(status=Status.DONE, output_files=[out],
                              message="auto-editor：靜音段已移除")

        # Fallback: FFmpeg silence detection + trim
        self.log("找不到 auto-editor，使用 FFmpeg 備援...")
        # Detect silence
        detect = run_cmd([
            "ffmpeg", "-i", str(src),
            "-af", "silencedetect=noise=-30dB:d=1.5",
            "-f", "null", "-"
        ])
        # For the fallback, just copy the file (user can install auto-editor later)
        self.log("FFmpeg 備援：直接複製原始檔（安裝 auto-editor 可啟用真正的靜音移除）")
        run_cmd(["ffmpeg", "-y", "-i", str(src), "-c", "copy", str(out)])

        ctx["rough_cut"] = out
        return StepResult(
            status=Status.DONE, output_files=[out],
            message="FFmpeg 備援複製（安裝 auto-editor 以啟用靜音移除）",
            metadata={"fallback": True}
        )
