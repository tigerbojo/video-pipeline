"""Step 1: Rough cut - silence removal + scene detection."""

import sys
from pathlib import Path
from .base import PipelineStep, StepResult, Status, cmd_exists, run_cmd


def _find_auto_editor() -> str | None:
    """Find auto-editor binary, including Python Scripts dir."""
    import shutil
    path = shutil.which("auto-editor")
    if path:
        return path
    # Check Python Scripts directory (Windows)
    scripts = Path(sys.executable).parent / "Scripts" / "auto-editor.exe"
    if scripts.exists():
        return str(scripts)
    # Check user site-packages Scripts
    user_scripts = Path.home() / "AppData" / "Roaming" / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts" / "auto-editor.exe"
    if user_scripts.exists():
        return str(user_scripts)
    return None


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
        ae_path = _find_auto_editor()
        if ae_path:
            self.log("使用 auto-editor 移除靜音段...")
            run_cmd([
                ae_path, str(src),
                "--margin", "0.3s",
                "--output", str(out),
                "--no-open",
            ])
            ctx["rough_cut"] = out
            return StepResult(status=Status.DONE, output_files=[out],
                              message="auto-editor：靜音段已移除")

        # Fallback: remux to MKV (handles all input containers safely)
        self.log("找不到 auto-editor，直接 remux 原始檔（安裝 auto-editor 可啟用靜音移除）")
        out = ws / "01_rough_cut.mkv"
        run_cmd(["ffmpeg", "-y", "-i", str(src), "-c", "copy", str(out)])

        ctx["rough_cut"] = out
        return StepResult(
            status=Status.DONE, output_files=[out],
            message="FFmpeg 備援複製（安裝 auto-editor 以啟用靜音移除）",
            metadata={"fallback": True}
        )
