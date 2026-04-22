"""Step 5: AI background music generation."""

from pathlib import Path
from .base import PipelineStep, StepResult, Status, cmd_exists


class MusicStep(PipelineStep):
    id = "music"
    name = "5. AI 配樂"
    description = "本地 AI 背景音樂生成"
    icon = "[MUSIC]"

    def check_deps(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            return True, "PyTorch 可用（ACE-Step 就緒）"
        except ImportError:
            pass
        if cmd_exists("ffmpeg"):
            return True, "ffmpeg 就緒（將產生靜音佔位）"
        return False, "無可用的配樂生成工具"

    def run(self, ctx: dict) -> StepResult:
        ws: Path = ctx["workspace"]
        out = ws / "05_bgm.mp3"
        engine = ctx.get("music_engine", "auto")
        prompt = ctx.get("music_prompt", "gentle ambient outdoor nature background music")
        duration = ctx.get("music_duration", 120)

        # Try ACE-Step
        if engine in ("auto", "ace-step"):
            try:
                return self._run_ace_step(prompt, duration, out, ctx)
            except Exception as e:
                self.log(f"ACE-Step 失敗：{e}，切換備援...")

        # Fallback: generate silent audio as placeholder
        self.log("產生靜音佔位背景音樂（安裝 ACE-Step 可啟用 AI 配樂）")
        self._generate_silence(out, duration)
        ctx["bgm"] = out
        return StepResult(
            status=Status.DONE, output_files=[out],
            message=f"靜音佔位（{duration}秒）。安裝 ACE-Step 以啟用 AI 配樂。",
            metadata={"fallback": True, "engine": "silence"}
        )

    def _run_ace_step(self, prompt: str, duration: int, output: Path, ctx: dict) -> StepResult:
        self.log(f"ACE-Step：生成「{prompt}」（{duration}秒）...")
        raise NotImplementedError(
            "ACE-Step 尚未整合。"
            "安裝方式：git clone https://github.com/ace-step/ACE-Step-1.5"
        )

    def _generate_silence(self, output: Path, duration: int):
        """Generate silent audio file as placeholder."""
        from .base import run_cmd
        run_cmd([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo",
            "-t", str(duration),
            "-q:a", "9",
            str(output)
        ])
