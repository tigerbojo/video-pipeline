"""Step 5: AI background music generation."""

from pathlib import Path
from .base import PipelineStep, StepResult, Status, cmd_exists


class MusicStep(PipelineStep):
    id = "music"
    name = "5. AI 配樂"
    description = "本地 AI 背景音樂生成"
    icon = "[MUSIC]"
    required = False

    def check_deps(self) -> tuple[bool, str]:
        from .engines.ace_step import is_available
        if is_available():
            return True, "ACE-Step 1.5 就緒"
        if cmd_exists("ffmpeg"):
            return True, "ffmpeg 就緒（ACE-Step 未安裝，靜音佔位）"
        return False, "無可用的配樂工具"

    def run(self, ctx: dict) -> StepResult:
        ws: Path = ctx["workspace"]
        out = ws / "05_bgm.mp3"
        engine = ctx.get("music_engine", "silence")
        prompt = ctx.get("music_prompt", "溫柔的戶外自然環境背景音樂")

        # Match music duration to video duration
        video_src = ctx.get("rough_cut") or ctx.get("source_video")
        try:
            from .base import run_cmd
            dur_str = run_cmd([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video_src)
            ]).strip()
            duration = min(int(float(dur_str)) + 5, 300)  # pad 5s, cap at 300s
        except Exception:
            duration = ctx.get("music_duration", 120)

        # ACE-Step
        if engine == "ace-step":
            try:
                from .engines.ace_step import generate_music, find_ace_step
                ace_dir = find_ace_step()
                if not ace_dir:
                    self.log("ACE-Step 未安裝，產生靜音佔位")
                    self._generate_silence(out, duration)
                else:
                    self.log(f"ACE-Step 生成配樂：「{prompt}」（{duration}秒）...")
                    generate_music(prompt, out, duration, ace_dir)
                    self.log(f"AI 配樂完成：{out.name}")
                    ctx["bgm"] = out
                    return StepResult(
                        status=Status.DONE, output_files=[out],
                        message=f"AI 配樂完成（ACE-Step，{duration}秒）",
                        metadata={"engine": "ace-step"}
                    )
            except Exception as e:
                self.log(f"ACE-Step 失敗：{e}")

        # Fallback: silence
        self.log(f"靜音佔位（{duration}秒）")
        self._generate_silence(out, duration)
        ctx["bgm"] = out
        return StepResult(
            status=Status.DONE, output_files=[out],
            message=f"靜音佔位（{duration}秒）",
            metadata={"fallback": True, "engine": "silence"}
        )

    def _generate_silence(self, output: Path, duration: int):
        from .base import run_cmd
        run_cmd([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-t", str(duration), "-q:a", "9",
            str(output)
        ])
