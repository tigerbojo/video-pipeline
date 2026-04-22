"""Step 2: AI narration script generation."""

import base64
import json
from pathlib import Path
from .base import PipelineStep, StepResult, Status, cmd_exists, run_cmd


NARRATION_PROMPT = (
    "你是一位專業的戶外探險影片旁白撰稿人。\n"
    "以下是一段戶外影片的關鍵幀截圖（按時間順序排列）。\n"
    "請根據這些畫面，用繁體中文撰寫一段自然、流暢的旁白腳本。\n\n"
    "要求：\n"
    "- 用第一人稱（我）敘述\n"
    "- 描述眼前看到的景色、感受、氛圍\n"
    "- 語氣溫暖自然，像在跟朋友分享旅途\n"
    "- 適當加入地理、生態、文化相關的小知識\n"
    "- 長度約 200-400 字（適合 1-3 分鐘影片配音）\n"
    "- 只輸出旁白文字，不要加標記或時間戳\n"
)


class NarrationStep(PipelineStep):
    id = "narration"
    name = "2. AI 旁白稿"
    description = "LLM 分析畫面自動生成旁白"
    icon = "[SCRIPT]"

    def check_deps(self) -> tuple[bool, str]:
        return True, "就緒"

    def run(self, ctx: dict) -> StepResult:
        ws: Path = ctx["workspace"]
        script_file = ws / "02_narration.txt"
        user_script: str = ctx.get("narration_script", "")
        narration_mode: str = ctx.get("narration_mode", "manual")

        # Mode 1: User provided script directly
        if narration_mode == "manual":
            if not user_script.strip():
                return StepResult(
                    status=Status.ERROR,
                    message="手動模式但旁白文字為空，請輸入旁白內容或切換為「AI 自動生成」"
                )
            self.log("使用手動輸入的旁白腳本")
            script_file.write_text(user_script, encoding="utf-8")
            ctx["narration_script_file"] = script_file
            ctx["narration_text"] = user_script
            return StepResult(
                status=Status.DONE, output_files=[script_file],
                message=f"旁白稿就緒（{len(user_script)} 字）"
            )

        # Mode 2: AI auto-generate from video frames
        if narration_mode == "ai":
            video_path = ctx.get("rough_cut") or ctx.get("source_video")
            style = ctx.get("narration_style", "")
            llm_provider = ctx.get("llm_provider", "ollama")

            # Extract key frames
            frames_dir = ws / "frames"
            frames_dir.mkdir(exist_ok=True)
            frames = self._extract_frames(video_path, frames_dir)
            self.log(f"已擷取 {len(frames)} 張關鍵幀")

            # Build prompt with style
            prompt = NARRATION_PROMPT
            if style.strip():
                prompt += f"\n旁白風格要求：{style}"

            # Generate narration
            if llm_provider == "ollama":
                ollama_url = ctx.get("ollama_url", "http://localhost:11434")
                ollama_model = ctx.get("ollama_model", "gemma4:26b")
                script = self._generate_with_ollama(frames, prompt, ollama_url, ollama_model)
            elif llm_provider == "gemini":
                api_key = ctx.get("llm_api_key", "")
                if not api_key:
                    return StepResult(status=Status.ERROR, message="請輸入 Gemini API Key")
                script = self._generate_with_gemini(frames, prompt, api_key)
            else:
                return StepResult(status=Status.ERROR, message=f"不支援的 LLM：{llm_provider}")

            script_file.write_text(script, encoding="utf-8")
            ctx["narration_script_file"] = script_file
            ctx["narration_text"] = script
            return StepResult(
                status=Status.DONE, output_files=[script_file],
                message=f"AI 旁白稿完成（{len(script)} 字）"
            )

        # Mode 3: Skip narration
        self.log("未設定旁白，跳過此步驟")
        ctx["narration_text"] = ""
        return StepResult(status=Status.SKIPPED, message="略過旁白生成")

    def _extract_frames(self, video_path: Path, output_dir: Path, interval: int = 10) -> list[Path]:
        """Extract one frame every N seconds from video."""
        if not cmd_exists("ffmpeg"):
            raise RuntimeError("需要 ffmpeg 來擷取影格")

        duration_str = run_cmd([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]).strip()
        duration = float(duration_str)
        self.log(f"影片長度：{duration:.0f} 秒")

        pattern = str(output_dir / "frame_%04d.jpg")
        run_cmd([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"fps=1/{interval},scale=1280:-1",
            "-q:v", "3",
            pattern
        ])

        frames = sorted(output_dir.glob("frame_*.jpg"))

        # Cap frames to avoid overwhelming the model
        max_frames = 20
        if len(frames) > max_frames:
            step = len(frames) // max_frames
            frames = frames[::step][:max_frames]
            self.log(f"取樣至 {len(frames)} 張")

        return frames

    def _generate_with_ollama(
        self, frames: list[Path], prompt: str, url: str, model: str
    ) -> str:
        """Send frames to local Ollama and generate narration."""
        import urllib.request

        # Encode all frames as base64
        images_b64 = []
        for f in frames:
            with open(f, "rb") as fh:
                images_b64.append(base64.b64encode(fh.read()).decode("utf-8"))

        self.log(f"呼叫 Ollama（{model}，{len(frames)} 張圖片）...")

        payload = json.dumps({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": images_b64,
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 2048,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{url.rstrip('/')}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(
                f"Ollama 連線失敗（{url}）：{e}\n"
                "請確認 Ollama 正在執行（在終端機執行 ollama serve）"
            )

        script = result.get("message", {}).get("content", "").strip()
        if not script:
            raise RuntimeError("Ollama 回傳空內容，請確認模型支援 vision")

        self.log(f"AI 生成完成：{len(script)} 字")
        return script

    def _generate_with_gemini(self, frames: list[Path], prompt: str, api_key: str) -> str:
        """Send frames to Gemini and generate narration."""
        import google.generativeai as genai
        import PIL.Image

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        parts = [prompt]
        for f in frames:
            parts.append(PIL.Image.open(f))

        self.log(f"呼叫 Gemini API（{len(frames)} 張圖片）...")
        response = model.generate_content(parts)
        script = response.text.strip()
        self.log(f"AI 生成完成：{len(script)} 字")
        return script
