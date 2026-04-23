"""Typed pipeline context — replaces untyped ctx dict."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PipelineContext:
    """Shared state passed between pipeline steps."""

    # Input
    source_video: Path = Path()
    workspace: Path = Path()

    # Mode
    mode: str = "narration"  # "narration" or "subtitle"
    remove_vocals: bool = False

    # Narration settings
    narration_mode: str = "skip"  # "ai", "manual", "skip"
    narration_script: str = ""
    narration_style: str = ""

    # LLM
    llm_provider: str = "ollama"
    llm_api_key: str = ""
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:26b"

    # TTS
    tts_engine: str = "edge-tts"
    tts_voice: str = "zh-TW-HsiaoChenNeural"
    voice_sample: Optional[str] = None
    sovits_url: str = "http://127.0.0.1:9880"

    # Subtitle mode
    asr_engine: str = "auto"
    whisper_model: str = "large-v3"
    sub_language: str = "zh"

    # Music
    music_engine: str = "silence"
    music_prompt: str = "溫柔的戶外自然環境背景音樂"
    music_duration: int = 120
    bgm_volume: float = 0.15

    # Intermediate outputs (populated by steps)
    rough_cut: Optional[Path] = None
    narration_text: str = ""
    narration_script_file: Optional[Path] = None
    voiceover: Optional[Path] = None
    voiceover_srt: Optional[Path] = None
    subtitle: Optional[Path] = None
    bgm: Optional[Path] = None
    final_output: Optional[Path] = None

    def to_dict(self) -> dict:
        """Convert to dict for backward compatibility with existing steps."""
        d = {}
        for k, v in self.__dict__.items():
            d[k] = v
        return d

    @staticmethod
    def from_dict(d: dict) -> "PipelineContext":
        """Create from dict (backward compat)."""
        ctx = PipelineContext()
        for k, v in d.items():
            if hasattr(ctx, k):
                setattr(ctx, k, v)
        return ctx
