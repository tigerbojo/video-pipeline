"""AI Video Pipeline - step registry and orchestrator."""

from pathlib import Path
from datetime import datetime
from .base import PipelineStep, Status
from .vocal_remove import VocalRemoveStep
from .rough_cut import RoughCutStep
from .narration import NarrationStep
from .voiceover import VoiceoverStep
from .subtitle import SubtitleStep
from .music import MusicStep
from .merge import MergeStep

PHASE1_STEPS = [VocalRemoveStep, RoughCutStep, NarrationStep]
PHASE2_STEPS = [VoiceoverStep, SubtitleStep, MusicStep, MergeStep]
ALL_STEPS = PHASE1_STEPS + PHASE2_STEPS


def create_pipeline() -> list[PipelineStep]:
    return [cls() for cls in ALL_STEPS]


def _make_ctx(
    source_video, workspace,
    remove_vocals=False,
    narration_mode="skip", narration_script="", narration_style="",
    llm_provider="ollama", llm_api_key="",
    ollama_url="http://localhost:11434", ollama_model="gemma4:26b",
    tts_engine="edge-tts", tts_voice="zh-TW-HsiaoChenNeural",
    voice_sample=None,
    asr_engine="auto",
    music_engine="auto", music_prompt="", music_duration=120,
    bgm_volume=0.15,
) -> dict:
    ws = Path(workspace) / datetime.now().strftime("%Y%m%d_%H%M%S")
    ws.mkdir(parents=True, exist_ok=True)
    return {
        "source_video": Path(source_video),
        "workspace": ws,
        "remove_vocals": remove_vocals,
        "narration_mode": narration_mode,
        "narration_script": narration_script,
        "narration_style": narration_style,
        "llm_provider": llm_provider,
        "llm_api_key": llm_api_key,
        "ollama_url": ollama_url,
        "ollama_model": ollama_model,
        "tts_engine": tts_engine,
        "tts_voice": tts_voice,
        "voice_sample": voice_sample,
        "asr_engine": asr_engine,
        "music_engine": music_engine,
        "music_prompt": music_prompt,
        "music_duration": music_duration,
        "bgm_volume": bgm_volume,
    }


NON_BLOCKING_STEPS = {"vocal_remove", "music"}  # These can fail without stopping pipeline

def _run_steps(step_classes, ctx, on_step_start=None, on_step_done=None, step_offset=0):
    steps = [cls() for cls in step_classes]
    for i, step in enumerate(steps):
        idx = i + step_offset
        if on_step_start:
            on_step_start(idx, step)
        result = step.execute(ctx)
        if on_step_done:
            on_step_done(idx, step, result)
        if result.status == Status.ERROR and step.id not in NON_BLOCKING_STEPS:
            break
    return steps


def run_phase1(on_step_start=None, on_step_done=None, **kwargs):
    """Phase 1: vocal removal + rough cut + AI narration. Returns ctx for phase 2."""
    ctx = _make_ctx(**kwargs)
    steps = _run_steps(PHASE1_STEPS, ctx, on_step_start, on_step_done, step_offset=0)
    return steps, ctx


def run_phase2(ctx, on_step_start=None, on_step_done=None):
    """Phase 2: voiceover + subtitle + music + merge. Uses ctx from phase 1."""
    steps = _run_steps(PHASE2_STEPS, ctx, on_step_start, on_step_done,
                       step_offset=len(PHASE1_STEPS))
    return steps, ctx
