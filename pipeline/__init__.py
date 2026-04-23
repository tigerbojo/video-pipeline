"""AI Video Pipeline - step registry and orchestrator."""

from pathlib import Path
from datetime import datetime
import uuid
from .base import PipelineStep, Status
from .vocal_remove import VocalRemoveStep
from .rough_cut import RoughCutStep
from .narration import NarrationStep
from .voiceover import VoiceoverStep
from .subtitle import SubtitleStep
from .music import MusicStep
from .merge import MergeStep
from .context import PipelineContext

PHASE1_STEPS = [VocalRemoveStep, RoughCutStep, NarrationStep]
PHASE2_STEPS = [VoiceoverStep, SubtitleStep, MusicStep, MergeStep]
ALL_STEPS = PHASE1_STEPS + PHASE2_STEPS


def create_pipeline() -> list[PipelineStep]:
    return [cls() for cls in ALL_STEPS]


def make_workspace(base: str) -> Path:
    ws = Path(base) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _run_steps(step_classes, ctx, on_step_start=None, on_step_done=None, step_offset=0):
    # Steps use dict interface for now (backward compat)
    ctx_dict = ctx if isinstance(ctx, dict) else ctx.to_dict()
    steps = [cls() for cls in step_classes]
    for i, step in enumerate(steps):
        idx = i + step_offset
        if on_step_start:
            on_step_start(idx, step)
        result = step.execute(ctx_dict)
        if on_step_done:
            on_step_done(idx, step, result)
        if result.status == Status.ERROR and step.required:
            break
    return steps, ctx_dict


def run_phase1(on_step_start=None, on_step_done=None, **kwargs):
    """Phase 1: vocal removal + rough cut + AI narration."""
    ws = make_workspace(kwargs.pop("workspace", "."))
    ctx = {
        "workspace": ws,
        "source_video": Path(kwargs.pop("source_video")),
        **kwargs,
    }
    steps, ctx = _run_steps(PHASE1_STEPS, ctx, on_step_start, on_step_done, step_offset=0)
    return steps, ctx


def run_phase2(ctx, on_step_start=None, on_step_done=None):
    """Phase 2: voiceover + subtitle + music + merge."""
    steps, ctx = _run_steps(PHASE2_STEPS, ctx, on_step_start, on_step_done,
                            step_offset=len(PHASE1_STEPS))
    return steps, ctx
