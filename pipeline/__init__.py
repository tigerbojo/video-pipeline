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

    # Start run tracker
    from .run_metadata import RunTracker
    tracker = RunTracker(ws, mode=kwargs.get("narration_mode", "narration"))
    tracker.meta.source_video = str(ctx["source_video"])

    # Wrap on_step_done to also record metadata
    orig_done = on_step_done
    def _tracking_done(i, step, result):
        tracker.record_step(
            step.name, result.status.value, result.duration,
            result.message, [str(f) for f in result.output_files],
        )
        if orig_done:
            orig_done(i, step, result)

    steps, ctx = _run_steps(PHASE1_STEPS, ctx, on_step_start, _tracking_done, step_offset=0)
    ctx["_tracker"] = tracker

    # Save partial metadata if Phase 1 fails (Phase 2 may never run)
    has_error = any(s.result and s.result.status == Status.ERROR and s.required
                    for s in [cls() for cls in PHASE1_STEPS]
                    if hasattr(s, 'result'))
    # Actually check the real steps
    for s in steps:
        if s.result and s.result.status == Status.ERROR and s.required:
            tracker.finish(success=False, error="Phase 1 failed")
            break

    return steps, ctx


def run_phase2(ctx, on_step_start=None, on_step_done=None):
    """Phase 2: voiceover + subtitle + music + merge."""
    tracker = ctx.get("_tracker")

    orig_done = on_step_done
    def _tracking_done(i, step, result):
        if tracker:
            tracker.record_step(
                step.name, result.status.value, result.duration,
                result.message, [str(f) for f in result.output_files],
            )
        if orig_done:
            orig_done(i, step, result)

    steps, ctx = _run_steps(PHASE2_STEPS, ctx, on_step_start, _tracking_done,
                            step_offset=len(PHASE1_STEPS))

    # Finalize metadata
    if tracker:
        final = ctx.get("final_output")
        has_error = any(s.result and s.result.status == Status.ERROR for s in steps if s.required)
        tracker.finish(
            success=not has_error,
            output_video=str(final) if final else "",
        )

    return steps, ctx
