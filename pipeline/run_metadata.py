"""Structured run metadata — saved per workspace for tracking and debugging."""

import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class StepRecord:
    name: str = ""
    status: str = ""
    duration_s: float = 0.0
    message: str = ""
    output_files: list[str] = field(default_factory=list)


@dataclass
class RunMetadata:
    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    mode: str = ""  # "narration" or "subtitle"
    source_video: str = ""
    source_duration_s: float = 0.0
    output_video: str = ""
    output_duration_s: float = 0.0
    whisper_model: str = ""
    tts_engine: str = ""
    llm_model: str = ""
    steps: list[StepRecord] = field(default_factory=list)
    total_duration_s: float = 0.0
    success: bool = False
    error: Optional[str] = None

    def save(self, workspace: Path):
        path = workspace / "run_metadata.json"
        self.finished_at = datetime.now().isoformat()
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def load(workspace: Path) -> Optional["RunMetadata"]:
        path = workspace / "run_metadata.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            meta = RunMetadata(**{k: v for k, v in data.items()
                                  if k != "steps"})
            meta.steps = [StepRecord(**s) for s in data.get("steps", [])]
            return meta
        except Exception:
            return None


class RunTracker:
    """Track a pipeline run and save metadata when done."""

    def __init__(self, workspace: Path, mode: str = "narration"):
        self.meta = RunMetadata(
            run_id=workspace.name,
            started_at=datetime.now().isoformat(),
            mode=mode,
        )
        self.workspace = workspace
        self._start_time = time.time()

    def record_step(self, step_name: str, status: str, duration: float,
                    message: str, output_files: list[str] = None):
        self.meta.steps.append(StepRecord(
            name=step_name,
            status=status,
            duration_s=round(duration, 2),
            message=message,
            output_files=output_files or [],
        ))

    def finish(self, success: bool, output_video: str = "", error: str = ""):
        self.meta.total_duration_s = round(time.time() - self._start_time, 2)
        self.meta.success = success
        self.meta.output_video = output_video
        self.meta.error = error
        self.meta.save(self.workspace)
        return self.meta
