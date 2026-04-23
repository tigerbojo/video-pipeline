"""Batch processing: queue multiple videos, resume interrupted jobs."""

import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str = ""
    video_path: str = ""
    mode: str = "narration"  # "narration" or "subtitle"
    status: JobStatus = JobStatus.QUEUED
    progress: str = ""
    result_path: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    settings: dict = field(default_factory=dict)


class BatchQueue:
    """Persistent job queue backed by a JSON file."""

    def __init__(self, queue_file: Path):
        self.queue_file = queue_file
        self.jobs: list[Job] = []
        self._cancelled: set[str] = set()
        self._load()

    def _load(self):
        if self.queue_file.exists():
            try:
                data = json.loads(self.queue_file.read_text(encoding="utf-8"))
                self.jobs = [Job(**j) for j in data]
            except Exception:
                self.jobs = []

    def _save(self):
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(j) for j in self.jobs]
        self.queue_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, video_path: str, mode: str = "narration", settings: dict = None) -> Job:
        job = Job(
            id=f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.jobs)}",
            video_path=video_path,
            mode=mode,
            settings=settings or {},
        )
        self.jobs.append(job)
        self._save()
        return job

    def cancel(self, job_id: str) -> bool:
        for j in self.jobs:
            if j.id == job_id and j.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                j.status = JobStatus.CANCELLED
                j.finished_at = datetime.now().isoformat()
                self._cancelled.add(job_id)
                self._save()
                return True
        return False

    def is_cancelled(self, job_id: str) -> bool:
        return job_id in self._cancelled

    def get_pending(self) -> list[Job]:
        return [j for j in self.jobs if j.status == JobStatus.QUEUED]

    def get_resumable(self) -> list[Job]:
        """Jobs that were RUNNING when interrupted (can resume)."""
        return [j for j in self.jobs if j.status == JobStatus.RUNNING]

    def update(self, job_id: str, **kwargs):
        for j in self.jobs:
            if j.id == job_id:
                for k, v in kwargs.items():
                    if hasattr(j, k):
                        setattr(j, k, v)
                self._save()
                return

    def summary(self) -> dict:
        counts = {}
        for j in self.jobs:
            counts[j.status] = counts.get(j.status, 0) + 1
        return counts

    def clear_done(self):
        self.jobs = [j for j in self.jobs if j.status not in (JobStatus.DONE, JobStatus.CANCELLED)]
        self._save()
