"""Pipeline step base class and shared types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
import subprocess
import shutil


class Status(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    status: Status
    output_files: list[Path] = field(default_factory=list)
    message: str = ""
    duration: float = 0.0
    metadata: dict = field(default_factory=dict)


class PipelineStep(ABC):
    """Base class for all pipeline steps."""

    id: str = ""
    name: str = ""
    description: str = ""
    icon: str = ""
    required: bool = True  # If True, check_deps failure = ERROR not SKIPPED

    def __init__(self):
        self.status = Status.PENDING
        self.result: StepResult | None = None
        self.log_lines: list[str] = []

    def log(self, msg: str):
        self.log_lines.append(msg)

    def execute(self, ctx: dict) -> StepResult:
        self.status = Status.RUNNING
        self.log_lines = []
        start = time.time()
        try:
            ok, reason = self.check_deps()
            if not ok:
                if self.required:
                    self.log(f"[錯誤] 必要工具缺失：{reason}")
                    self.status = Status.ERROR
                    self.result = StepResult(status=Status.ERROR, message=f"必要工具缺失：{reason}")
                else:
                    self.log(f"[略過] {reason}")
                    self.status = Status.SKIPPED
                    self.result = StepResult(status=Status.SKIPPED, message=reason)
                return self.result

            self.log(f"[開始] {self.name}")
            self.result = self.run(ctx)
            self.status = self.result.status
            self.log(f"[{self.result.status.value.upper()}] {self.result.message}")
        except Exception as e:
            self.status = Status.ERROR
            self.result = StepResult(status=Status.ERROR, message=str(e))
            self.log(f"[錯誤] {e}")
        finally:
            if self.result:
                self.result.duration = time.time() - start
        return self.result

    @abstractmethod
    def run(self, ctx: dict) -> StepResult:
        """Execute the step. ctx contains workspace path and prior step outputs."""
        ...

    @abstractmethod
    def check_deps(self) -> tuple[bool, str]:
        """Return (ok, message). If not ok, step will be skipped."""
        ...


def cmd_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run_cmd(args: list[str], cwd: str | None = None, timeout: int = 600) -> str:
    """執行 shell 指令並回傳標準輸出。"""
    result = subprocess.run(
        args, capture_output=True, text=True,
        cwd=cwd, timeout=timeout, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"指令執行失敗：{' '.join(args)}\n{result.stderr}")
    return result.stdout
