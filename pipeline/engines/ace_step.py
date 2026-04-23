"""ACE-Step 1.5 music generation engine.

Requires: git clone https://github.com/ace-step/ACE-Step-1.5
The model runs locally, ~4GB VRAM, generates music in seconds on GPU.
"""

import json
from pathlib import Path


# ACE-Step can run as:
# 1. Python module (if installed in same env)
# 2. Subprocess (separate Python env with its own deps)
# 3. HTTP API (if someone wraps it in a server)

# We use subprocess approach for isolation (like GPT-SoVITS)

ACE_STEP_DIRS = [
    Path("H:/dev/ACE-Step-1.5"),
    Path.home() / "ACE-Step-1.5",
    Path.home() / "ace-step",
]

# ACE-Step entry points to try (varies by version)
ENTRY_POINTS = [
    "acestep/acestep_v15_pipeline.py",
    "infer.py",
    "generate.py",
]

MAX_DURATION = 300  # ACE-Step practical limit


def find_ace_step() -> Path | None:
    """Find ACE-Step installation directory."""
    for d in ACE_STEP_DIRS:
        if not d.exists():
            continue
        for entry in ENTRY_POINTS:
            if (d / entry).exists():
                return d
    return None


def _find_entry(ace_dir: Path) -> str:
    """Find the correct entry point script."""
    for entry in ENTRY_POINTS:
        if (ace_dir / entry).exists():
            return entry
    raise FileNotFoundError(f"找不到 ACE-Step 入口腳本：{ENTRY_POINTS}")


def generate_music(
    prompt: str,
    output_path: Path,
    duration: int = 60,
    ace_step_dir: Path | None = None,
) -> Path:
    """Generate music using ACE-Step 1.5."""
    import subprocess

    ace_dir = ace_step_dir or find_ace_step()
    if not ace_dir:
        raise FileNotFoundError(
            "找不到 ACE-Step 1.5。安裝方式：\n"
            "  git clone https://github.com/ace-step/ACE-Step-1.5 H:/dev/ACE-Step-1.5\n"
            "  cd ACE-Step-1.5 && pip install -r requirements.txt"
        )

    # Cap duration to avoid OOM
    duration = min(duration, MAX_DURATION)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entry = _find_entry(ace_dir)

    result = subprocess.run(
        [
            "python", str(ace_dir / entry),
            "--prompt", prompt,
            "--output", str(output_path),
            "--duration", str(duration),
        ],
        capture_output=True, text=True, timeout=600,
        cwd=str(ace_dir),
        encoding="utf-8", errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(f"ACE-Step 生成失敗：{result.stderr[-300:]}")

    if not output_path.exists() or output_path.stat().st_size < 1000:
        raise RuntimeError("ACE-Step 輸出檔案為空")

    return output_path


def is_available() -> bool:
    """Check if ACE-Step is installed and accessible."""
    return find_ace_step() is not None
