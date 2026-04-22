"""Audio utilities: concat, convert, duration."""

import subprocess
import tempfile
from pathlib import Path


def concat_audio_files(file_paths: list[Path], output_path: Path) -> None:
    """Concatenate multiple audio files using FFmpeg."""
    if len(file_paths) == 1:
        import shutil
        shutil.copy2(file_paths[0], output_path)
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in file_paths:
            f.write(f"file '{p}'\n")
        list_file = f.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", list_file, "-c", "copy", str(output_path)],
            capture_output=True, check=True,
            encoding="utf-8", errors="replace",
        )
    finally:
        Path(list_file).unlink(missing_ok=True)


def convert_to_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "192k") -> None:
    """Convert WAV to MP3."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-b:a", bitrate, str(mp3_path)],
        capture_output=True, check=True,
        encoding="utf-8", errors="replace",
    )


def get_audio_duration(file_path: Path) -> float:
    """Get audio duration in seconds."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(file_path)],
        capture_output=True, text=True, check=True,
        encoding="utf-8", errors="replace",
    )
    return float(result.stdout.strip())
