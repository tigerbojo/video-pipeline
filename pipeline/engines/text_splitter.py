"""Split long text into segments at sentence boundaries for TTS."""

import re


def split_text(text: str, max_length: int = 300) -> list[str]:
    """Split text into chunks that fit within TTS model limits."""
    sentences = re.split(r'(?<=[。！？.!?\n])', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    segments: list[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) > max_length and current:
            segments.append(current)
            current = sentence
        else:
            current += sentence

    if current:
        segments.append(current)

    return segments if segments else [text]
