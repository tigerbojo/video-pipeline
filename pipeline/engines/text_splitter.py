"""Split long text into segments at sentence boundaries for TTS."""

import re


def split_text(text: str, max_length: int = 300) -> list[str]:
    """Split text into chunks that fit within TTS model limits.
    Handles Chinese punctuation including pause marks."""
    # Split on all Chinese/English sentence and clause boundaries
    sentences = re.split(r'(?<=[。！？.!?\n；;，,])', text)
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

    # Hard-split any remaining oversized segments
    final = []
    for seg in segments:
        if len(seg) <= max_length:
            final.append(seg)
        else:
            for i in range(0, len(seg), max_length):
                chunk = seg[i:i + max_length]
                if chunk.strip():
                    final.append(chunk)

    return final if final else [text]
