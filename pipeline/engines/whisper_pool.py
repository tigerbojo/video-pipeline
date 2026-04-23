"""Whisper model singleton pool — avoid reloading on every call."""

_cache: dict = {}


def get_model(model_name: str = "large-v3", device: str = "auto", compute_type: str = "auto"):
    """Get or create a cached WhisperModel instance."""
    from faster_whisper import WhisperModel

    # Auto-detect device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    key = (model_name, device, compute_type)
    if key not in _cache:
        try:
            _cache[key] = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception:
            # Fallback chain
            if device == "cuda":
                return get_model(model_name, "cpu", "int8")
            elif model_name != "base":
                return get_model("base", "cpu", "int8")
            else:
                raise

    return _cache[key]


def get_model_info() -> str:
    """Return info about cached models."""
    if not _cache:
        return "no models loaded"
    return ", ".join(f"{k[0]}({k[1]})" for k in _cache)
