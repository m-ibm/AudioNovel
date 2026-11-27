"""Small dependency-free helpers for incremental audio fragment handling."""
from __future__ import annotations

import wave
from collections.abc import Iterable
from pathlib import Path


_OPENING_MARKS = frozenset("“‘«‹([{（「『")


def merge_punctuation_fragments(fragments: Iterable[str]) -> list[str]:
    """Keep punctuation-only tokens attached to speech, never as audio fragments.

    Leading formatting markers are discarded. Standalone opening quotation/
    bracket marks are carried forward to the next spoken fragment; other
    punctuation-only tokens are appended to the preceding spoken fragment.
    """
    merged: list[str] = []
    pending_prefix = ""

    for fragment in fragments:
        part = str(fragment or "").strip()
        if not part:
            continue

        if any(character.isalnum() for character in part):
            merged.append(pending_prefix + part)
            pending_prefix = ""
        elif all(character in _OPENING_MARKS for character in part):
            pending_prefix += part
        elif merged:
            # Closing quotes, ellipses, and standalone markup stay on the
            # preceding speech fragment, avoiding a zero-character alignment.
            merged[-1] += part
        # A leading non-opening punctuation/markup token has no spoken content
        # and is intentionally omitted.

    return merged


def write_silent_wav(
    output_path: Path | str,
    *,
    sample_rate: int = 24000,
    duration_seconds: float = 0.10,
) -> None:
    """Write a short mono PCM silence WAV for a row with no spoken content."""
    path = Path(output_path)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, round(sample_rate * duration_seconds))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)
