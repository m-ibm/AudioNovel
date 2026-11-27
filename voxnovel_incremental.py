#!/usr/bin/env python3
"""Incremental XTTS audiobook renderer for a prepared VoxNovel book.csv.

- Deterministic per-row Speaker/voice_actor assignment comes from book.csv.
- Existing clips are skipped only when a manifest fingerprint and WAV hash match.
- Fragments shorter than --min-pad-words are padded with the tested continuation,
  force-aligned with wav2vec2 CTC, then cropped just after the source speech.
- A failed alignment prevents the padded clip from becoming final output; its raw
  audio is preserved under OUTPUT/.work/ for inspection.
- Every completed row is committed immediately to manifest.json, so reruns resume.

Run from the VoxNovel project root. This does not import headless_voxnovel.py and
never clears the clip/output directories.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from voxnovel_fragment_utils import merge_punctuation_fragments, write_silent_wav

from voxnovel_xtts_backend import (
    VoxNovelTTSBackend,
    choose_device,
    configure_cpu_threads,
    ensure_nltk_punkt,
    sent_tokenize,
    split_long_sentence,
)

PADDING_TEXT = "I am continuing to speak clearly for the rest of this recording."
CROP_TAIL_SECONDS = 0.22
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*(?:-[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*)*")
CHAPTER_MARKERS = ("NEWCHAPTERABC",)
MANIFEST_VERSION = 2

@dataclass
class AlignmentResult:
    crop_seconds: float
    audio_seconds: float
    mean_char_posterior: float
    aligned_source_chars: int
    source_chars: int
    padding_end_seconds: float


class EnglishCTCAligner:
    """Forced-align known English text with torchaudio's wav2vec2 CTC model."""

    def __init__(self, device: torch.device) -> None:
        pipelines = getattr(torchaudio, "pipelines", None)
        bundle = getattr(pipelines, "WAV2VEC2_ASR_BASE_960H", None) if pipelines else None
        if bundle is None:
            raise RuntimeError(
                "This torchaudio build does not expose WAV2VEC2_ASR_BASE_960H. "
                "The padding variants require this English CTC alignment model. "
                "Do not install WhisperX into the TTS environment as a substitute; "
                "that may change its torch/torchaudio dependencies."
            )
        self.device = device
        print("Loading torchaudio WAV2VEC2_ASR_BASE_960H forced-alignment model...", flush=True)
        self.model = bundle.get_model().to(device).eval()
        self.labels = list(bundle.get_labels())
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        self.blank_id = self.label_to_id.get("-")
        if self.blank_id is None:
            raise RuntimeError(f"Could not identify CTC blank label in labels: {self.labels!r}")
        if "|" not in self.label_to_id:
            raise RuntimeError(f"Could not identify CTC word-boundary label in labels: {self.labels!r}")
        self.sample_rate = int(bundle.sample_rate)

    def _normalize(self, text: str) -> list[str]:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = text.replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-")
        text = text.upper()
        # Apostrophes are phonetic only inside words (don't, I'm). Leading or
        # trailing quotation/apostrophe marks are punctuation, not spoken.
        text = re.sub(r"(?<![A-Z0-9])'(?![A-Z0-9])", " ", text)
        result: list[str] = []
        for ch in text:
            if ch in self.label_to_id and ch != "-":
                result.append(ch)
            elif ch.isspace() or ch in "-‐‑–—.,!?;:()[]{}\"“”/\\&%$£€":
                if result and result[-1] != "|":
                    result.append("|")
            elif ch.isalpha() or ch.isdigit():
                # Never silently delete a letter/number from the transcript: that
                # would make a forced-alignment crop unsafe.
                raise ValueError(f"CTC aligner cannot represent source character {ch!r} in {text!r}")
        while result and result[-1] == "|":
            result.pop()
        while result and result[0] == "|":
            result.pop(0)
        return result

    def _load_audio_16k(self, wav_path: Path) -> tuple[torch.Tensor, float]:
        waveform, sr = torchaudio.load(str(wav_path))
        if waveform.ndim != 2 or waveform.shape[1] == 0:
            raise RuntimeError(f"Empty or invalid WAV: {wav_path}")
        waveform = waveform.mean(dim=0, keepdim=True)
        duration = waveform.shape[1] / float(sr)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return waveform.to(self.device), duration

    @staticmethod
    def _viterbi(log_probs: np.ndarray, token_ids: list[int], blank_id: int) -> np.ndarray:
        """Return the CTC best state path for an exact target token sequence."""
        frames = int(log_probs.shape[0])
        if not token_ids:
            raise ValueError("Target text contains no alignable English characters")
        states = np.full(2 * len(token_ids) + 1, blank_id, dtype=np.int64)
        states[1::2] = np.asarray(token_ids, dtype=np.int64)
        state_count = len(states)
        if frames < len(token_ids):
            raise ValueError(
                f"Audio has only {frames} CTC frames for {len(token_ids)} target characters"
            )

        neg_inf = -1.0e30
        prev = np.full(state_count, neg_inf, dtype=np.float32)
        back = np.zeros((frames, state_count), dtype=np.int8)
        prev[0] = log_probs[0, blank_id]
        if state_count > 1:
            prev[1] = log_probs[0, states[1]]

        for t in range(1, frames):
            cur = np.full(state_count, neg_inf, dtype=np.float32)
            for s in range(state_count):
                best_score = prev[s]
                best_step = 0
                if s >= 1 and prev[s - 1] > best_score:
                    best_score = prev[s - 1]
                    best_step = 1
                if (
                    s >= 2
                    and states[s] != blank_id
                    and states[s] != states[s - 2]
                    and prev[s - 2] > best_score
                ):
                    best_score = prev[s - 2]
                    best_step = 2
                cur[s] = best_score + log_probs[t, states[s]]
                back[t, s] = best_step
            prev = cur

        if state_count == 1:
            state = 0
        else:
            # A CTC sequence may finish on the final label or its following blank.
            state = state_count - 1 if prev[state_count - 1] >= prev[state_count - 2] else state_count - 2
        path = np.zeros(frames, dtype=np.int32)
        path[-1] = state
        for t in range(frames - 1, 0, -1):
            state -= int(back[t, state])
            path[t - 1] = state
        return path

    def align_source_end(self, wav_path: Path, source_text: str, full_text: str) -> AlignmentResult:
        source_chars = self._normalize(source_text)
        full_chars = self._normalize(full_text)
        source_last_char = max((i for i, ch in enumerate(source_chars) if ch != "|"), default=-1)
        if source_last_char < 0:
            raise ValueError(
                f"Source fragment has no alignable characters (punctuation-only): {source_text!r}"
            )

        # Preserve a word boundary between source and padding.
        if source_chars and full_chars[:len(source_chars)] == source_chars:
            combined_chars = list(full_chars)
        else:
            combined_chars = list(source_chars)
            if combined_chars and combined_chars[-1] != "|":
                combined_chars.append("|")
            pad_chars = self._normalize(PADDING_TEXT)
            if pad_chars and combined_chars and pad_chars[0] == "|":
                pad_chars = pad_chars[1:]
            combined_chars.extend(pad_chars)

        # The source prefix is exactly its normalized character sequence.
        if combined_chars[:len(source_chars)] != source_chars:
            raise RuntimeError("Internal error while constructing source+padding alignment transcript")

        unknown = [ch for ch in combined_chars if ch not in self.label_to_id]
        if unknown:
            raise ValueError(f"CTC aligner does not support character(s): {sorted(set(unknown))}")
        ids = [self.label_to_id[ch] for ch in combined_chars]

        waveform, audio_seconds = self._load_audio_16k(wav_path)
        with torch.inference_mode():
            emissions, lengths = self.model(waveform)
        frame_count = int(lengths[0].item()) if lengths is not None else int(emissions.shape[1])
        emission = emissions[0, :frame_count].float().log_softmax(dim=-1).cpu().numpy()
        path = self._viterbi(emission, ids, self.blank_id)

        # Each target character occupies state 1, 3, 5, ... in the expanded CTC path.
        source_frame_ends: list[int] = []
        source_scores: list[float] = []
        for char_index, ch in enumerate(source_chars):
            if ch == "|":
                continue
            state_index = 2 * char_index + 1
            frames = np.flatnonzero(path == state_index)
            if frames.size == 0:
                continue
            source_frame_ends.append(int(frames[-1]))
            source_scores.extend(emission[frames, self.label_to_id[ch]].tolist())

        aligned_count = len(source_frame_ends)
        source_char_count = sum(ch != "|" for ch in source_chars)
        if aligned_count != source_char_count:
            raise RuntimeError(
                f"Forced alignment placed {aligned_count}/{source_char_count} source characters; refusing to crop"
            )
        last_frame = max(source_frame_ends)
        frame_seconds = audio_seconds / float(frame_count)
        source_end = (last_frame + 1) * frame_seconds
        crop_seconds = min(audio_seconds, source_end + CROP_TAIL_SECONDS)
        if crop_seconds <= 0 or crop_seconds >= audio_seconds - 0.03:
            raise RuntimeError(
                f"Forced alignment did not leave a distinct padding tail (crop={crop_seconds:.3f}s, "
                f"audio={audio_seconds:.3f}s); refusing to crop"
            )
        mean_posterior = float(np.exp(np.mean(source_scores))) if source_scores else 0.0
        pad_end = audio_seconds - source_end
        return AlignmentResult(
            crop_seconds=crop_seconds,
            audio_seconds=audio_seconds,
            mean_char_posterior=mean_posterior,
            aligned_source_chars=aligned_count,
            source_chars=source_char_count,
            padding_end_seconds=pad_end,
        )

def count_words(text: str) -> int:
    return len(WORD_RE.findall(str(text or "")))


def clean_text(text: str) -> str:
    value = str(text or "")
    for marker in CHAPTER_MARKERS:
        value = value.replace(marker, "")
    return value.strip()


def safe_audio_filename(audio_id: str) -> str:
    if Path(audio_id).name != audio_id or audio_id in {"", ".", ".."}:
        raise ValueError(f"Unsafe audio_id: {audio_id!r}")
    return audio_id if audio_id.lower().endswith(".wav") else f"{audio_id}.wav"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wav_info(path: Path) -> tuple[int, int, int]:
    """Read basic PCM WAV metadata without relying on deprecated torchaudio.info."""
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes(), handle.getframerate(), handle.getnchannels()


def valid_wav(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 44:
        return False
    try:
        frames, sample_rate, channels = wav_info(path)
        return frames > 0 and sample_rate > 0 and channels in (1, 2)
    except Exception:
        return False


def wav_duration(path: Path) -> float:
    frames, sample_rate, _ = wav_info(path)
    if frames <= 0 or sample_rate <= 0:
        raise RuntimeError(f"Invalid WAV duration metadata: {path}")
    return float(frames) / float(sample_rate)


def row_fragments(text: str) -> list[str]:
    fragments: list[str] = []
    for sentence in sent_tokenize(text):
        fragments.extend(split_long_sentence(sentence))
    # Sentence splitting can emit trailing quote/markup tokens as standalone
    # fragments. Keep punctuation with speech instead of padding/alignment on it.
    return merge_punctuation_fragments(fragments)


def hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"manifest_version": MANIFEST_VERSION, "rows": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("manifest_version") != MANIFEST_VERSION or not isinstance(value.get("rows"), dict):
        raise ValueError(f"Unsupported or malformed generation manifest: {path}")
    return value


def write_csv_record(path: Path, fields: list[str], record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(record)
        handle.flush()
        os.fsync(handle.fileno())


def combine_wavs(paths: list[Path], output_path: Path) -> None:
    if not paths:
        raise RuntimeError("No generated fragments for this row")
    tensors: list[torch.Tensor] = []
    sample_rate: int | None = None
    channels: int | None = None
    for path in paths:
        if not valid_wav(path):
            raise RuntimeError(f"Generated fragment is not a valid WAV: {path}")
        waveform, rate = torchaudio.load(str(path))
        if sample_rate is None:
            sample_rate, channels = int(rate), int(waveform.shape[0])
        elif int(rate) != sample_rate or int(waveform.shape[0]) != channels:
            raise RuntimeError(f"Inconsistent fragment audio format: {path}")
        tensors.append(waveform)
    joined = torch.cat(tensors, dim=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial.wav")
    partial.unlink(missing_ok=True)
    try:
        torchaudio.save(str(partial), joined, int(sample_rate))
        if not valid_wav(partial):
            raise RuntimeError(f"Combined clip failed WAV validation: {partial}")
        os.replace(partial, output_path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def align_and_crop(
    backend: VoxNovelTTSBackend,
    aligner: EnglishCTCAligner,
    source_text: str,
    actor: str,
    language: str,
    raw_path: Path,
    cropped_path: Path,
    audio_id: str,
    fragment_number: int,
    audit_path: Path,
) -> bool:
    # A few short source fragments contain numerals (e.g. "1926" or "P38").
    # The English CTC label set has no digit tokens, so do not drop those
    # characters or fail the full row: synthesize that fragment normally and
    # record that padding/alignment was bypassed for this specific fragment.
    try:
        aligner._normalize(source_text)
    except ValueError as exc:
        backend.generate_fragment(source_text, cropped_path, actor, language)
        if not valid_wav(cropped_path):
            raise RuntimeError(f"Normal-synthesis fallback returned an invalid fragment: {cropped_path}")
        write_csv_record(
            audit_path,
            ["audio_id", "fragment_number", "source_fragment", "synthesis_text", "alignment_status",
             "raw_duration_seconds", "crop_duration_seconds", "mean_char_posterior", "aligned_source_chars",
             "source_chars", "padding_after_source_seconds", "raw_audio_path", "raw_audio_retained", "error"],
            {"audio_id": audio_id, "fragment_number": fragment_number, "source_fragment": source_text,
             "synthesis_text": source_text, "alignment_status": "NORMAL_FALLBACK_UNSUPPORTED_SOURCE_CHAR",
             "raw_duration_seconds": "", "crop_duration_seconds": f"{wav_duration(cropped_path):.3f}",
             "mean_char_posterior": "", "aligned_source_chars": "", "source_chars": "",
             "padding_after_source_seconds": "", "raw_audio_path": "", "raw_audio_retained": "no",
             "error": str(exc)},
        )
        return False

    synthesis_text = source_text.rstrip() + " " + PADDING_TEXT
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    # split_sentences=False is essential: the model must synthesize source+padding
    # in one continuous call for the source/padding boundary to be alignable.
    backend.generate_fragment(
        synthesis_text, raw_path, actor, language, split_sentences=False
    )
    aligned = aligner.align_source_end(raw_path, source_text, synthesis_text)
    waveform, sample_rate = torchaudio.load(str(raw_path))
    keep_frames = max(1, min(waveform.shape[1], round(aligned.crop_seconds * sample_rate)))
    torchaudio.save(str(cropped_path), waveform[:, :keep_frames], sample_rate)
    if not valid_wav(cropped_path):
        raise RuntimeError(f"Cropped fragment failed WAV validation: {cropped_path}")
    write_csv_record(
        audit_path,
        ["audio_id", "fragment_number", "source_fragment", "synthesis_text", "alignment_status",
         "raw_duration_seconds", "crop_duration_seconds", "mean_char_posterior", "aligned_source_chars",
         "source_chars", "padding_after_source_seconds", "raw_audio_path", "raw_audio_retained", "error"],
        {"audio_id": audio_id, "fragment_number": fragment_number, "source_fragment": source_text,
         "synthesis_text": synthesis_text, "alignment_status": "OK",
         "raw_duration_seconds": f"{aligned.audio_seconds:.3f}",
         "crop_duration_seconds": f"{keep_frames / sample_rate:.3f}",
         "mean_char_posterior": f"{aligned.mean_char_posterior:.4f}",
         "aligned_source_chars": aligned.aligned_source_chars, "source_chars": aligned.source_chars,
         "padding_after_source_seconds": f"{aligned.padding_end_seconds:.3f}",
         "raw_audio_path": "", "raw_audio_retained": "no (deleted after successful crop)", "error": ""},
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-csv", type=Path, default=Path("Working_files/Book/book.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("Working_files/generated_audio_clips_incremental"))
    parser.add_argument("--model", default=os.environ.get("VOXNOVEL_TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"))
    parser.add_argument("--language", default="", help="Fallback language if book.csv language is empty (default: en)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
                        help="Use MPS only from native macOS Python; Docker/Colima does not expose Apple MPS.")
    parser.add_argument("--min-pad-words", type=int, default=10,
                        help="Pad each generated fragment with fewer than this many words (default: 10)")
    parser.add_argument("--no-padding", action="store_true", help="Disable the short-fragment padding/cropping repair")
    parser.add_argument("--max-rows", type=int, default=0, help="Test only the first N rows that need generation")
    parser.add_argument("--only-audio-ids", default="", help="Optional comma-separated audio_id allowlist for a targeted smoke test")
    parser.add_argument("--preload-aligner-only", action="store_true", help="Download/load the CTC aligner model, then exit")
    args = parser.parse_args()

    args.book_csv = args.book_csv.resolve()
    args.output_dir = args.output_dir.resolve()
    project_root = Path.cwd().resolve()
    if not args.book_csv.is_file():
        raise FileNotFoundError(f"book.csv not found: {args.book_csv}")
    if args.min_pad_words < 1:
        raise ValueError("--min-pad-words must be at least 1")

    df = pd.read_csv(args.book_csv, keep_default_na=False)
    required = {"Text", "Speaker", "audio_id", "voice_actor", "language"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"book.csv missing required columns: {missing}")
    if df.empty:
        raise ValueError("book.csv contains no rows")
    if df.audio_id.astype(str).duplicated().any():
        raise ValueError("book.csv contains duplicate audio_id values")
    if args.only_audio_ids.strip():
        wanted = {value.strip() for value in args.only_audio_ids.split(",") if value.strip()}
        found = set(df.audio_id.astype(str)) & wanted
        missing_ids = sorted(wanted - found)
        if missing_ids:
            raise ValueError("Unknown --only-audio-ids: " + ", ".join(missing_ids))
        df = df[df.audio_id.astype(str).isin(wanted)].copy()
    if not args.no_padding and args.model.lower().find("xtts") < 0:
        raise ValueError("Short-clip padding/cropping is implemented only for XTTS models")

    if args.device == "auto":
        device = choose_device()
    elif args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
        device = torch.device("cuda")
    else:
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("--device mps requested, but torch.backends.mps.is_available() is false")
        device = torch.device("mps")

    configure_cpu_threads()
    ensure_nltk_punkt()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    manifest_rows: dict[str, Any] = manifest["rows"]
    work_root = args.output_dir / ".work"
    stale_root = args.output_dir / ".stale"
    audit_path = args.output_dir / "padding_alignment.csv"
    errors_path = args.output_dir / "generation_errors.csv"

    if args.preload_aligner_only:
        if args.no_padding:
            raise ValueError("--preload-aligner-only cannot be combined with --no-padding")
        print(f"Loading CTC aligner on {device}; first load may download its checkpoint.", flush=True)
        EnglishCTCAligner(device)
        print("CTC aligner loaded successfully.", flush=True)
        return 0

    # Validate actors before loading the expensive TTS model.
    actors = sorted(set(df.voice_actor.astype(str).str.strip()))
    actors = [a for a in actors if a]
    if not actors:
        raise ValueError("No voice_actor values in book.csv")
    probe = VoxNovelTTSBackend(args.model, device, project_root)
    ref_hashes: dict[str, list[dict[str, str]]] = {}
    missing_voices: list[str] = []
    for actor in actors:
        try:
            refs = probe.reference_files(actor)
            if not refs:
                missing_voices.append(actor)
                continue
            ref_hashes[actor] = [{"path": Path(r).name, "sha256": sha256_file(Path(r))} for r in refs]
        except Exception:
            missing_voices.append(actor)
    if missing_voices:
        raise FileNotFoundError(
            "Missing voice sample(s) for: " + ", ".join(missing_voices) +
            ". Each voice_actor must have at least one .wav/.mp3 under tortoise/voices/<voice_actor>/."
        )
    del probe

    aligner: EnglishCTCAligner | None = None
    backend = VoxNovelTTSBackend(args.model, device, project_root)
    queue: list[tuple[int, pd.Series, Path, str, str, list[str]]] = []
    valid_skipped = 0
    stale_clips = 0
    for index, row in df.iterrows():
        audio_id = str(row.audio_id).strip()
        filename = safe_audio_filename(audio_id)
        output_path = args.output_dir / filename
        text = clean_text(row.Text)
        actor = str(row.voice_actor).strip()
        language = str(row.language).strip() or args.language or "en"
        # Rows containing only punctuation/formatting (or blank text) have no
        # speech to synthesize. They still receive a short silent WAV below so
        # every book.csv audio_id retains a corresponding output file.
        fragments = row_fragments(text)
        payload = {
            "audio_id": audio_id, "text": text, "voice_actor": actor, "language": language,
            "model": args.model, "device": str(device), "padding_enabled": not args.no_padding,
            "min_pad_words": args.min_pad_words, "padding_text": PADDING_TEXT if not args.no_padding else "",
            "crop_tail_seconds": CROP_TAIL_SECONDS if not args.no_padding else None,
            "reference_files": ref_hashes[actor],
        }
        fingerprint = hash_payload(payload)
        old = manifest_rows.get(audio_id, {})
        if (old.get("status") == "generated" and old.get("fingerprint") == fingerprint
                and output_path.is_file() and valid_wav(output_path)
                and old.get("wav_sha256") == sha256_file(output_path)):
            valid_skipped += 1
            continue
        if output_path.exists():
            stale_clips += 1
            stale_root.mkdir(parents=True, exist_ok=True)
            archive = stale_root / f"{int(time.time())}_{filename}"
            os.replace(output_path, archive)
        queue.append((index, row, output_path, fingerprint, language, fragments))

    print(f"Rows in book.csv: {len(df)}")
    print(f"Valid manifest-matched WAVs skipped: {valid_skipped}")
    print(f"Existing untracked/stale WAVs moved aside: {stale_clips}")
    print(f"Rows queued for generation: {len(queue)}")
    print(f"TTS model: {args.model}; device: {device}; padding: {'off' if args.no_padding else f'<{args.min_pad_words} words'}")
    if args.max_rows > 0:
        queue = queue[:args.max_rows]
        print(f"Test limit enabled: generating at most {len(queue)} rows")
    if not queue:
        print("Nothing to generate.")
        return 0

    # Load CTC alignment only if at least one queued fragment actually needs padding.
    needs_aligner = not args.no_padding and any(
        count_words(fragment) < args.min_pad_words for _, _, _, _, _, fragments in queue for fragment in fragments
    )
    if needs_aligner:
        print("Loading English wav2vec2 CTC aligner for padded fragments...", flush=True)
        aligner = EnglishCTCAligner(device)
    backend.prepare_special_voice_lists(set(actors))

    errors = 0
    started = time.time()
    for index, row, output_path, fingerprint, language, fragments in tqdm(queue, desc="Rendering clips"):
        audio_id = str(row.audio_id).strip()
        actor = str(row.voice_actor).strip()
        row_work = work_root / audio_id
        if row_work.exists():
            # A previous failed/incomplete attempt is retained; use a new attempt folder.
            row_work = work_root / f"{audio_id}_{int(time.time())}"
        row_work.mkdir(parents=True, exist_ok=True)
        fragment_paths: list[Path] = []
        padded_count = 0
        unpadded_fallback_count = 0
        try:
            for number, fragment in enumerate(fragments, start=1):
                fragment_path = row_work / f"part_{number:03d}.wav"
                if not args.no_padding and count_words(fragment) < args.min_pad_words:
                    assert aligner is not None
                    raw_path = row_work / "untrimmed" / f"part_{number:03d}_padded.wav"
                    was_padded = align_and_crop(backend, aligner, fragment, actor, language, raw_path,
                                                 fragment_path, audio_id, number, audit_path)
                    if was_padded:
                        padded_count += 1
                    else:
                        unpadded_fallback_count += 1
                else:
                    backend.generate_fragment(fragment, fragment_path, actor, language)
                    if not valid_wav(fragment_path):
                        raise RuntimeError(f"TTS returned an invalid fragment: {fragment_path}")
                fragment_paths.append(fragment_path)
            if fragment_paths:
                combine_wavs(fragment_paths, output_path)
            else:
                # Preserve the one-WAV-per-row contract without asking XTTS to
                # pronounce formatting markers or an empty transcript.
                write_silent_wav(output_path)
            if not valid_wav(output_path):
                raise RuntimeError(f"Generated row WAV failed validation: {output_path}")
            clip_hash = sha256_file(output_path)
            entry = {
                "status": "generated", "fingerprint": fingerprint, "wav_sha256": clip_hash,
                "file": output_path.name, "duration_seconds": round(wav_duration(output_path), 3),
                "voice_actor": actor, "language": language, "text": clean_text(row.Text),
                "fragments": len(fragments), "padded_fragments": padded_count,
                "unpadded_fallback_fragments": unpadded_fallback_count,
                "silent_placeholder": not bool(fragments),
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "model": args.model, "device": str(device),
            }
            manifest_rows[audio_id] = entry
            atomic_write_json(manifest_path, manifest)
            shutil.rmtree(row_work, ignore_errors=True)
        except Exception as exc:
            errors += 1
            manifest_rows[audio_id] = {
                "status": "failed", "fingerprint": fingerprint, "voice_actor": actor,
                "language": language, "text": clean_text(row.Text), "error": str(exc),
                "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "work_directory": str(row_work),
                "model": args.model, "device": str(device),
            }
            atomic_write_json(manifest_path, manifest)
            write_csv_record(errors_path, ["audio_id", "csv_row", "voice_actor", "error", "work_directory"],
                             {"audio_id": audio_id, "csv_row": index, "voice_actor": actor,
                              "error": repr(exc), "work_directory": str(row_work)})
            print(f"\nFAILED {audio_id} ({actor}): {exc}", file=sys.stderr, flush=True)
            # Continue to later rows. Rerunning the command retries every failed row.

    elapsed = time.time() - started
    generated_now = sum(1 for item in manifest_rows.values() if item.get("status") == "generated")
    failed_total = sum(1 for item in manifest_rows.values() if item.get("status") == "failed")
    print(f"\nRun complete in {elapsed:.1f}s. Newly failed rows: {errors}; failed rows in manifest: {failed_total}.")
    print(f"Generated/manifest-complete clips: {generated_now}; skipped valid clips this run: {valid_skipped}.")
    print(f"Clips: {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    if errors:
        print("Rerun the same command to retry failed rows; completed rows remain skipped.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Each completed clip and manifest entry is already saved; rerun to continue.", file=sys.stderr)
        raise SystemExit(130)
