#!/usr/bin/env python3
"""Assemble generated row WAVs into a chapter-marked M4B audiobook using ffmpeg."""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import pandas as pd


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.stdout.strip()


def probe_audio(path: Path) -> tuple[float, int, int]:
    raw = run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels", "-show_entries", "format=duration", "-of", "json", str(path)])
    data = json.loads(raw)
    streams = data.get("streams") or []
    if not streams: raise RuntimeError(f"No audio stream in {path}")
    duration = float((data.get("format") or {}).get("duration", 0))
    sr, channels = int(streams[0]["sample_rate"]), int(streams[0]["channels"])
    if duration <= 0: raise RuntimeError(f"Invalid audio duration in {path}: {duration}")
    return duration, sr, channels


def ffconcat_quote(path: Path) -> str:
    # concat demuxer single-quoted path escaping
    return "file '" + str(path.resolve()).replace("'", "'\\''") + "'\n"


def ffmetadata_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#").replace("\n", " ").replace("\r", " ")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--book-csv", type=Path, default=Path("Working_files/Book/book.csv"))
    p.add_argument("--clips-dir", type=Path, default=Path("Working_files/generated_audio_clips_incremental"))
    p.add_argument("--output-dir", type=Path, default=Path("output_audiobook"))
    p.add_argument("--title", default="The City & The City")
    p.add_argument("--author", default="China Miéville")
    p.add_argument("--silence-ms", type=int, default=750)
    p.add_argument("--allow-missing-manifest", action="store_true", help="Check files only; skip manifest state/hash verification")
    a = p.parse_args()
    for exe in ("ffmpeg", "ffprobe"):
        if shutil.which(exe) is None: raise RuntimeError(f"Required executable not found: {exe}")
    a.book_csv = a.book_csv.resolve(); a.clips_dir = a.clips_dir.resolve(); a.output_dir = a.output_dir.resolve()
    df = pd.read_csv(a.book_csv, keep_default_na=False)
    required = {"audio_id", "chapter", "chapter_title", "Text", "voice_actor"}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols: raise ValueError(f"book.csv missing columns: {missing_cols}")
    df = df.sort_values(["chapter", "Start Location", "audio_id"] if "Start Location" in df.columns else ["chapter", "audio_id"], kind="stable")
    manifest_path = a.clips_dir / "manifest.json"
    if not manifest_path.is_file() and not a.allow_missing_manifest:
        raise FileNotFoundError(f"Generation manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"rows": {}}
    manifest_rows = manifest.get("rows", {})
    clip_paths: list[Path] = []
    clip_metadata: dict[Path, tuple[float, int, int]] = {}
    for _, row in df.iterrows():
        aid = str(row.audio_id)
        path = a.clips_dir / (aid if aid.lower().endswith(".wav") else f"{aid}.wav")
        if not path.is_file(): raise FileNotFoundError(f"Missing generated clip {aid}: {path}")
        duration, sr, channels = probe_audio(path)
        clip_metadata[path] = (duration, sr, channels)
        state = manifest_rows.get(aid)
        if not a.allow_missing_manifest:
            if not state or state.get("status") != "generated": raise RuntimeError(f"Clip {aid} is not marked generated in manifest")
            import hashlib
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            if state.get("wav_sha256") != h: raise RuntimeError(f"Clip {aid} hash does not match manifest")
        if clip_paths:
            _, expected_sr, expected_ch = clip_metadata[clip_paths[0]]
            if (sr, channels) != (expected_sr, expected_ch): raise RuntimeError(f"Audio format mismatch in {path}")
        clip_paths.append(path)
    if not clip_paths: raise RuntimeError("No clip rows in book.csv")
    first_duration, sample_rate, channels = clip_metadata[clip_paths[0]]
    if channels not in (1, 2): raise RuntimeError(f"Only mono/stereo supported for silence insertion; first clip has {channels} channels")
    silence_seconds = max(0, a.silence_ms) / 1000.0
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="voxnovel_assemble_") as tmp_s:
        tmp = Path(tmp_s)
        silence_path = tmp / "silence.wav"
        concat_path = tmp / "all_clips.txt"
        metadata_path = tmp / "chapters.ffmeta"
        layout = "mono" if channels == 1 else "stereo"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             f"anullsrc=channel_layout={layout}:sample_rate={sample_rate}", "-t", f"{silence_seconds:.3f}",
             "-c:a", "pcm_s16le", str(silence_path)])

        lines: list[str] = []
        chapter_entries: list[tuple[int, str, int, int]] = []
        timeline_ms = 0
        groups = list(df.groupby("chapter", sort=True))
        processed_rows = 0
        total_rows = len(df)
        for gidx, (chapter, group) in enumerate(groups):
            chapter_start = timeline_ms
            chapter_title_values = [str(x).strip() for x in group.chapter_title.unique() if str(x).strip()]
            chapter_title = chapter_title_values[0] if chapter_title_values else f"Chapter {chapter}"
            records = list(group.iterrows())
            for pos, (_, row) in enumerate(records):
                aid = str(row.audio_id)
                clip = a.clips_dir / (aid if aid.lower().endswith(".wav") else f"{aid}.wav")
                lines.append(ffconcat_quote(clip))
                dur, _, _ = clip_metadata[clip]
                timeline_ms += round(dur * 1000)
                processed_rows += 1
                if processed_rows < total_rows and silence_seconds > 0:
                    # Keep the same pause between chapter-final and next-chapter
                    # clips as between rows within one chapter.
                    lines.append(ffconcat_quote(silence_path))
                    timeline_ms += round(silence_seconds * 1000)
            chapter_entries.append((int(chapter), chapter_title, chapter_start, timeline_ms))
        concat_path.write_text("".join(lines), encoding="utf-8")
        metadata_lines = [";FFMETADATA1", f"title={ffmetadata_escape(a.title)}", f"artist={ffmetadata_escape(a.author)}"]
        for ch, title, start, end in chapter_entries:
            metadata_lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={start}", f"END={end}", f"title={ffmetadata_escape(f'Chapter {ch}: {title}')}" ]
        metadata_path.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
        output = a.output_dir / "The_City_and_The_City.m4b"
        cmd = ["ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_path),
               "-f", "ffmetadata", "-i", str(metadata_path), "-map", "0:a:0", "-map_metadata", "1", "-map_chapters", "1",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)]
        subprocess.run(cmd, check=True)
    print(f"Created: {output}")
    print(f"Rows: {len(df)}; chapters: {len(chapter_entries)}; inserted inter-row silence: {a.silence_ms} ms")
    return 0

if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
