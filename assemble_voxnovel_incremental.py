#!/usr/bin/env python3
"""Assemble generated row WAVs into a chapter-marked M4B audiobook using ffmpeg."""
from __future__ import annotations
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import pandas as pd


def progress_line(stage: str, current: int, total: int, width: int = 28) -> str:
    """Return a stable, testable text progress bar."""
    fraction = min(1.0, max(0.0, current / total)) if total > 0 else 0.0
    filled = int(width * fraction)
    return f"{stage}: [{'#' * filled}{'-' * (width - filled)}] {fraction * 100:5.1f}% ({current}/{total})"


class Progress:
    """Write a compact progress bar to stderr, including in redirected logs."""
    def __init__(self, stage: str, total: int):
        self.stage, self.total = stage, max(1, total)
        self.last_reported = -1
        self.tty = sys.stderr.isatty()

    def update(self, current: int) -> None:
        current = min(current, self.total)
        interval = max(1, self.total // 20)
        if current != self.total and current - self.last_reported < interval:
            return
        line = progress_line(self.stage, current, self.total)
        print(line, file=sys.stderr, end="\r" if self.tty and current < self.total else "\n", flush=True)
        self.last_reported = current


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
    p.add_argument("--title", default="Audiobook", help="Audiobook title (default: Audiobook)")
    p.add_argument("--author", default="Unknown Author", help="Audiobook author (default: Unknown Author)")
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
    validation_progress = Progress("Validating clips", len(df))
    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
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
        validation_progress.update(row_number)
    if not clip_paths: raise RuntimeError("No clip rows in book.csv")
    _, sample_rate, channels = clip_metadata[clip_paths[0]]
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
        safe_title = re.sub(r"[^A-Za-z0-9._-]+", "_", a.title).strip("._") or "Audiobook"
        output = a.output_dir / f"{safe_title}.m4b"
        cmd = ["ffmpeg", "-hide_banner", "-y", "-progress", "pipe:1", "-nostats", "-f", "concat", "-safe", "0", "-i", str(concat_path),
               "-f", "ffmetadata", "-i", str(metadata_path), "-map", "0:a:0", "-map_metadata", "1", "-map_chapters", "1",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)]
        encoding_progress = Progress("Encoding audiobook", max(1, timeline_ms))
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            key, sep, value = line.strip().partition("=")
            if not sep:
                continue
            if key == "out_time_ms":
                try:
                    encoding_progress.update(int(value) // 1000)
                except ValueError:
                    pass
            elif key == "progress" and value == "end":
                encoding_progress.update(timeline_ms)
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd)
    print(f"Created: {output}")
    print(f"Rows: {len(df)}; chapters: {len(chapter_entries)}; inserted inter-row silence: {a.silence_ms} ms")
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
