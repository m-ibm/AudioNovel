#!/usr/bin/env python3
"""Assemble generated row WAVs into a chapter-marked M4B audiobook using ffmpeg."""
from __future__ import annotations
import argparse
import csv
import json
import posixpath
import shutil
import subprocess
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlsplit
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


def _local_name(tag: str) -> str:
    """Return an XML element's local name, ignoring its namespace."""
    return tag.rsplit("}", 1)[-1].lower()


def _archive_member(base_dir: str, href: str) -> str | None:
    """Resolve an EPUB-relative href safely to a ZIP member name."""
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc:
        return None
    relative = unquote(parsed.path).replace("\\", "/")
    if not relative:
        return None
    member = posixpath.normpath(posixpath.join(base_dir, relative.lstrip("/")))
    if member in ("", ".", "..") or member.startswith("../") or member.startswith("/"):
        return None
    return member


def _find_epub_cover_member(archive: zipfile.ZipFile) -> tuple[str, str] | None:
    """Find the cover image using EPUB 3, EPUB 2, and common legacy declarations."""
    container = ET.fromstring(archive.read("META-INF/container.xml"))
    rootfile = next(
        (node for node in container.iter()
         if _local_name(node.tag) == "rootfile" and node.attrib.get("full-path")),
        None,
    )
    if rootfile is None:
        raise ValueError("EPUB container.xml does not identify a package document")

    opf_path = posixpath.normpath(rootfile.attrib["full-path"])
    if opf_path.startswith("../") or opf_path.startswith("/"):
        raise ValueError("EPUB package document path is invalid")
    package = ET.fromstring(archive.read(opf_path))
    opf_dir = posixpath.dirname(opf_path)

    items_by_id: dict[str, tuple[str, str, str]] = {}
    items_by_member: dict[str, tuple[str, str, str]] = {}
    for node in package.iter():
        if _local_name(node.tag) != "item":
            continue
        item_id = node.attrib.get("id", "")
        href = node.attrib.get("href", "")
        media_type = node.attrib.get("media-type", "").lower()
        member = _archive_member(opf_dir, href)
        if not href or not member:
            continue
        item = (member, media_type, item_id)
        if item_id:
            items_by_id[item_id] = item
        items_by_member[member] = item

    candidates: list[tuple[str, str, str]] = []

    def add(item: tuple[str, str, str] | None) -> None:
        if item and item[1].startswith("image/") and item not in candidates:
            candidates.append(item)

    # EPUB 3: the cover image is identified by the cover-image manifest property.
    for item in items_by_id.values():
        node = next(
            (entry for entry in package.iter()
             if _local_name(entry.tag) == "item" and entry.attrib.get("id") == item[2]),
            None,
        )
        if node and "cover-image" in node.attrib.get("properties", "").split():
            add(item)

    # EPUB 2: metadata contains <meta name="cover" content="<manifest-item-id>"/>.
    for node in package.iter():
        if _local_name(node.tag) == "meta" and node.attrib.get("name", "").lower() == "cover":
            add(items_by_id.get(node.attrib.get("content", "")))

    # Some older EPUBs point to an XHTML cover page in <guide>; look for its image.
    for reference in package.iter():
        if _local_name(reference.tag) != "reference" or reference.attrib.get("type", "").lower() != "cover":
            continue
        href = reference.attrib.get("href", "")
        member = _archive_member(opf_dir, href)
        add(items_by_member.get(member or ""))
        if member and member in archive.namelist():
            try:
                page = ET.fromstring(archive.read(member))
            except ET.ParseError:
                continue
            page_dir = posixpath.dirname(member)
            for image_node in page.iter():
                node_name = _local_name(image_node.tag)
                image_href = image_node.attrib.get("src") if node_name == "img" else None
                if node_name == "image":
                    image_href = (image_node.attrib.get("href")
                                  or image_node.attrib.get("{http://www.w3.org/1999/xlink}href"))
                if image_href:
                    image_member = _archive_member(page_dir, image_href)
                    add(items_by_member.get(image_member or ""))

    # Compatibility fallback for EPUBs with incomplete/older cover declarations.
    for item in items_by_id.values():
        member, media_type, item_id = item
        filename = posixpath.basename(member).lower()
        if "cover" in item_id.lower() or "cover" in filename:
            add(item)

    for member, media_type, _ in candidates:
        if member in archive.namelist():
            return member, media_type
    return None


def extract_cover_image(epub_file: Path, output_path: Path) -> Path | None:
    """Extract an EPUB's declared cover image using only Python's standard library."""
    if epub_file.suffix.lower() != ".epub":
        print(f"WARNING: Expected an .epub input, got {epub_file}; assembling without cover art.", file=sys.stderr)
        return None
    try:
        with zipfile.ZipFile(epub_file) as archive:
            cover = _find_epub_cover_member(archive)
            if cover is None:
                print(f"WARNING: No cover image was found in {epub_file}; assembling without cover art.", file=sys.stderr)
                return None
            member, media_type = cover
            image_bytes = archive.read(member)
        suffix = Path(member).suffix
        if not suffix:
            suffix = {
                "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
                "image/gif": ".gif", "image/svg+xml": ".svg",
            }.get(media_type, ".img")
        destination = output_path.with_suffix(suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(image_bytes)
        return destination
    except (OSError, zipfile.BadZipFile, KeyError, ET.ParseError, ValueError) as exc:
        print(f"WARNING: Could not extract a cover from {epub_file}: {exc}; assembling without cover art.", file=sys.stderr)
        return None


def resolve_cover_image(cover_image: Path | None, epub_file: Path | None, temp_dir: Path) -> Path | None:
    """Resolve an explicit image or extract one from an EPUB; explicit images win exclusively."""
    if cover_image is not None and epub_file is not None:
        raise ValueError("Specify only one of --cover-image or --epub-file")
    if cover_image is not None:
        path = cover_image.expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Cover image does not exist or is empty: {path}")
        return path
    if epub_file is not None:
        epub = epub_file.expanduser().resolve()
        if not epub.is_file():
            raise FileNotFoundError(f"Source EPUB does not exist: {epub}")
        if epub.suffix.lower() != ".epub":
            raise ValueError(f"Expected an .epub input: {epub}")
        return extract_cover_image(epub, temp_dir / "book-cover.jpg")
    print("WARNING: No --cover-image or --epub-file supplied; assembling without cover art.", file=sys.stderr)
    return None

def resolve_cover_image(cover_image: Path | None, ebook_file: Path | None, temp_dir: Path) -> Path | None:
    """Resolve an explicit image or extract one from an e-book; explicit images win exclusively."""
    if cover_image is not None and ebook_file is not None:
        raise ValueError("Specify only one of --cover-image or --ebook-file")
    if cover_image is not None:
        path = cover_image.expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Cover image does not exist or is empty: {path}")
        return path
    if ebook_file is not None:
        ebook = ebook_file.expanduser().resolve()
        if not ebook.is_file():
            raise FileNotFoundError(f"Source e-book does not exist: {ebook}")
        return extract_cover_image(ebook, temp_dir / "book-cover.jpg")
    print("WARNING: No --cover-image or --ebook-file supplied; assembling without cover art.", file=sys.stderr)
    return None


def build_ffmpeg_command(concat_path: Path, metadata_path: Path, output: Path,
                         cover_image: Path | None = None) -> list[str]:
    """Build the final mux command, mapping cover art as an attached picture."""
    cmd = ["ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_path),
           "-f", "ffmetadata", "-i", str(metadata_path)]
    if cover_image is not None:
        cmd += ["-i", str(cover_image)]
    cmd += ["-map", "0:a:0", "-map_metadata", "1", "-map_chapters", "1"]
    if cover_image is not None:
        cmd += ["-map", "2:v:0", "-c:v", "copy", "-disposition:v:0", "attached_pic",
                "-metadata:s:v:0", "title=Cover", "-metadata:s:v:0", "comment=Cover (front)"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)]
    return cmd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--book-csv", type=Path, default=Path("Working_files/Book/book.csv"))
    p.add_argument("--clips-dir", type=Path, default=Path("Working_files/generated_audio_clips_incremental"))
    p.add_argument("--output-dir", type=Path, default=Path("output_audiobook"))
    p.add_argument("--title", default="The City & The City")
    p.add_argument("--author", default="China Miéville")
    cover_group = p.add_mutually_exclusive_group()
    cover_group.add_argument("--cover-image", type=Path, help="Image file to embed as M4B cover art")
    cover_group.add_argument("--epub-file", "--ebook-file", dest="epub_file", type=Path, help="Source EPUB; extract its cover using Python")
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
        cover_image = resolve_cover_image(a.cover_image, a.epub_file, tmp)
        cmd = build_ffmpeg_command(concat_path, metadata_path, output, cover_image)
        subprocess.run(cmd, check=True)
    print(f"Created: {output}")
    print(f"Rows: {len(df)}; chapters: {len(chapter_entries)}; inserted inter-row silence: {a.silence_ms} ms")
    return 0

if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
