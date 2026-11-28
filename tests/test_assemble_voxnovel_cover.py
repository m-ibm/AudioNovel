"""Tests for incremental M4B cover-art handling."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from assemble_voxnovel_incremental import (
    build_ffmpeg_command,
    extract_cover_image,
    resolve_cover_image,
)


class CoverArtTests(unittest.TestCase):
    def test_command_maps_cover_as_attached_picture(self):
        command = build_ffmpeg_command(
            Path("clips.txt"), Path("chapters.ffmeta"), Path("book.m4b"),
            Path("cover.jpg"),
        )
        self.assertEqual(command[command.index("-map") + 1], "0:a:0")
        self.assertIn("2:v:0", command)
        self.assertIn("attached_pic", command)
        self.assertIn("192k", command)
        self.assertEqual(command[-1], "book.m4b")

    def test_command_without_cover_keeps_audio_only(self):
        command = build_ffmpeg_command(
            Path("clips.txt"), Path("chapters.ffmeta"), Path("book.m4b")
        )
        self.assertNotIn("attached_pic", command)
        self.assertNotIn("2:v:0", command)
        self.assertEqual(command[command.index("-map") + 1], "0:a:0")

    def test_explicit_cover_is_used_without_calibre(self):
        with tempfile.TemporaryDirectory() as temp:
            cover = Path(temp) / "cover.png"
            cover.write_bytes(b"image")
            self.assertEqual(resolve_cover_image(cover, None, Path(temp)), cover)

    def test_missing_explicit_cover_is_an_error(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                resolve_cover_image(Path(temp) / "missing.jpg", None, Path(temp))

    def test_ebook_cover_is_extracted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ebook = root / "book.epub"
            ebook.write_bytes(b"epub")
            output = root / "cover.jpg"

            def fake_run(command, **kwargs):
                self.assertEqual(command[0], "ebook-meta")
                output.write_bytes(b"jpeg")
                class Result:
                    returncode = 0
                    stderr = ""
                return Result()

            with patch("assemble_voxnovel_incremental.shutil.which", return_value="/usr/bin/ebook-meta"), \
                 patch("assemble_voxnovel_incremental.subprocess.run", side_effect=fake_run):
                self.assertEqual(extract_cover_image(ebook, output), output)

    def test_failed_extraction_returns_no_cover(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ebook = root / "book.epub"
            ebook.write_bytes(b"epub")
            output = root / "cover.jpg"
            class Result:
                returncode = 1
                stderr = "no embedded cover"
            with patch("assemble_voxnovel_incremental.shutil.which", return_value="/usr/bin/ebook-meta"), \
                 patch("assemble_voxnovel_incremental.subprocess.run", return_value=Result()):
                self.assertIsNone(extract_cover_image(ebook, output))

    def test_both_cover_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cover = root / "cover.jpg"
            ebook = root / "book.epub"
            cover.write_bytes(b"image")
            ebook.write_bytes(b"epub")
            with self.assertRaises(ValueError):
                resolve_cover_image(cover, ebook, root)


if __name__ == "__main__":
    unittest.main()
