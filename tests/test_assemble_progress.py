"""Regression tests for assembly progress reporting."""
import csv
import hashlib
import io
import json
import subprocess
import sys
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from assemble_voxnovel_incremental import Progress, progress_line


class ProgressFormattingTests(unittest.TestCase):
    def test_zero_progress(self):
        self.assertEqual(progress_line("Test", 0, 10), "Test: [----------------------------]   0.0% (0/10)")

    def test_half_progress(self):
        self.assertEqual(progress_line("Test", 5, 10), "Test: [##############--------------]  50.0% (5/10)")

    def test_completion(self):
        self.assertTrue(progress_line("Test", 10, 10).endswith("100.0% (10/10)"))

    def test_clamps_overflow(self):
        self.assertTrue(progress_line("Test", 12, 10).endswith("100.0% (12/10)"))

    def test_zero_total_is_safe(self):
        self.assertTrue(progress_line("Test", 0, 0).endswith("0.0% (0/0)"))

    def test_non_tty_progress_is_logged(self):
        stream = io.StringIO()
        with patch("sys.stderr", stream):
            progress = Progress("Checking", 4)
            progress.update(1)
            progress.update(2)
            progress.update(4)
        lines = stream.getvalue().splitlines()
        self.assertTrue(lines[0].startswith("Checking:"))
        self.assertTrue(lines[-1].endswith("100.0% (4/4)"))

    def test_small_book_end_to_end_with_ffmpeg(self):
        if not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")):
            self.skipTest("ffmpeg and ffprobe are required for the assembly integration test")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clips = root / "clips"
            output = root / "output"
            clips.mkdir()
            rows = [
                {"audio_id": "audio_1_1", "chapter": 1, "chapter_title": "Opening", "Text": "First.", "voice_actor": "Narrator", "Start Location": 1},
                {"audio_id": "audio_2_1", "chapter": 2, "chapter_title": "Next", "Text": "Second.", "voice_actor": "Narrator", "Start Location": 2},
            ]
            manifest_rows = {}
            for index, row in enumerate(rows, start=1):
                wav_path = clips / f"{row['audio_id']}.wav"
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                     "-i", f"sine=frequency={400 + index * 100}:duration=0.25",
                     "-c:a", "pcm_s16le", str(wav_path)],
                    check=True,
                )
                manifest_rows[row["audio_id"]] = {
                    "status": "generated",
                    "wav_sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                }
            book_csv = root / "book.csv"
            with book_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (clips / "manifest.json").write_text(json.dumps({"rows": manifest_rows}), encoding="utf-8")

            script_path = Path(__file__).resolve().parents[1] / "assemble_voxnovel_incremental.py"
            result = subprocess.run(
                [sys.executable, str(script_path), "--book-csv", str(book_csv),
                 "--clips-dir", str(clips), "--output-dir", str(output),
                 "--title", "Integration Test Book", "--author", "Test Author",
                 "--silence-ms", "50"],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertIn("Validating clips:", result.stderr)
            self.assertIn("Encoding audiobook:", result.stderr)
            m4b = output / "Integration_Test_Book.m4b"
            self.assertTrue(m4b.is_file())
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format_tags=title,artist:chapter=tags", "-of", "json", str(m4b)],
                text=True, capture_output=True, check=True,
            )
            metadata = json.loads(probe.stdout)
            self.assertEqual(metadata["format"]["tags"]["title"], "Integration Test Book")
            self.assertEqual(metadata["format"]["tags"]["artist"], "Test Author")
            self.assertEqual(len(metadata.get("chapters", [])), 2)


if __name__ == "__main__":
    unittest.main()
