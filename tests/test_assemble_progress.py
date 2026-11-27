"""Regression tests for assembly progress reporting."""
import io
import unittest
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


if __name__ == "__main__":
    unittest.main()
