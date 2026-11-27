"""Regression tests for punctuation-only incremental fragments."""
from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from voxnovel_fragment_utils import merge_punctuation_fragments, write_silent_wav


class MergePunctuationFragmentsTests(unittest.TestCase):
    def test_standalone_closing_quote_is_joined_to_previous_speech(self) -> None:
        self.assertEqual(
            merge_punctuation_fragments(["“You know that area: breach?", "”"]),
            ["“You know that area: breach?”"],
        )

    def test_standalone_markup_without_speech_is_removed(self) -> None:
        self.assertEqual(merge_punctuation_fragments(["***"]), [])

    def test_leading_markup_is_removed_before_spoken_fragment(self) -> None:
        self.assertEqual(
            merge_punctuation_fragments(["***", "A spoken sentence."]),
            ["A spoken sentence."],
        )

    def test_opening_quote_is_carried_to_following_speech(self) -> None:
        self.assertEqual(
            merge_punctuation_fragments(["“", "A spoken sentence."]),
            ["“A spoken sentence."],
        )

    def test_closing_punctuation_between_fragments_stays_with_previous(self) -> None:
        self.assertEqual(
            merge_punctuation_fragments(["First sentence.", "…", "Second sentence."]),
            ["First sentence.…", "Second sentence."],
        )

    def test_unicode_words_and_numbers_count_as_spoken_content(self) -> None:
        self.assertEqual(
            merge_punctuation_fragments(["Besźel", "1926"]),
            ["Besźel", "1926"],
        )

    def test_blank_and_punctuation_only_rows_can_have_valid_silent_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "audio_262_2.wav"
            write_silent_wav(output)
            with wave.open(str(output), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getframerate(), 24000)
                self.assertGreater(wav_file.getnframes(), 0)
                frames = wav_file.readframes(wav_file.getnframes())
                self.assertEqual(frames, b"\x00\x00" * wav_file.getnframes())


if __name__ == "__main__":
    unittest.main()
