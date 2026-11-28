"""Tests for incremental M4B cover-art handling."""
from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from assemble_voxnovel_incremental import (
    build_ffmpeg_command,
    extract_cover_image,
    resolve_cover_image,
)


def make_epub(path: Path, package_xml: str, extra_files: dict[str, bytes] | None = None) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="EPUB/package.opf" '
            'media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        archive.writestr("EPUB/package.opf", package_xml)
        for name, data in (extra_files or {}).items():
            archive.writestr(name, data)


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
            with patch("assemble_voxnovel_incremental.shutil.which") as which:
                self.assertEqual(resolve_cover_image(cover, None, Path(temp)), cover)
                which.assert_not_called()

    def test_missing_explicit_cover_is_an_error(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                resolve_cover_image(Path(temp) / "missing.jpg", None, Path(temp))

    def test_epub3_cover_image_property_is_extracted_without_calibre(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            epub = root / "book.epub"
            cover_bytes = b"\xff\xd8\xfffake-jpeg"
            package = (
                '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                '<manifest><item id="cover" href="images/front.jpg" '
                'media-type="image/jpeg" properties="cover-image"/></manifest></package>'
            )
            make_epub(epub, package, {"EPUB/images/front.jpg": cover_bytes})
            with patch("assemble_voxnovel_incremental.shutil.which") as which:
                extracted = extract_cover_image(epub, root / "tmp" / "cover.jpg")
                which.assert_not_called()
            self.assertEqual(extracted, root / "tmp" / "cover.jpg")
            self.assertEqual(extracted.read_bytes(), cover_bytes)

    def test_epub2_meta_cover_reference_is_extracted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            epub = root / "book.epub"
            cover_bytes = b"\x89PNG\r\n\x1a\nfake-png"
            package = (
                '<package xmlns="http://www.idpf.org/2007/opf" version="2.0">'
                '<metadata><meta name="cover" content="cover-id"/></metadata>'
                '<manifest><item id="cover-id" href="images/cover.png" '
                'media-type="image/png"/></manifest></package>'
            )
            make_epub(epub, package, {"EPUB/images/cover.png": cover_bytes})
            extracted = extract_cover_image(epub, root / "tmp" / "cover.jpg")
            self.assertEqual(extracted.suffix, ".png")
            self.assertEqual(extracted.read_bytes(), cover_bytes)

    def test_epub_guide_cover_page_is_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            epub = root / "book.epub"
            package = (
                '<package xmlns="http://www.idpf.org/2007/opf" version="2.0">'
                '<manifest><item id="cover-page" href="text/cover.xhtml" '
                'media-type="application/xhtml+xml"/><item id="cover-image" '
                'href="images/cover.jpg" media-type="image/jpeg"/></manifest>'
                '<guide><reference type="cover" href="text/cover.xhtml"/></guide></package>'
            )
            page = (
                '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
                '<img src="../images/cover.jpg"/></body></html>'
            ).encode()
            cover = b"\xff\xd8\xffcover"
            make_epub(epub, package, {"EPUB/text/cover.xhtml": page, "EPUB/images/cover.jpg": cover})
            extracted = extract_cover_image(epub, root / "tmp" / "cover.jpg")
            self.assertEqual(extracted.read_bytes(), cover)

    def test_no_declared_cover_returns_none(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            epub = root / "book.epub"
            package = (
                '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                '<manifest><item id="chapter" href="chapter.xhtml" '
                'media-type="application/xhtml+xml"/></manifest></package>'
            )
            make_epub(epub, package, {"EPUB/chapter.xhtml": b"<html/>"})
            self.assertIsNone(extract_cover_image(epub, root / "cover.jpg"))

    def test_both_cover_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cover = root / "cover.jpg"
            epub = root / "book.epub"
            cover.write_bytes(b"image")
            epub.write_bytes(b"not a real epub")
            with self.assertRaises(ValueError):
                resolve_cover_image(cover, epub, root)


if __name__ == "__main__":
    unittest.main()
