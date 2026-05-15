"""Tests for anton.clipboard — all clipboard/subprocess access is mocked."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from anton.clipboard import (
    ClipboardImage,
    ClipboardResult,
    PastedImageRegistry,
    UploadedFile,
    cleanup_old_uploads,
    grab_clipboard,
    is_clipboard_supported,
    parse_dropped_paths,
    replace_image_paths_in_pasted,
    save_clipboard_image,
)


class TestIsClipboardSupported:
    def test_supported_darwin_with_pillow(self):
        with patch("anton.clipboard.platform") as mock_platform:
            mock_platform.system.return_value = "Darwin"
            with patch.dict("sys.modules", {"PIL": MagicMock(), "PIL.ImageGrab": MagicMock()}):
                assert is_clipboard_supported() is True

    def test_linux_supported_with_tools_and_pillow(self):
        with patch("anton.clipboard.platform") as mock_platform:
            mock_platform.system.return_value = "Linux"
            with patch("anton.clipboard._linux_clipboard_tool", return_value="xclip"):
                with patch.dict("sys.modules", {"PIL": MagicMock(), "PIL.ImageGrab": MagicMock()}):
                    assert is_clipboard_supported() is True

    def test_linux_unsupported_without_tools(self):
        with patch("anton.clipboard.platform") as mock_platform:
            mock_platform.system.return_value = "Linux"
            with patch("anton.clipboard._linux_clipboard_tool", return_value=None):
                with patch.dict("sys.modules", {"PIL": MagicMock(), "PIL.ImageGrab": MagicMock()}):
                    assert is_clipboard_supported() is False

    def test_unsupported_other_platform(self):
        with patch("anton.clipboard.platform") as mock_platform:
            mock_platform.system.return_value = "FreeBSD"
            assert is_clipboard_supported() is False

    def test_unsupported_no_pillow(self):
        with patch("anton.clipboard.platform") as mock_platform:
            mock_platform.system.return_value = "Darwin"
            # Force ImportError for PIL.ImageGrab
            with patch.dict("sys.modules", {"PIL": None, "PIL.ImageGrab": None}):
                assert is_clipboard_supported() is False


class TestGrabClipboard:
    def test_image_found(self):
        fake_img = MagicMock()
        fake_img.size = (800, 600)
        fake_img.mode = "RGB"

        mock_ig = MagicMock()
        mock_ig.grabclipboard.return_value = fake_img

        mock_image_mod = MagicMock()
        mock_image_mod.Image = fake_img.__class__

        with patch.dict("sys.modules", {
            "PIL": MagicMock(),
            "PIL.ImageGrab": mock_ig,
            "PIL.Image": mock_image_mod,
        }):
            # Also need to make isinstance work — use a simpler approach
            with patch("anton.clipboard._grab_image", return_value=fake_img):
                result = grab_clipboard()

        assert result.image is not None
        assert result.image.width == 800
        assert result.image.height == 600
        assert result.image.mode == "RGB"
        assert result.file_paths == []
        assert result.text == ""

    def test_text_found(self):
        with patch("anton.clipboard._grab_image", return_value=None):
            with patch("anton.clipboard._grab_text", return_value="hello world"):
                result = grab_clipboard()

        assert result.image is None
        assert result.file_paths == []
        assert result.text == "hello world"

    def test_file_paths_found(self, tmp_path):
        # Create real files so parse_dropped_paths sees them
        f1 = tmp_path / "test.txt"
        f1.write_text("hi")

        with patch("anton.clipboard._grab_image", return_value=None):
            with patch("anton.clipboard._grab_text", return_value=str(f1)):
                result = grab_clipboard()

        assert result.image is None
        assert len(result.file_paths) == 1
        assert result.file_paths[0] == f1
        assert result.text == ""

    def test_empty_clipboard(self):
        with patch("anton.clipboard._grab_image", return_value=None):
            with patch("anton.clipboard._grab_text", return_value=""):
                result = grab_clipboard()

        assert result.image is None
        assert result.file_paths == []
        assert result.text == ""

    def test_pillow_missing(self):
        with patch("anton.clipboard._grab_image", return_value=None):
            with patch("anton.clipboard._grab_text", return_value=""):
                result = grab_clipboard()

        assert result.image is None


class TestGrabImageInternal:
    def test_returns_none_without_pillow(self):
        from anton.clipboard import _grab_image

        with patch.dict("sys.modules", {"PIL": None, "PIL.ImageGrab": None}):
            # Import will raise ImportError
            assert _grab_image() is None

    def test_list_returned_is_skipped(self):
        """macOS Finder file copy returns a list of paths."""
        from anton.clipboard import _grab_image

        mock_ig = MagicMock()
        mock_ig.grabclipboard.return_value = ["/path/to/file.png"]

        with patch.dict("sys.modules", {"PIL.ImageGrab": mock_ig, "PIL": MagicMock()}):
            result = _grab_image()

        assert result is None


class TestSaveClipboardImage:
    def test_saves_png_and_metadata(self, tmp_path):
        uploads = tmp_path / "uploads"

        fake_img = MagicMock()
        fake_img.size = (1024, 768)
        fake_img.mode = "RGBA"
        fake_img.tobytes.return_value = b"\x00" * 4096

        def mock_save(path, format=None):
            Path(path).write_bytes(b"PNG_DATA_HERE")

        fake_img.save = mock_save

        result = save_clipboard_image(fake_img, uploads)

        assert isinstance(result, UploadedFile)
        assert result.path.exists()
        assert result.path.suffix == ".png"
        assert result.width == 1024
        assert result.height == 768
        assert result.format == "PNG"
        assert result.original_type == "clipboard"
        assert result.size_bytes > 0

        # Check sidecar metadata
        meta_path = result.path.with_suffix(".png.meta.json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["source"] == "clipboard"
        assert meta["width"] == 1024
        assert meta["height"] == 768

    def test_creates_dir_if_missing(self, tmp_path):
        uploads = tmp_path / "deep" / "nested" / "uploads"

        fake_img = MagicMock()
        fake_img.size = (100, 100)
        fake_img.mode = "RGB"
        fake_img.tobytes.return_value = b"\x00" * 100

        def mock_save(path, format=None):
            Path(path).write_bytes(b"PNG")

        fake_img.save = mock_save

        result = save_clipboard_image(fake_img, uploads)
        assert uploads.is_dir()
        assert result.path.exists()

    def test_unique_filenames(self, tmp_path):
        uploads = tmp_path / "uploads"

        def make_img(data_byte: bytes):
            img = MagicMock()
            img.size = (10, 10)
            img.mode = "RGB"
            img.tobytes.return_value = data_byte * 4096
            img.save = lambda path, format=None: Path(path).write_bytes(b"PNG")
            return img

        r1 = save_clipboard_image(make_img(b"\x01"), uploads)
        r2 = save_clipboard_image(make_img(b"\x02"), uploads)

        assert r1.path != r2.path
        assert r1.path.name != r2.path.name


class TestCleanupOldUploads:
    def test_deletes_old_keeps_recent(self, tmp_path):
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        old_file = uploads / "old.png"
        old_file.write_bytes(b"old")
        old_meta = uploads / "old.png.meta.json"
        old_meta.write_text("{}")

        new_file = uploads / "new.png"
        new_file.write_bytes(b"new")

        # Make old file look 10 days old
        import os
        old_mtime = time.time() - (10 * 86400)
        os.utime(old_file, (old_mtime, old_mtime))
        os.utime(old_meta, (old_mtime, old_mtime))

        removed = cleanup_old_uploads(uploads, max_age_days=7)

        assert removed == 2  # old.png + old.png.meta.json
        assert not old_file.exists()
        assert not old_meta.exists()
        assert new_file.exists()

    def test_handles_missing_dir(self, tmp_path):
        nonexistent = tmp_path / "nope"
        removed = cleanup_old_uploads(nonexistent)
        assert removed == 0


class TestReplaceImagePathsInPasted:
    def test_single_image_path(self, tmp_path):
        f = tmp_path / "test.png"
        f.write_text("image data")
        registry = PastedImageRegistry()

        text = f"Check this: {f}"
        rewritten, registered = replace_image_paths_in_pasted(text, registry)

        assert len(registered) == 1
        assert registered[0].id == 1
        assert f"[Image #1]" in rewritten
        assert str(f) not in rewritten

    def test_quoted_image_path(self, tmp_path):
        f = tmp_path / "test.jpg"
        f.write_text("image data")
        registry = PastedImageRegistry()

        text = f"'{f}'"
        rewritten, registered = replace_image_paths_in_pasted(text, registry)

        assert len(registered) == 1
        assert "[Image #1]" in rewritten

    def test_multiple_image_paths(self, tmp_path):
        f1 = tmp_path / "first.png"
        f2 = tmp_path / "second.jpg"
        f1.write_text("img1")
        f2.write_text("img2")
        registry = PastedImageRegistry()

        text = f"{f1} {f2}"
        rewritten, registered = replace_image_paths_in_pasted(text, registry)

        assert len(registered) == 2
        assert "[Image #1]" in rewritten
        assert "[Image #2]" in rewritten

    def test_placeholder_followed_by_path_with_space(self, tmp_path):
        """Test: [Image #1] '/path/to/image.png' with space between them.

        When detector adds a space before processing (the fix for the paste bug),
        the function receives text like this and should detect the path correctly.
        """
        f = tmp_path / "second.png"
        f.write_text("image data")
        registry = PastedImageRegistry()

        # First, add a dummy image to start ID at 2
        registry.add(tmp_path / "dummy.png")

        # After detector inserts space: [Image #1] '/path/to/image.png'
        text = f"[Image #1] '{f}'"
        rewritten, registered = replace_image_paths_in_pasted(text, registry)

        assert len(registered) == 1
        assert registered[0].id == 2  # Registry continues from 1
        assert "[Image #2]" in rewritten
        assert str(f) not in rewritten

    def test_nonexistent_path_ignored(self, tmp_path):
        registry = PastedImageRegistry()
        text = "/nonexistent/path.png other text"
        rewritten, registered = replace_image_paths_in_pasted(text, registry)

        assert len(registered) == 0
        assert rewritten == text  # Unchanged

    def test_non_image_extensions_ignored(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("text")
        registry = PastedImageRegistry()

        text = f"{f}"
        rewritten, registered = replace_image_paths_in_pasted(text, registry)

        assert len(registered) == 0
        assert rewritten == text


class TestParseDroppedPaths:
    def test_absolute_path(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = parse_dropped_paths(str(f))
        assert result == [f]

    def test_quoted_path(self, tmp_path):
        d = tmp_path / "my dir"
        d.mkdir()
        f = d / "file.txt"
        f.write_text("hello")
        result = parse_dropped_paths(f"'{f}'")
        assert result == [f]

    def test_nonexistent_filtered(self, tmp_path):
        fake = tmp_path / "nonexistent.txt"
        result = parse_dropped_paths(str(fake))
        assert result == []

    def test_multiple_files(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("a")
        f2.write_text("b")
        text = f"{f1}\n{f2}"
        result = parse_dropped_paths(text)
        assert len(result) == 2

    def test_relative_path_ignored(self):
        result = parse_dropped_paths("relative/path.txt")
        assert result == []

    def test_empty_input(self):
        result = parse_dropped_paths("")
        assert result == []
