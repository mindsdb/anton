"""Tests for publisher module including bug report functionality."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from anton.publisher import (
    _find_referenced_files,
    _zip_html,
    publish,
    publish_bug_report,
)


class TestPublisher:
    """Test publisher functionality."""

    def test_zip_html_single_file(self, tmp_path):
        """Test zipping a single HTML file."""
        html_file = tmp_path / "test.html"
        html_file.write_text("<html><body>Test</body></html>")

        zipped = _zip_html(html_file)
        assert isinstance(zipped, bytes)
        assert len(zipped) > 0

    def test_zip_html_with_references(self, tmp_path):
        """Test zipping HTML with referenced files."""
        html_file = tmp_path / "index.html"
        css_file = tmp_path / "style.css"
        js_file = tmp_path / "script.js"

        html_file.write_text("""
            <html>
            <head>
                <link rel="stylesheet" href="style.css">
                <script src="script.js"></script>
            </head>
            <body>Test</body>
            </html>
        """)
        css_file.write_text("body { color: red; }")
        js_file.write_text("console.log('test');")

        # Test finding references
        refs = _find_referenced_files(html_file)
        assert len(refs) == 2
        assert css_file in refs
        assert js_file in refs

    def test_publish_success(self, tmp_path):
        """Test successful publish."""
        html_file = tmp_path / "test.html"
        html_file.write_text("<html><body>Test</body></html>")

        mock_response = json.dumps(
            {
                "user_prefix": "test-user",
                "md5": "abc123",
                "view_url": "https://example.com/view/abc123",
                "files": ["index.html"],
            }
        )

        with patch("anton.publisher.minds_request", return_value=mock_response):
            result = publish(html_file, api_key="test-key")

            assert result["user_prefix"] == "test-user"
            assert result["view_url"] == "https://example.com/view/abc123"

    def test_publish_file_not_found(self):
        """Test publish with non-existent file."""
        with pytest.raises(FileNotFoundError):
            publish(Path("/non/existent/file.html"), api_key="test-key")


class TestBugReportPublisher:
    """Test bug report publishing functionality."""

    def test_publish_bug_report_success(self, tmp_path):
        """Test successful bug report publish."""
        bug_report_file = tmp_path / "bug_report.json"
        bug_report_data = {
            "timestamp": "2024-01-01T00:00:00",
            "anton_version": "1.0.0",
            "system_info": {"platform": "test"},
            "user_description": "Test bug",
        }
        bug_report_file.write_text(json.dumps(bug_report_data))

        mock_response = json.dumps(
            {
                "status": "success",
                "message": "Bug report received",
                "report_id": "BUG-12345",
            }
        )

        with patch("anton.publisher.minds_request", return_value=mock_response) as mock_request:
            result = publish_bug_report(bug_report_file, api_key="test-key")

            # Verify response
            assert result["status"] == "success"
            assert "report_id" in result

            # Verify request was made to bug-report endpoint
            call_args = mock_request.call_args
            assert "/bug-report" in call_args[0][0]

            # Verify payload structure
            payload = json.loads(call_args[1]["payload"])
            assert payload["bug_report"] is True
            assert payload["filename"] == "bug_report.json"
            assert "file_content" in payload

    def test_publish_bug_report_custom_url(self, tmp_path):
        """Test bug report with custom URL."""
        bug_report_file = tmp_path / "bug_report.json"
        bug_report_file.write_text(json.dumps({"test": "data"}))

        mock_response = json.dumps({"status": "success"})

        with patch("anton.publisher.minds_request", return_value=mock_response) as mock_request:
            publish_bug_report(
                bug_report_file,
                api_key="test-key",
                bug_report_url="https://custom.example.com",
            )

            # Verify custom URL was used
            call_args = mock_request.call_args
            assert call_args[0][0] == "https://custom.example.com/bug-report"

    def test_publish_bug_report_file_not_found(self):
        """Test bug report publish with non-existent file."""
        with pytest.raises(FileNotFoundError):
            publish_bug_report(Path("/non/existent/bug_report.json"), api_key="test-key")

    def test_publish_bug_report_network_error(self, tmp_path):
        """Test bug report publish with network error."""
        bug_report_file = tmp_path / "bug_report.json"
        bug_report_file.write_text(json.dumps({"test": "data"}))

        with patch("anton.publisher.minds_request", side_effect=Exception("Network error")):
            with pytest.raises(Exception, match="Network error"):
                publish_bug_report(bug_report_file, api_key="test-key")
