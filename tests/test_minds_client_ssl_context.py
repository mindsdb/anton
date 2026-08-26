"""Tests for minds_request()'s TLS trust source.

On some Windows machines (typically behind a TLS-inspecting corporate
proxy/AV whose root CA is trusted by the OS but not picked up by CPython's
own Windows cert-store enumeration), verified Minds Hub requests fail with
SSLCertVerificationError even though the browser trusts the same proxy.
minds_request() must build its SSL context via `truststore`, which delegates
verification to the OS-native trust APIs, instead of relying on urllib's
implicit ssl.create_default_context() fallback.
"""
from __future__ import annotations

import ssl
from unittest.mock import MagicMock, patch

import truststore

from anton.minds_client import minds_request


def test_verified_request_uses_truststore_context():
    """verify=True (the default) must build a truststore.SSLContext, not
    rely on the implicit default context urllib would otherwise use."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = b"{}"
        minds_request("https://view.mindshub.ai/upload", "key")

    _, kwargs = mock_urlopen.call_args
    assert isinstance(kwargs["context"], truststore.SSLContext)
    assert kwargs["context"].verify_mode == ssl.CERT_REQUIRED
    assert kwargs["context"].check_hostname is True


def test_unverified_request_disables_hostname_and_cert_checks():
    """verify=False must still fully disable verification (used only for
    explicit user opt-out via ANTON_MINDS_SSL_VERIFY=false)."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = b"{}"
        minds_request("https://view.mindshub.ai/upload", "key", verify=False)

    _, kwargs = mock_urlopen.call_args
    assert kwargs["context"].check_hostname is False
    assert kwargs["context"].verify_mode == ssl.CERT_NONE
