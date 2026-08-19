"""The setup summary must name the endpoint that will actually serve requests.

A MindsHub key and URL are set for anyone signed in, so deciding the label from
`minds_url` told a user pointed at a local model that they were on MindsHub —
the claim they installed a local model to be able to trust.
"""
from anton.cli import is_minds_host


def test_local_endpoint_is_not_minds_even_when_signed_in():
    assert (
        is_minds_host("http://192.168.1.100:1234/v1", "https://api.mindshub.ai")
        is False
    )


def test_public_minds_hosts():
    assert is_minds_host("https://api.mindshub.ai/v1") is True
    assert is_minds_host("https://llm.mdb.ai/api/v1") is True


def test_self_hosted_gateway_matches_by_minds_url():
    assert is_minds_host("http://gateway.internal:8080/v1", "http://gateway.internal:8080") is True


def test_same_host_different_port_is_not_the_gateway():
    assert is_minds_host("http://gateway.internal:1234/v1", "http://gateway.internal:8080") is False


def test_lookalike_host_is_not_minds():
    assert is_minds_host("https://notmindshub.ai/v1") is False


def test_unset_base_is_not_minds():
    assert is_minds_host("", "https://api.mindshub.ai") is False
