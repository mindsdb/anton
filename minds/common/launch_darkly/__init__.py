import threading

from ldclient.client import LDClient
from ldclient.config import Config, HTTPConfig

from minds.common.logger import setup_logging
from minds.common.settings.app_settings import get_app_settings

# Set up logging
logger = setup_logging()

_init_lock = threading.Lock()
_is_initialized = False
_client: LDClient | None = None


def init_launchdarkly() -> None:
    """
    Initialize the LaunchDarkly SDK exactly once per process.
    Note: This cannot be shared across multiple OS worker processes. If you need
    to reduce outgoing connections to LaunchDarkly across workers, configure a Relay Proxy.
    """
    global _is_initialized
    global _client

    if _is_initialized:
        return

    with _init_lock:
        if _is_initialized:
            return
        settings = get_app_settings()

        sdk_key = settings.launchdarkly.sdk_key or ""
        offline = settings.launchdarkly.offline_mode
        base_uri = settings.launchdarkly.base_uri
        stream_uri = settings.launchdarkly.stream_uri
        events_uri = settings.launchdarkly.events_uri
        send_events = settings.launchdarkly.send_events
        diagnostic_opt_out = settings.launchdarkly.diagnostic_opt_out
        http_connect_timeout = settings.launchdarkly.http_connect_timeout
        http_read_timeout = settings.launchdarkly.http_read_timeout

        logger.info("Initializing LaunchDarkly client")
        logger.info("LaunchDarkly offline mode: %s", offline)

        # Avoid logging the SDK key (sensitive).
        if not settings.launchdarkly.sdk_key and not offline:
            logger.warning("LaunchDarkly SDK key is not set (LAUNCHDARKLY__SDK_KEY).")

        # ldclient.Config expects URI strings and have no setter methods, so we need to pass them as kwargs.
        config_kwargs: dict = {
            "sdk_key": sdk_key,
            "offline": offline,
            "diagnostic_opt_out": diagnostic_opt_out,
            "send_events": send_events,
            "http": HTTPConfig(
                connect_timeout=http_connect_timeout,
                read_timeout=http_read_timeout,
            ),
        }
        if base_uri:
            config_kwargs["base_uri"] = base_uri
        if stream_uri:
            config_kwargs["stream_uri"] = stream_uri
        if events_uri:
            config_kwargs["events_uri"] = events_uri

        # Construct a singleton LDClient instance per process. We use the class directly instead of
        # ldclient.set_config()/ldclient.get() because some environments package `ldclient` as a
        # namespace package without those convenience exports.
        _client = LDClient(config=Config(**config_kwargs))

        _is_initialized = True

        try:
            if _client is not None and not _client.is_initialized():
                logger.error("LaunchDarkly failed to initialize. Check network connectivity and SDK credentials.")
        except Exception:
            # Don't break app startup if the SDK throws during health check.
            logger.exception("LaunchDarkly initialization health check failed")


def get_client():
    """
    Return the process-global LaunchDarkly client, initializing it if needed.
    """
    global _client
    init_launchdarkly()
    if _client is None:
        # Should not happen, but keep a clear error if initialization failed.
        raise RuntimeError("LaunchDarkly client was not initialized")
    return _client


def close_launchdarkly() -> None:
    """
    Close the LaunchDarkly client to tear down streaming connections.

    This is mainly useful for short-lived processes (background jobs/queue workers)
    to avoid stale/leaked connections.
    """
    global _is_initialized
    global _client

    with _init_lock:
        if not _is_initialized:
            return
        try:
            if _client is not None:
                _client.close()
        finally:
            _client = None
            _is_initialized = False
