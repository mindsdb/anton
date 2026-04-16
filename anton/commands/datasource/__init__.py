"""Datasource slash-command handlers."""

from anton.commands.datasource.helpers import show_credential_help
from anton.commands.datasource.connect import _PROMPT_RECONNECT_CANCEL
from anton.commands.datasource.manage import handle_list_data_sources, handle_remove_data_source
from anton.commands.datasource.custom import (
    handle_add_custom_datasource,
    _CustomDatasourceField,
    _CustomDatasourceSpec,
)
from anton.commands.datasource.verify import run_connection_test, handle_test_datasource
from anton.commands.datasource.connect import handle_connect_datasource

__all__ = [
    "_PROMPT_RECONNECT_CANCEL",
    "show_credential_help",
    "handle_list_data_sources",
    "handle_remove_data_source",
    "handle_add_custom_datasource",
    "run_connection_test",
    "handle_test_datasource",
    "handle_connect_datasource",
]
