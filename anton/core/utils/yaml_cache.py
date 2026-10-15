"""`yaml.safe_load` for text that is read again and again.

A turn builds `DatasourceRegistry()` several times and lists skills at least
once, and each of those re-reads the same datasources.md blocks and SKILL.md
frontmatter. PyYAML's pure-Python parser made that most of a turn's setup CPU.

The cache is keyed by the text itself, not by a file's timestamp, so an edited
file is parsed afresh on its next read and an unchanged one only once.
"""

from __future__ import annotations

import copy
import functools
import sys
from typing import Any

import yaml

# Distinct texts a process parses: the built-in datasources.md blocks, the
# user's, and the frontmatter of every skill it reads. Past this, the least
# recently used text is parsed again on its next read.
_MAX_TEXTS = 512

# The longest text the cache keeps. The largest built-in datasources.md block
# is about 2,300 characters and a shipped SKILL.md frontmatter under 900. A
# longer text is parsed on every call and freed with its result.
_MAX_CACHED_TEXT_CHARS = 8 * 1024

# The most memory a kept parse may take. A short text can still parse large:
# each merge key (`<<: *anchor`) copies every entry of the anchored mapping, so
# a text under the cap above can parse into more than 10 MB. The largest
# built-in datasources.md block parses into about 11 KB. A larger parse is
# returned without being kept, so the cache holds at most about 50 MB: 512
# parses and the texts that key them.
_MAX_CACHED_PARSE_BYTES = 64 * 1024

# The most nodes and scalar characters that aliases and merge keys may add to
# a text's own. PyYAML keeps an alias as a second reference to the anchored
# node, but a merge key copies the node's entries while parsing, and `str()` or
# JSON copies the whole node at every reference, so nested aliases multiply.
# A 100-entry mapping merged 40 times adds about 28,000.
_MAX_ALIAS_GROWTH = 64 * 1024


def _node_units(node: yaml.Node) -> int:
    """One per node, plus the characters of a scalar's text."""
    if isinstance(node, yaml.ScalarNode):
        return 1 + len(node.value)
    return 1


def _child_nodes(node: yaml.Node) -> list[yaml.Node]:
    if isinstance(node, yaml.SequenceNode):
        return node.value
    if isinstance(node, yaml.MappingNode):
        return [child for pair in node.value for child in pair]
    return []


class AliasLoopError(yaml.YAMLError):
    """An alias refers to a node that holds it."""


class AliasGrowthError(yaml.YAMLError):
    """Aliases add more than `_MAX_ALIAS_GROWTH` nodes and characters."""


def _alias_growth(root: yaml.Node) -> int:
    """How many units aliases add: the document with every alias copied out,
    less the document as written.

    Each distinct node is visited once, so this takes time linear in the text
    however far the aliases would expand.

    Raises `AliasLoopError` on an alias to a node that holds it. Each `str()`
    a caller makes of a value prints such a loop in full, so a count that
    visits each node once cannot bound what the caller prints. No skill or
    datasource block needs one.
    """
    expanded: dict[int, int] = {}
    entered: set[int] = set()
    written = 0
    pending: list[tuple[yaml.Node, bool]] = [(root, False)]
    while pending:
        node, children_done = pending.pop()
        if children_done:
            expanded[id(node)] = _node_units(node) + sum(
                expanded[id(child)] for child in _child_nodes(node)
            )
            written += _node_units(node)
        elif id(node) not in entered:
            entered.add(id(node))
            pending.append((node, True))
            pending.extend((child, False) for child in _child_nodes(node))
        elif id(node) not in expanded:
            # Entered but not finished: the node holds the one being walked.
            raise AliasLoopError("an alias refers to a node that holds it")
    return expanded[id(root)] - written


class _BoundedSafeLoader(yaml.SafeLoader):
    """`yaml.SafeLoader` that refuses a text whose aliases expand past
    `_MAX_ALIAS_GROWTH` or refer to a node that holds them, before it builds
    any Python object from it."""

    def get_single_node(self) -> yaml.Node | None:
        node = super().get_single_node()
        if node is not None and _alias_growth(node) > _MAX_ALIAS_GROWTH:
            raise AliasGrowthError(
                f"aliases expand this YAML by more than {_MAX_ALIAS_GROWTH} "
                "nodes and characters"
            )
        return node


def _load(text: str) -> Any:
    return yaml.load(text, Loader=_BoundedSafeLoader)


class _TooLargeToKeep(Exception):
    """Hands a parse back from `_parse` without `lru_cache` keeping it."""

    def __init__(self, parsed: Any) -> None:
        super().__init__()
        self.parsed = parsed


def _larger_than(value: Any, limit: int) -> bool:
    """Whether `value` and everything it holds take more than `limit` bytes.

    Counts an object once for every place that holds it, so a mapping that
    merge keys or aliases name many times counts many times. Stops once the
    count passes `limit`, so it ends quickly on a parse that holds millions of
    entries.
    """
    pending = [value]
    while pending:
        node = pending.pop()
        limit -= sys.getsizeof(node)
        if limit < 0:
            return True
        if isinstance(node, dict):
            pending.extend(node.keys())
            pending.extend(node.values())
        elif isinstance(node, (list, tuple, set)):
            pending.extend(node)
    return False


@functools.lru_cache(maxsize=_MAX_TEXTS)
def _parse(text: str) -> Any:
    parsed = _load(text)
    if _larger_than(parsed, _MAX_CACHED_PARSE_BYTES):
        raise _TooLargeToKeep(parsed)
    return parsed


def safe_load_cached(text: str) -> Any:
    """`yaml.safe_load(text)`, parsing each distinct text once.

    Returns a deep copy, so a caller can change what it gets without changing
    the cached parse or another caller's copy. A `yaml.YAMLError` is raised on
    every call, because `lru_cache` stores nothing for a call that raised. That
    includes a text whose aliases would add more than `_MAX_ALIAS_GROWTH` nodes
    and characters, or refer to a node that holds them, which is refused before
    any Python object is built.
    A text longer than `_MAX_CACHED_TEXT_CHARS`, or one whose parse is larger
    than `_MAX_CACHED_PARSE_BYTES`, is parsed afresh every time.
    """
    if len(text) > _MAX_CACHED_TEXT_CHARS:
        return _load(text)
    try:
        return copy.deepcopy(_parse(text))
    except _TooLargeToKeep as too_large:
        return too_large.parsed
