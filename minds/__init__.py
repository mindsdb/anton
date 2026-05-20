"""Top-level package init.

Suppresses a small set of cosmetic third-party warnings here so they fire at
most once per worker — and before any submodule (which would trigger them
during class-body evaluation) gets a chance to import.
"""

import warnings

# Prefect's TimeZone alias uses Field(default="UTC") on an Annotated type alias;
# pydantic v2 warns this attribute has no effect. Upstream bug, harmless.
warnings.filterwarnings(
    "ignore",
    message=r".*'default' attribute with value 'UTC'.*",
)

# `schema` is the correct, intentional field name on data-catalog Table and the
# public TableResponse / TreeNodeResponse API contracts. Renaming would cascade
# into services, tests, and the JSON API. Pydantic still functions correctly.
warnings.filterwarnings(
    "ignore",
    message=r'Field name "schema" in ".*" shadows an attribute in parent .*',
)
