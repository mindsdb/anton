from pydantic import BaseModel, ConfigDict, Field


class TreeNodeResponse(BaseModel):
    """Response schema for tree nodes."""

    model_config = ConfigDict(populate_by_name=True)  # Allow both 'class_' and 'class' as field names

    name: str = Field(..., description="Name of the tree node")
    class_: str = Field(..., alias="class", description="Type of node (db, table, schema, job)")
    type: str | None = Field(None, description="Specific type (data, project, system, table, view)")
    engine: str | None = Field(None, description="Engine type")
    deletable: bool = Field(False, description="Whether the node can be deleted")
    visible: bool = Field(True, description="Whether the node is visible")
    schema: str | None = Field(None, description="Schema name (for tables)")
    children: list["TreeNodeResponse"] | None = Field(None, description="Child nodes")


# Enable forward reference for self-referencing model
TreeNodeResponse.model_rebuild()


class TreeDatabaseResponse(TreeNodeResponse):
    """Response schema for database tree nodes."""

    class_: str = Field("db", alias="class", description="Always 'db' for database nodes")


class TreeTableResponse(TreeNodeResponse):
    """Response schema for table tree nodes."""

    class_: str = Field("table", alias="class", description="Always 'table' for table nodes")


class TreeQueryParams(BaseModel):
    """Query parameters for tree endpoints."""

    with_schemas: bool = Field(False, description="Include schema information for data databases")
