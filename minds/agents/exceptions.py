class QueryPlanningError(Exception):
    """Custom exception for errors during query planning."""

    pass


class QueryGenerationError(Exception):
    """Custom exception for errors during SQL generation."""

    pass


class DataCatalogValidationError(QueryPlanningError):
    """Exception for data catalog validation errors (e.g., tables don't exist in catalog)."""

    pass
