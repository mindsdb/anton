from fastapi import APIRouter, Depends, HTTPException, Query, Request

from minds.client.mindsdb import create_mindsdb_client_from_request
from minds.requests.context import extract_context_from_request
from minds.schemas.tree import TreeNodeResponse
from minds.services.tree import TreeService

router = APIRouter()


def get_tree_service(request: Request) -> TreeService:
    """
    Dependency function to create TreeService with user context and MindsDB client.
    """
    context = extract_context_from_request(request)
    mindsdb_client = create_mindsdb_client_from_request(request)

    return TreeService(mindsdb_client=mindsdb_client, user_id=context.user_id)


@router.get("/", response_model=list[TreeNodeResponse])
async def get_databases_tree(tree_service: TreeService = Depends(get_tree_service)) -> list[TreeNodeResponse]:
    """
    Get tree structure of all databases available on MindsDB.

    Returns a list of database nodes with metadata including:
    - name: database name
    - class: node type ('db')
    - type: database type ('data', 'project', 'system')
    - engine: database engine
    - deletable: whether the database can be deleted
    - visible: whether the database is visible
    """
    try:
        return tree_service.get_databases_tree()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get databases tree: {str(e)}") from e


@router.get("/{database_name}", response_model=list[TreeNodeResponse])
async def get_database_tree(
    database_name: str,
    with_schemas: bool = Query(False, description="Include schema information for data databases"),
    tree_service: TreeService = Depends(get_tree_service),
) -> list[TreeNodeResponse]:
    """
    Get tree structure of tables and schemas within a specific database.

    Args:
        database_name: Name of the database to explore
        with_schemas: Whether to include schema information for data databases

    Returns a list of table/schema nodes with metadata including:
    - name: table/schema name
    - class: node type ('table', 'schema', 'job')
    - type: table type ('table', 'view', 'job', 'system view')
    - engine: table engine (if applicable)
    - deletable: whether the item can be deleted
    - schema: schema name (for tables in schemas)
    - children: nested tables (for schema nodes)
    """
    try:
        return tree_service.get_database_tree(database_name, with_schemas=with_schemas)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get database tree: {str(e)}") from e
