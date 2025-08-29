from typing import Any

from mindsdb_sdk.server import Server

from minds.schemas.tree import TreeNodeResponse


class TreeService:
    """Service for handling tree operations with MindsDB."""

    def __init__(self, mindsdb_client: Server, user_id: str):
        self.user_id = user_id
        self.mindsdb_client = mindsdb_client

    def get_databases_tree(self) -> list[TreeNodeResponse]:
        """
        Get the tree structure of all databases available on MindsDB.

        :return: List of TreeNodeResponse objects representing databases
        """
        try:
            # Get tree data from MindsDB SDK
            tree_nodes = self.mindsdb_client.tree()

            # Convert to response format
            responses = []
            for node in tree_nodes:
                response_data = {
                    "name": node.name,
                    "class": node.class_,
                    "type": node.type,
                    "engine": node.engine,
                    "deletable": node.deletable,
                    "visible": node.visible,
                }
                responses.append(TreeNodeResponse(**response_data))

            return responses

        except Exception as e:
            raise RuntimeError(f"Failed to get databases tree: {str(e)}") from e

    def get_database_tree(self, database_name: str, with_schemas: bool = False) -> list[TreeNodeResponse]:
        """
        Get the tree structure of tables/schemas within a specific database.

        :param database_name: Name of the database
        :param with_schemas: Whether to include schema information
        :return: List of TreeNodeResponse objects representing tables/schemas
        """
        try:
            # Get database from MindsDB SDK
            database = self.mindsdb_client.databases.get(database_name)

            # Get tree data
            tree_nodes = database.tree(with_schemas=with_schemas)

            # Convert to response format
            responses = []
            for node in tree_nodes:
                response_data = {
                    "name": node.name,
                    "class": node.class_,
                    "type": node.type,
                    "engine": node.engine,
                    "deletable": node.deletable,
                    "visible": node.visible,
                }

                # Add schema if it's a table node
                if hasattr(node, "schema") and node.schema:
                    response_data["schema"] = node.schema

                # Add children if they exist
                if node.children:
                    response_data["children"] = self._convert_tree_nodes_to_responses(node.children)

                responses.append(TreeNodeResponse(**response_data))

            return responses

        except AttributeError as e:
            if "doesn't exist" in str(e).lower():
                raise ValueError(f"Database '{database_name}' not found") from e
            raise RuntimeError(f"Failed to get database tree: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to get database tree: {str(e)}") from e

    def _convert_tree_nodes_to_responses(self, nodes: list[Any]) -> list[dict[str, Any]]:
        """
        Convert tree nodes to response dictionaries recursively.

        :param nodes: List of tree nodes
        :return: List of response dictionaries
        """
        responses = []
        for node in nodes:
            response_data = {
                "name": node.name,
                "class": node.class_,
                "type": node.type,
                "engine": node.engine,
                "deletable": node.deletable,
                "visible": node.visible,
            }

            # Add schema if it's a table node
            if hasattr(node, "schema") and node.schema:
                response_data["schema"] = node.schema

            # Add children recursively
            if node.children:
                response_data["children"] = self._convert_tree_nodes_to_responses(node.children)

            responses.append(response_data)

        return responses
