"""
Unit tests for Mind model.

Tests the SQLModel Mind class including:
- Model creation and validation
- Field constraints and defaults
- Helper methods
- Database constraints
"""

import pytest

from minds.model.mind import Mind


class TestMindModel:
    """Test suite for Mind model."""

    @pytest.fixture
    def sample_mind_data(self):
        """Sample mind data for testing."""
        return {
            "name": "test-mind",
            "provider": "openai",
            "model_name": "gpt-4o",
            "user_id": "user-123",
            "parameters": {"temperature": 0.7, "max_tokens": 100},
            "datasources": ["datasource1", "datasource2"],
            "description": "Test mind for unit tests",
            "is_active": True,
        }

    def test_mind_creation_with_all_fields(self, sample_mind_data):
        """Test creating a Mind with all fields."""
        mind = Mind(**sample_mind_data)

        assert mind.name == "test-mind"
        assert mind.provider == "openai"
        assert mind.model_name == "gpt-4o"
        assert mind.user_id == "user-123"
        assert mind.parameters == {"temperature": 0.7, "max_tokens": 100}
        assert mind.datasources == ["datasource1", "datasource2"]
        assert mind.description == "Test mind for unit tests"
        assert mind.is_active is True

    def test_mind_creation_with_minimal_fields(self):
        """Test creating a Mind with minimal required fields."""
        mind = Mind(
            name="minimal-mind", provider="openai", model_name="gpt-4o", user_id="user-123"
        )

        assert mind.name == "minimal-mind"
        assert mind.provider == "openai"
        assert mind.model_name == "gpt-4o"
        assert mind.user_id == "user-123"

        # Test default values
        assert mind.parameters == {}
        assert mind.datasources == []
        assert mind.description is None
        assert mind.is_active is True

    def test_mind_default_values(self):
        """Test Mind model default values."""
        mind = Mind(
            name="default-test", provider="openai", model_name="gpt-4o", user_id="user-123"
        )

        # Test that default factories create new instances
        assert isinstance(mind.parameters, dict)
        assert isinstance(mind.datasources, list)
        assert mind.is_active is True

        mind2 = Mind(
            name="default-test-2", provider="openai", model_name="gpt-4o", user_id="user-123"
        )

        mind.parameters["test"] = "value"
        mind.datasources.append("test-datasource")

        assert "test" not in mind2.parameters
        assert "test-datasource" not in mind2.datasources

    def test_add_datasource_method(self, sample_mind_data):
        """Test add_datasource helper method."""
        mind = Mind(**sample_mind_data)
        original_count = len(mind.datasources)

        # Add new datasource
        mind.add_datasource("new-datasource")
        assert len(mind.datasources) == original_count + 1
        assert "new-datasource" in mind.datasources

        # Add existing datasource (should not duplicate)
        mind.add_datasource("datasource1")
        assert len(mind.datasources) == original_count + 1
        assert mind.datasources.count("datasource1") == 1

    def test_add_datasource_method_with_none_datasources(self):
        """Test add_datasource when datasources is None."""
        mind = Mind(
            name="test",
            provider="openai",
            model_name="gpt-4o",
            user_id="user-123",
            datasources=None,
        )

        mind.add_datasource("first-datasource")
        assert mind.datasources == ["first-datasource"]

    def test_remove_datasource_method(self, sample_mind_data):
        """Test remove_datasource helper method."""
        mind = Mind(**sample_mind_data)
        original_count = len(mind.datasources)

        # Remove existing datasource
        result = mind.remove_datasource("datasource1")
        assert result is True
        assert len(mind.datasources) == original_count - 1
        assert "datasource1" not in mind.datasources

        # Remove non-existing datasource
        result = mind.remove_datasource("non-existing")
        assert result is False
        assert len(mind.datasources) == original_count - 1

    def test_remove_datasource_method_with_none_datasources(self):
        """Test remove_datasource when datasources is None."""
        mind = Mind(
            name="test",
            provider="openai",
            model_name="gpt-4o",
            user_id="user-123",
            datasources=None,
        )

        result = mind.remove_datasource("any-datasource")
        assert result is False

    def test_remove_datasource_method_with_empty_datasources(self):
        """Test remove_datasource when datasources is empty."""
        mind = Mind(
            name="test",
            provider="openai",
            model_name="gpt-4o",
            user_id="user-123",
            datasources=[],
        )

        result = mind.remove_datasource("any-datasource")
        assert result is False

    def test_mind_string_representation(self, sample_mind_data):
        """Test Mind string representation."""
        mind = Mind(**sample_mind_data)
        mind_str = str(mind)

        # Should contain key identifying information
        assert "test-mind" in mind_str

    def test_mind_json_serialization(self, sample_mind_data):
        """Test Mind JSON serialization."""
        mind = Mind(**sample_mind_data)

        # Test that parameters and datasources are properly serializable
        assert isinstance(mind.parameters, dict)
        assert isinstance(mind.datasources, list)

        # Test model_dump works
        mind_dict = mind.model_dump()
        assert isinstance(mind_dict, dict)
        assert mind_dict["name"] == "test-mind"
        assert mind_dict["parameters"] == {"temperature": 0.7, "max_tokens": 100}
        assert mind_dict["datasources"] == ["datasource1", "datasource2"]

    def test_mind_field_constraints(self):
        """Test Mind field constraints and validation."""
        # Test name field constraints
        mind = Mind(
            name="a" * 256,  # Max length
            provider="openai",
            model_name="gpt-4o",
            user_id="user-123",
        )
        assert len(mind.name) == 256

        # Test provider field constraints
        mind = Mind(
            name="test",
            provider="a" * 50,  # Max length
            model_name="gpt-4o",
            user_id="user-123",
        )
        assert len(mind.provider) == 50

        # Test model_name field constraints
        mind = Mind(
            name="test",
            provider="openai",
            model_name="a" * 256,  # Max length
            user_id="user-123",
        )
        assert len(mind.model_name) == 256

        # Test user_id field constraints
        mind = Mind(
            name="test",
            provider="openai",
            model_name="gpt-4o",
            user_id="a" * 256,  # Max length
        )
        assert len(mind.user_id) == 256

        # Test user_id field constraints
        mind = Mind(
            name="test",
            provider="openai",
            model_name="gpt-4o",
            user_id="user-123",
        )
        assert len(mind.user_id) == 256

    def test_mind_unique_constraint(self):
        """Test Mind unique constraint configuration."""
        # Check that the unique constraint exists
        assert hasattr(Mind, "__table_args__")
        table_args = Mind.__table_args__

        # Find the UniqueConstraint
        unique_constraints = [arg for arg in table_args if hasattr(arg, "columns")]
        assert len(unique_constraints) > 0

        # Check the constraint columns
        constraint = unique_constraints[0]
        constraint_columns = [col.name for col in constraint.columns]
        assert "name" in constraint_columns
        assert "user_id" in constraint_columns

    def test_mind_indexes(self):
        """Test Mind index configuration."""
        # Test that indexed fields are properly configured
        name_field = Mind.__table__.columns["name"]
        user_id_field = Mind.__table__.columns["user_id"]

        assert name_field.index is True
        assert user_id_field.index is True

    def test_mind_json_fields(self):
        """Test Mind JSON field handling."""
        mind = Mind(
            name="json-test",
            provider="openai",
            model_name="gpt-4o",
            user_id="user-123",
            parameters={"nested": {"deep": {"value": 42}}},
            datasources=["complex", "datasource", "list"],
        )

        # Test that complex JSON structures are handled
        assert mind.parameters["nested"]["deep"]["value"] == 42
        assert len(mind.datasources) == 3

    def test_mind_text_field(self):
        """Test Mind text field (description)."""
        long_description = "A" * 1000  # Long text

        mind = Mind(
            name="text-test",
            provider="openai",
            model_name="gpt-4o",
            user_id="user-123",
            description=long_description,
        )

        assert mind.description == long_description
        assert len(mind.description) == 1000

    def test_mind_boolean_field(self):
        """Test Mind boolean field (is_active)."""
        # Test True value
        mind_active = Mind(
            name="active-mind",
            provider="openai",
            model_name="gpt-4o",
            user_id="user-123",
            is_active=True,
        )
        assert mind_active.is_active is True

        # Test False value
        mind_inactive = Mind(
            name="inactive-mind",
            provider="openai",
            model_name="gpt-4o",
            user_id="user-123",
            is_active=False,
        )
        assert mind_inactive.is_active is False

    def test_mind_inheritance_from_base(self, sample_mind_data):
        """Test that Mind inherits from BaseSQLModel correctly."""
        mind = Mind(**sample_mind_data)

        # Test that it has the expected base fields
        assert hasattr(mind, "id")
        assert hasattr(mind, "created_on")
        assert hasattr(mind, "modified_on")

        assert mind.created_on is None
        assert mind.modified_on is None
