from datetime import datetime
from uuid import UUID

from minds.model.mind import Mind


class TestMind:
    """Test cases for Mind model class."""

    def test_mind_initialization_with_name(self):
        """Test that Mind can be instantiated with a name."""
        mind_name = "Test Mind"
        mind = Mind(name=mind_name)

        assert mind.name == mind_name
        # Inherited from BaseSQLModel
        assert mind.id is None
        assert mind.created_on is None
        assert mind.modified_on is None

    def test_mind_inherits_from_base_sql_model(self):
        """Test that Mind inherits all properties from BaseSQLModel."""
        mind = Mind(name="Test Mind")

        # Should have all BaseSQLModel attributes
        assert hasattr(mind, "id")
        assert hasattr(mind, "created_on")
        assert hasattr(mind, "modified_on")

        # Should have the count class method
        assert hasattr(Mind, "count")

    def test_mind_table_name(self):
        """Test that Mind has correct table name."""
        assert Mind.__tablename__ == "minds"

    def test_mind_with_all_fields(self):
        """Test Mind instantiation with all fields including inherited ones."""
        test_id = UUID("12345678-1234-5678-1234-567812345678")
        test_datetime = datetime(2023, 1, 1, 12, 0, 0)
        mind_name = "Complete Mind"

        mind = Mind(
            id=test_id,
            name=mind_name,
            created_on=test_datetime,
            modified_on=test_datetime,
        )

        assert mind.id == test_id
        assert mind.name == mind_name
        assert mind.created_on == test_datetime
        assert mind.modified_on == test_datetime

    def test_mind_name_field_description(self):
        """Test that name field has proper description."""
        fields = Mind.model_fields
        name_field = fields["name"]

        assert name_field.description == "Name of the mind"

    def test_mind_name_field_type(self):
        """Test that name field has correct type."""
        fields = Mind.model_fields
        name_field = fields["name"]

        assert name_field.annotation is str


    def test_mind_is_table_model(self):
        """Test that Mind is configured as a table model."""
        # Verify that Mind is a SQLModel table
        assert hasattr(Mind, "__table__")
        assert Mind.__table__.name == "minds"

    def test_mind_model_validation(self):
        """Test that Mind model validates fields correctly."""
        # Name field should work when provided
        mind = Mind(name="Valid Mind")
        assert mind.name == "Valid Mind"

        # Test that Mind can be created without a name (field has no default but is optional)
        mind_no_name = Mind()
        assert hasattr(mind_no_name, "name")
        # The name field exists but may be None or unset

    def test_mind_string_representation(self):
        """Test string representation of Mind model."""
        mind = Mind(name="Test Mind")

        # The string representation should contain the name
        str_repr = str(mind)
        assert "Test Mind" in str_repr
