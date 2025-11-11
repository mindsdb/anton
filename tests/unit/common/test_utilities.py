from minds.common.utilities import safe_parse


class TestSafeParse:
    def test_parse_valid_json(self):
        """Test parsing a valid JSON string."""
        json_str = '{"key": "value", "number": 42}'
        result = safe_parse(json_str)
        assert result == {"key": "value", "number": 42}

    def test_parse_python_literal_dict(self):
        """Test parsing a Python literal dict with single quotes."""
        python_str = "{'key': 'value', 'flag': True}"
        result = safe_parse(python_str)
        assert result == {"key": "value", "flag": True}
