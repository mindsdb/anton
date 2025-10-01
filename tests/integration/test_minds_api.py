import logging, os
import time
import pytest
from requests import Session
from .config import MINDS_API_BASE_URL
from .conftest import poll_mind_transitions, DATASOURCE_CONFIGS
# States we expect a mind to pass through before completion

@pytest.mark.happy_path
class TestMindsAPI:

    def test_mind_crud_workflow(self, api_client, temporary_mind):
        """
        Tests the full CRUD lifecycle of a Mind.
        """
        mind_name, (ds_name, config) = temporary_mind

        # --- READ: Verify mind creation ---
        logging.info(f"TEST: Verifying creation of mind '{mind_name}'")
        get_resp = api_client.get(f"{MINDS_API_BASE_URL}/api/v1/minds/{mind_name}")
        assert get_resp.status_code == 200, f"Failed to get mind. Response: {get_resp.text}"
        assert get_resp.json()["name"] == mind_name

        # Poll until final state
        mind_data = poll_mind_transitions(api_client, mind_name)
        assert mind_data["status"] == "COMPLETED"

        # Validate table exists for this datasource
        expected_table = config["sample_table"].lower()
        found = False
        for ds in mind_data.get("datasources", []):
            if ds.get("name") != ds_name:
                continue
            for table in ds.get("tables", []):
                if expected_table in table.lower():
                    found = True
                    break
        assert found, (
            f"Expected table '{expected_table}' not found in mind '{mind_name}' "
            f"datasource '{ds_name}'. Found tables: {ds.get('tables', [])}"
        )
        # --- UPDATE: Modify mind parameters ---
        logging.info(f"TEST: Updating mind '{mind_name}'")
        update_payload = {"parameters": {"temperature": 0.5}}
        update_resp = api_client.put(f"{MINDS_API_BASE_URL}/api/v1/minds/{mind_name}", json=update_payload)
        assert update_resp.status_code == 200
        assert update_resp.json()["parameters"]["temperature"] == 0.5

        # --- DELETE: Clean up mind ---
        logging.info(f"TEST: Deleting mind '{mind_name}'")
        delete_resp = api_client.delete(f"{MINDS_API_BASE_URL}/api/v1/minds/{mind_name}")
        assert delete_resp.status_code in (200, 204), f"Failed to delete mind. Response: {delete_resp.text}"

        # Verify deletion
        get_after_delete = api_client.get(f"{MINDS_API_BASE_URL}/api/v1/minds/{mind_name}")
        assert get_after_delete.status_code == 404

    def test_list_minds(self, api_client, temporary_mind):
        # Unpack mind_name and datasource info
        mind_name, (ds_name, config) = temporary_mind

        logging.info("TEST: Listing all minds and searching for the new mind.")
        max_retries = 5
        delay_seconds = 2

        for attempt in range(max_retries):
            list_resp = api_client.get(f"{MINDS_API_BASE_URL}/api/v1/minds/")
            assert list_resp.status_code == 200, f"Failed to list minds. Response: {list_resp.text}"
            minds = list_resp.json()
            assert isinstance(minds, list)

            if any(mind["name"] == mind_name for mind in minds):
                logging.info(f"VERIFIED: Found '{mind_name}' in the list on attempt {attempt + 1}.")
                return

            logging.warning(f"Attempt {attempt + 1}/{max_retries}: Mind '{mind_name}' not found. Retrying in {delay_seconds}s...")
            time.sleep(delay_seconds)

        pytest.fail(f"Mind '{mind_name}' was not found in the list after {max_retries} attempts.")

