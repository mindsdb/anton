import logging
import time
import uuid

import pytest

from .conftest import MINDS_API_BASE_URL, poll_mind_transitions


def get_and_verify_mind(api_client, mind_name, expected_status=200):
    """
    Reusable helper to fetch a mind and assert the HTTP status.
    Returns the mind's JSON data if 200, otherwise None.
    """
    logging.info(f"HELPER: GET /api/v1/minds/{mind_name} (expecting {expected_status})")
    get_resp = api_client.get(f"{MINDS_API_BASE_URL}/api/v1/minds/{mind_name}")

    assert get_resp.status_code == expected_status, (
        f"Failed to get mind '{mind_name}'. "
        f"Expected {expected_status}, got {get_resp.status_code}. "
        f"Response: {get_resp.text}"
    )

    if expected_status == 200:
        return get_resp.json()
    return None


# --- Test Class ---


@pytest.mark.happy_path
class TestMindsAPI:
    def test_mind_crud_workflow(self, api_client, temporary_datasource):
        """
        Tests the full C-R-U-D lifecycle explicitly.
        """
        # --- 1. SETUP (from fixture) ---
        ds_name, config = temporary_datasource
        mind_name = f"test-crud-explicit-{uuid.uuid4()}"
        initial_tables = [config["sample_table"], "home_rentals"]

        # --- 2. CREATE (POST) ---
        logging.info(f"TEST: Explicitly creating mind '{mind_name}'")
        payload = {
            "name": mind_name,
            "provider": "openai",
            "model_name": "gpt-4",
            "parameters": {},
            "datasources": [
                {
                    "name": ds_name,
                    "tables": initial_tables,
                }
            ],
        }
        create_resp = api_client.post(f"{MINDS_API_BASE_URL}/api/v1/minds/", json=payload)
        assert create_resp.status_code == 201, f"Failed to create mind. Response: {create_resp.text}"

        # --- 3. POLL until COMPLETED ---
        logging.info(f"TEST: Polling for mind '{mind_name}' to complete...")
        mind_data = poll_mind_transitions(api_client, mind_name)
        assert mind_data["status"] == "COMPLETED"
        logging.info(f"SUCCESS: Mind '{mind_name}' is COMPLETED.")

        # --- 4. READ (GET) after Create ---
        logging.info(f"TEST: Verifying mind '{mind_name}' exists after create.")
        mind_data = get_and_verify_mind(api_client, mind_name, expected_status=200)

        assert mind_data["name"] == mind_name
        found_ds = mind_data.get("datasources", [])[0]
        assert found_ds["name"] == ds_name

        def normalize_table(t):
            # "public.house_sales" -> "house_sales"
            return t.split(".")[-1].lower()

        # Convert to sets for flexible comparison (fixes AssertionError)
        tables_from_api = set(normalize_table(t) for t in found_ds.get("tables", []))
        expected_tables = set(normalize_table(t) for t in initial_tables)

        # Check if the tables we sent are present in the API response
        assert expected_tables.issubset(tables_from_api), (
            f"Mind tables missing expected tables. Expected: {expected_tables}, Got: {tables_from_api}"
        )
        logging.info(f"SUCCESS: Verified mind '{mind_name}' initial data.")

        # --- 5. UPDATE (PUT) ---
        logging.info(f"TEST: Updating mind '{mind_name}' with new tables")
        updated_tables = ["house_sales", "orders"]

        update_payload = {"datasources": [{"name": ds_name, "tables": updated_tables}]}

        update_resp = api_client.put(f"{MINDS_API_BASE_URL}/api/v1/minds/{mind_name}", json=update_payload)
        assert update_resp.status_code == 200, f"Failed to update mind. Response: {update_resp.text}"

        # Verify the change in the response from the PUT
        response_data = update_resp.json()
        tables_from_api = set(normalize_table(t) for t in response_data["datasources"][0].get("tables", []))
        expected_tables = set(normalize_table(t) for t in updated_tables)

        assert expected_tables.issubset(tables_from_api)

        # Poll until final state after update
        logging.info(f"TEST: Polling for mind '{mind_name}' to complete after update...")
        mind_data = poll_mind_transitions(api_client, mind_name)
        assert mind_data["status"] == "COMPLETED"
        logging.info("SUCCESS: Mind update is COMPLETED.")

        # --- 6. READ (GET) after Update ---
        logging.info(f"TEST: Verifying mind '{mind_name}' data is persisted after update.")
        updated_mind_data = get_and_verify_mind(api_client, mind_name, expected_status=200)

        assert updated_mind_data["name"] == mind_name
        found_ds = updated_mind_data.get("datasources", [])[0]
        assert found_ds["name"] == ds_name

        tables_from_api = set(normalize_table(t) for t in found_ds.get("tables", []))
        expected_tables = set(normalize_table(t) for t in updated_tables)

        assert expected_tables.issubset(tables_from_api), (
            f"Mind tables missing expected tables after update. Expected: {expected_tables}, Got: {tables_from_api}"
        )
        logging.info(f"SUCCESS: Verified mind '{mind_name}' updated data.")

        # --- 7. DELETE (DELETE) ---
        logging.info(f"TEST: Deleting mind '{mind_name}'")
        delete_resp = api_client.delete(f"{MINDS_API_BASE_URL}/api/v1/minds/{mind_name}")
        assert delete_resp.status_code in (200, 204), f"Failed to delete mind. Response: {delete_resp.text}"
        logging.info("SUCCESS: Mind deleted.")

        # --- 8. READ after DELETE (GET) ---
        logging.info(f"TEST: Verifying mind '{mind_name}' is gone (GET 404).")
        get_and_verify_mind(api_client, mind_name, expected_status=404)
        logging.info("SUCCESS: Verified mind deletion (404).")

    def test_list_minds(self, api_client, temporary_mind):
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

            logging.warning(
                f"Attempt {attempt + 1}/{max_retries}: Mind '{mind_name}' not found. Retrying in {delay_seconds}s..."
            )
            time.sleep(delay_seconds)

        pytest.fail(f"Mind '{mind_name}' was not found in the list after {max_retries} attempts.")
