import pytest
from anton_state.schema import Attr, StateSchema
from anton_state.factory import open_store
from anton_state.sqlite_driver import SQLiteDriver
from anton_state.dynamo_driver import DynamoDBDriver

M = StateSchema(pk=Attr(name="pk"), sk=Attr(name="sk"))


def test_none_state_selects_sqlite(tmp_path):
    store = open_store(M, state=None, local_path=str(tmp_path / "s.db"))
    assert isinstance(store._driver, SQLiteDriver)


def test_local_path_from_env(tmp_path, monkeypatch):
    p = str(tmp_path / "envstate.db")
    monkeypatch.setenv("ANTON_ARTIFACT_STATE_PATH", p)
    store = open_store(M, state=None)
    assert isinstance(store._driver, SQLiteDriver)
    assert store._driver.path == p


def test_schema_loaded_from_manifest_file_when_omitted(tmp_path, monkeypatch):
    # single source: don't pass schema — the factory reads the manifest file
    man = tmp_path / "state_manifest.json"
    M.to_manifest(man)
    monkeypatch.setenv("ANTON_STATE_MANIFEST", str(man))
    store = open_store(state=None, local_path=str(tmp_path / "s.db"))
    assert store.schema == M


def test_state_dict_selects_dynamo():
    from moto import mock_aws
    import boto3
    state = {
        "table": "t1", "region": "us-east-1",
        "credentials": {"key": "k", "secret": "s", "token": "t"},
    }
    with mock_aws():
        boto3.client("dynamodb", region_name="us-east-1").create_table(
            TableName="t1",
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}],
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        store = open_store(M, state=state)
        assert isinstance(store._driver, DynamoDBDriver)
