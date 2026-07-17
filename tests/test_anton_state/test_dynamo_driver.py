import time
import boto3
import pytest
from moto import mock_aws
from anton_state.schema import Attr, Index, StateSchema
from anton_state.errors import ConditionalCheckFailed
from anton_state.dynamo_driver import DynamoDBDriver

M = StateSchema(
    pk=Attr(name="pk"),
    sk=Attr(name="sk"),
    gsis=[Index(name="by_user", pk=Attr(name="user_id"))],
    ttl_attribute="expires_at",
)
CREDS = {"key": "AKIA", "secret": "secret", "token": "tok", "exp": "2999-01-01T00:00:00Z"}


def _create_table(name):
    ddb = boto3.client("dynamodb", region_name="us-east-1")
    ddb.create_table(
        TableName=name,
        KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}],
        AttributeDefinitions=[
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
            {"AttributeName": "user_id", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
        GlobalSecondaryIndexes=[{
            "IndexName": "by_user",
            "KeySchema": [{"AttributeName": "user_id", "KeyType": "HASH"}],
            "Projection": {"ProjectionType": "ALL"},
        }],
    )
    ddb.get_waiter("table_exists").wait(TableName=name)


@pytest.fixture
def drv():
    with mock_aws():
        _create_table("t1")
        yield DynamoDBDriver("t1", "us-east-1", CREDS, M)


def test_put_get_roundtrip(drv):
    drv.put({"pk": "u1", "sk": "profile", "name": "Alice"}, if_not_exists=False, if_version=None)
    assert drv.get("u1", "profile", consistent=True)["name"] == "Alice"


def test_get_missing_none(drv):
    assert drv.get("nope", "x", consistent=True) is None


def test_if_not_exists_conflict(drv):
    drv.put({"pk": "u1", "sk": "s"}, if_not_exists=True, if_version=None)
    with pytest.raises(ConditionalCheckFailed):
        drv.put({"pk": "u1", "sk": "s"}, if_not_exists=True, if_version=None)


def test_optimistic_lock(drv):
    drv.put({"pk": "u1", "sk": "s", "_v": 1}, if_not_exists=False, if_version=None)
    drv.put({"pk": "u1", "sk": "s", "_v": 2}, if_not_exists=False, if_version=1)
    with pytest.raises(ConditionalCheckFailed):
        drv.put({"pk": "u1", "sk": "s", "_v": 3}, if_not_exists=False, if_version=1)


def test_ttl_filtered_on_read(drv):
    drv.put({"pk": "u1", "sk": "s", "expires_at": time.time() - 5}, if_not_exists=False, if_version=None)
    assert drv.get("u1", "s", consistent=True) is None


def test_query_prefix(drv):
    drv.put({"pk": "u1", "sk": "msg#1"}, if_not_exists=False, if_version=None)
    drv.put({"pk": "u1", "sk": "msg#2"}, if_not_exists=False, if_version=None)
    drv.put({"pk": "u1", "sk": "note#1"}, if_not_exists=False, if_version=None)
    rows = drv.query("u1", sk_prefix="msg#", index=None, filters=None, consistent=True, limit=None)
    assert sorted(r["sk"] for r in rows) == ["msg#1", "msg#2"]


def test_query_gsi(drv):
    drv.put({"pk": "p1", "sk": "s1", "user_id": "u9"}, if_not_exists=False, if_version=None)
    drv.put({"pk": "p2", "sk": "s2", "user_id": "u9"}, if_not_exists=False, if_version=None)
    rows = drv.query("u9", sk_prefix=None, index="by_user", filters=None, consistent=False, limit=None)
    assert sorted(r["pk"] for r in rows) == ["p1", "p2"]


def test_number_roundtrip_float_and_int(drv):
    # float (from time.time()) and int must both be written (float→Decimal) and returned as py numbers
    drv.put({"pk": "u1", "sk": "s", "ratio": 3.14, "count": 7}, if_not_exists=False, if_version=None)
    got = drv.get("u1", "s", consistent=True)
    assert got["ratio"] == 3.14 and isinstance(got["ratio"], float)
    assert got["count"] == 7 and isinstance(got["count"], int)


def test_unknown_index_raises_validation(drv):
    from anton_state.errors import StateValidationError
    with pytest.raises(StateValidationError):
        drv.query("u1", sk_prefix=None, index="by_ghost", filters=None, consistent=False, limit=None)
