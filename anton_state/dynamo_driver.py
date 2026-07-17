"""Cloud driver: DynamoDB via boto3 with per-invocation STS credentials.

ConsistentRead=True on the base table (read-after-write as locally); GSIs are
always eventual. Throttle → StateThrottled with limited retries, so the
OnDemandThroughput cap does not manifest as a hang.
"""
from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key
from botocore.config import Config
from botocore.exceptions import ClientError

from .base import _VERSION_ATTR
from .errors import ConditionalCheckFailed, StateThrottled, StateValidationError
from .schema import StateSchema
from .validation import validate_item, validate_key

_THROTTLE_CODES = {
    "ProvisionedThroughputExceededException",
    "ThrottlingException",
    "RequestLimitExceeded",
}


def _to_dynamo(value):
    """Recursively convert float → Decimal(str(x)) (boto3 resource rejects float)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: _to_dynamo(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_dynamo(v) for v in value]
    return value


def _from_dynamo(value):
    """Recursively convert Decimal → int (if integral) / float, so plain py numbers go out."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {k: _from_dynamo(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_dynamo(v) for v in value]
    return value


class DynamoDBDriver:
    def __init__(self, table: str, region: str, credentials: dict, schema: StateSchema):
        self.schema = schema
        self._pk = schema.pk.name
        self._sk = schema.sk.name if schema.sk else None
        self._ttl = schema.ttl_attribute
        self._gsi = {g.name: g for g in schema.gsis}
        resource = boto3.resource(
            "dynamodb",
            region_name=region,
            aws_access_key_id=credentials["key"],
            aws_secret_access_key=credentials["secret"],
            aws_session_token=credentials.get("token"),
            config=Config(retries={"max_attempts": 2, "mode": "standard"}),
        )
        self._table = resource.Table(table)

    def _key(self, pk: str, sk: str | None) -> dict:
        k = {self._pk: pk}
        if self._sk is not None:
            k[self._sk] = sk
        return k

    def _fresh(self, item: dict | None) -> dict | None:
        if item is None:
            return None
        if self._ttl and self._ttl in item and float(item[self._ttl]) < time.time():
            return None
        return _from_dynamo(item)

    def get(self, pk: str, sk: str | None, *, consistent: bool) -> dict | None:
        validate_key(pk, sk, self.schema)
        try:
            resp = self._table.get_item(Key=self._key(pk, sk), ConsistentRead=consistent)
        except ClientError as e:
            raise self._map(e)
        return self._fresh(resp.get("Item"))

    def put(self, item: dict, *, if_not_exists: bool, if_version: int | None) -> None:
        validate_item(item, self.schema)
        kwargs: dict[str, Any] = {"Item": _to_dynamo(item)}
        if if_not_exists:
            kwargs["ConditionExpression"] = "attribute_not_exists(#pk)"
            kwargs["ExpressionAttributeNames"] = {"#pk": self._pk}
        elif if_version is not None:
            kwargs["ConditionExpression"] = "#v = :v"
            kwargs["ExpressionAttributeNames"] = {"#v": _VERSION_ATTR}
            kwargs["ExpressionAttributeValues"] = {":v": if_version}
        try:
            self._table.put_item(**kwargs)
        except ClientError as e:
            raise self._map(e)

    def delete(self, pk: str, sk: str | None, *, if_version: int | None) -> None:
        validate_key(pk, sk, self.schema)
        kwargs: dict[str, Any] = {"Key": self._key(pk, sk)}
        if if_version is not None:
            kwargs["ConditionExpression"] = "#v = :v"
            kwargs["ExpressionAttributeNames"] = {"#v": _VERSION_ATTR}
            kwargs["ExpressionAttributeValues"] = {":v": if_version}
        try:
            self._table.delete_item(**kwargs)
        except ClientError as e:
            raise self._map(e)

    def query(
        self,
        pk: str,
        *,
        sk_prefix: str | None,
        index: str | None,
        filters: dict[str, Any] | None,
        consistent: bool,
        limit: int | None,
    ) -> list[dict]:
        kwargs: dict[str, Any] = {}
        if index is not None:
            if index not in self._gsi:
                raise StateValidationError(
                    f"unknown index '{index}' (declared: {sorted(self._gsi)})"
                )
            gsi = self._gsi[index]
            cond = Key(gsi.pk.name).eq(pk)
            if sk_prefix is not None and gsi.sk is not None:
                cond = cond & Key(gsi.sk.name).begins_with(sk_prefix)
            kwargs["IndexName"] = index
            # ConsistentRead is not allowed on a GSI — do not pass it.
        else:
            cond = Key(self._pk).eq(pk)
            if sk_prefix is not None and self._sk is not None:
                cond = cond & Key(self._sk).begins_with(sk_prefix)
            kwargs["ConsistentRead"] = consistent
        kwargs["KeyConditionExpression"] = cond
        if limit is not None:
            kwargs["Limit"] = limit  # page-size hint; we top up via pagination below

        if filters:
            from boto3.dynamodb.conditions import Attr as DAttr
            fexpr = None
            for k, v in filters.items():
                clause = DAttr(k).eq(v)
                fexpr = clause if fexpr is None else (fexpr & clause)
            kwargs["FilterExpression"] = fexpr

        now = time.time()
        out: list[dict] = []
        try:
            while True:
                resp = self._table.query(**kwargs)
                for item in resp.get("Items", []):
                    if self._ttl and self._ttl in item and float(item[self._ttl]) < now:
                        continue
                    out.append(_from_dynamo(item))
                    if limit is not None and len(out) >= limit:
                        return out[:limit]
                lek = resp.get("LastEvaluatedKey")
                if not lek:
                    break
                kwargs["ExclusiveStartKey"] = lek
        except ClientError as e:
            raise self._map(e)
        return out

    def _map(self, e: ClientError) -> Exception:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "ConditionalCheckFailedException":
            return ConditionalCheckFailed(str(e))
        if code in _THROTTLE_CODES:
            return StateThrottled(str(e))
        return e
