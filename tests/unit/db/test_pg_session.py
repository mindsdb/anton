import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mod(monkeypatch):
    """
    Import the module under test with minimal side effects.
    """
    from importlib import reload

    import minds.db.pg_session as pg_session

    # make sure caches are clean before each test
    pg_session._engines.clear()
    pg_session._session_factories.clear()
    return reload(pg_session)


# ---------- _create_engine ----------


def test__create_engine_happy_path_calls_sqlalchemy_with_pooling(monkeypatch, mod):
    # Arrange: mock SQLAlchemy's create_engine
    engine_obj = object()
    create_engine_mock = MagicMock(return_value=engine_obj)

    # Ensure constants used inside the module have predictable values
    monkeypatch.setattr(mod, "DB_POOL_SIZE", 3)
    monkeypatch.setattr(mod, "DB_MAX_OVERFLOW", 7)
    monkeypatch.setattr(mod, "DB_POOL_TIMEOUT", 55)
    monkeypatch.setattr(mod, "DB_POOL_RECYCLE", 1200)
    monkeypatch.setattr(mod, "DB_POOL_PRE_PING", True)
    monkeypatch.setattr(mod, "create_engine", create_engine_mock)

    uri = "postgresql://test:test@localhost:5432/test"

    # Act
    engine = mod._create_engine(uri)  # type: ignore[arg-type]

    # Assert
    assert engine is engine_obj
    # Verify create_engine called with correct args/kwargs
    assert create_engine_mock.call_args.args[0] == uri
    assert create_engine_mock.call_args.kwargs == {
        "pool_size": 3,
        "max_overflow": 7,
        "pool_timeout": 55,
        "pool_recycle": 1200,
        "pool_pre_ping": True,
    }


def test__create_engine_wraps_errors_as_runtimeerror(monkeypatch, mod):
    monkeypatch.setattr(mod, "create_engine", MagicMock(side_effect=Exception("Kaboom!")))

    with pytest.raises(RuntimeError) as ei:
        mod._create_engine("postgresql://test:test@localhost:5432/test")  # type: ignore[arg-type]

    # message is lower-cased in the wrapper
    assert "creation failed" in str(ei.value)
    assert "kaboom!" in str(ei.value)


# ---------- get_engine (cache by db_uri.name) ----------


def test_get_engine_caches_by_enum_name(monkeypatch, mod):
    engine_a = object()
    engine_b = object()
    create_engine_mock = MagicMock(side_effect=[engine_a, engine_b])
    monkeypatch.setattr(mod, "_create_engine", create_engine_mock)

    # Use string URIs instead of SimpleNamespace objects
    uri_a = "postgresql://test:test@localhost:5432/db_a"
    uri_b = "postgresql://test:test@localhost:5432/db_b"

    e1 = mod.get_engine(uri_a)
    e2 = mod.get_engine(uri_a)
    e3 = mod.get_engine(uri_b)

    assert e1 is e2  # same uri -> cached
    assert e1 is not e3  # different uri -> different engine
    assert create_engine_mock.call_count == 2  # created once per unique uri
    assert create_engine_mock.call_args_list[0].kwargs["db_uri"] == uri_a
    assert create_engine_mock.call_args_list[1].kwargs["db_uri"] == uri_b


# ---------- get_session_factory (cache by engine id) ----------


def test_get_session_factory_builds_sqlmodel_sessionmaker_and_caches(monkeypatch, mod):
    created = []

    def side_effect(*, class_, autocommit, autoflush, bind, expire_on_commit):
        created.append(
            dict(
                class_=class_,
                autocommit=autocommit,
                autoflush=autoflush,
                bind=bind,
                expire_on_commit=expire_on_commit,
            )
        )

        def factory():
            return {"session_for": id(bind)}

        return factory

    monkeypatch.setattr(mod, "sessionmaker", MagicMock(side_effect=side_effect))

    engine1 = object()
    engine2 = object()

    f1a = mod.get_session_factory(engine1)
    f1b = mod.get_session_factory(engine1)
    f2 = mod.get_session_factory(engine2)

    # caching: same engine -> same factory object
    assert f1a is f1b
    # different engine -> different factory
    assert f1a is not f2

    # verify construction args for first call
    assert created[0]["class_"] is mod.SQLModelSession
    assert created[0]["autocommit"] is False
    assert created[0]["autoflush"] is False
    assert created[0]["bind"] is engine1
    assert created[0]["expire_on_commit"] is False

    # sanity: factory returns a session-like marker
    assert f1a() == {"session_for": id(engine1)}
    assert f2() == {"session_for": id(engine2)}


# ---------- get_session (happy path) ----------


def test_get_session_returns_session_from_factory(monkeypatch, mod):
    engine = object()

    def factory():
        class Dummy:
            closed = False

            def close(self):
                self.closed = True

        return Dummy()

    monkeypatch.setattr(mod, "get_engine", MagicMock(return_value=engine))
    monkeypatch.setattr(
        mod,
        "get_session_factory",
        MagicMock(side_effect=lambda e: factory if e is engine else None),
    )

    sess = mod.get_session()
    # on success, we just get back the session and it's not closed
    assert hasattr(sess, "close")
    assert getattr(sess, "closed", False) is False
