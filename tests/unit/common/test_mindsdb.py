import types

import pytest

import minds.common.mindsdb as mindsdb_module


class _Leaf:
    def __init__(self, name: str):
        self.name = name


class _Identifier:
    def __init__(self, parts):
        self.parts = list(parts)


class _Cte:
    def __init__(self, query):
        self.query = query


class _Select:
    def __init__(
        self,
        *,
        from_table=None,
        targets=None,
        cte=None,
        where=None,
        group_by=None,
        having=None,
        order_by=None,
    ):
        self.from_table = from_table
        self.targets = [] if targets is None else list(targets)
        self.cte = cte
        self.where = where
        self.group_by = group_by
        self.having = having
        self.order_by = order_by


class _Union:
    def __init__(self, left, right):
        self.left = left
        self.right = right


class _Intersect:  # pragma: no cover - used only to satisfy isinstance checks
    pass


class _Except:  # pragma: no cover - used only to satisfy isinstance checks
    pass


class _Join:
    def __init__(self, *, left, right, condition=None):
        self.left = left
        self.right = right
        self.condition = condition


class _Function:
    def __init__(self, args, from_arg=None):
        self.args = list(args)
        self.from_arg = from_arg


class _BinaryOperation:
    def __init__(self, args):
        self.args = list(args)


class _UnaryOperation:  # pragma: no cover - used only to satisfy isinstance checks
    pass


class _BetweenOperation:  # pragma: no cover - used only to satisfy isinstance checks
    pass


class _Exists:  # pragma: no cover - used only to satisfy isinstance checks
    pass


class _NotExists:  # pragma: no cover - used only to satisfy isinstance checks
    pass


class _WindowFunction:
    def __init__(self, *, function, partition=None, order_by=None):
        self.function = function
        self.partition = partition
        self.order_by = order_by


class _TypeCast:
    def __init__(self, *, arg):
        self.arg = arg


class _Tuple:
    def __init__(self, *, items):
        self.items = list(items)


class _Insert:
    def __init__(self, *, table=None, values=None, from_select=None):
        self.table = table
        self.values = values
        self.from_select = from_select


class _Update:
    def __init__(self, *, table=None, where=None, update_columns=None, from_select=None):
        self.table = table
        self.where = where
        self.update_columns = update_columns
        self.from_select = from_select


class _CreateTable:
    def __init__(self, *, columns=None, name=None, from_select=None):
        self.columns = columns
        self.name = name
        self.from_select = from_select


class _Delete:
    def __init__(self, *, where=None):
        self.where = where


class _OrderBy:
    def __init__(self, *, field=None):
        self.field = field


class _Case:
    def __init__(self, *, rules, default):
        # rules: list[tuple[condition, result]]
        self.rules = list(rules)
        self.default = default


@pytest.fixture
def dummy_ast(monkeypatch):
    """
    `query_traversal` relies on `mindsdb_sql_parser.ast` classes, but we only need
    a tiny subset of behavior for unit tests: `isinstance` checks + a few attributes.
    """
    ast_stub = types.SimpleNamespace(
        Select=_Select,
        Identifier=_Identifier,
        Union=_Union,
        Intersect=_Intersect,
        Except=_Except,
        Join=_Join,
        Function=_Function,
        BinaryOperation=_BinaryOperation,
        UnaryOperation=_UnaryOperation,
        BetweenOperation=_BetweenOperation,
        Exists=_Exists,
        NotExists=_NotExists,
        WindowFunction=_WindowFunction,
        TypeCast=_TypeCast,
        Tuple=_Tuple,
        Insert=_Insert,
        Update=_Update,
        CreateTable=_CreateTable,
        Delete=_Delete,
        OrderBy=_OrderBy,
        Case=_Case,
    )
    monkeypatch.setattr(mindsdb_module, "ast", ast_stub, raising=True)
    monkeypatch.setattr(mindsdb_module, "Identifier", ast_stub.Identifier, raising=True)
    return ast_stub


def test_extract_database_engines_from_query_uses_is_table_identifier_detection(dummy_ast):
    """
    Mirrors the main production usage in `minds/services/conversations.py`:
    traverse a parsed query and extract the database engines involved based on
    table identifiers.
    """
    query = dummy_ast.Select(
        from_table=dummy_ast.Join(
            left=dummy_ast.Identifier(["db1", "table1"]),
            right=dummy_ast.Identifier(["db2", "table2"]),
            condition=None,
        ),
        targets=[],
    )

    engines_by_db = {"db1": "postgres", "db2": "mssql"}

    def get_database(db_name: str):
        return types.SimpleNamespace(engine=engines_by_db[db_name])

    mindsdb_client = types.SimpleNamespace(databases=types.SimpleNamespace(get=get_database))

    database_engines = mindsdb_module.extract_database_engines_from_select(query, mindsdb_client)
    assert set(database_engines) == {"postgres", "mssql"}


def test_extract_database_engines_from_select_excludes_cte_names(dummy_ast):
    """
    Ensure CTE names are not treated as databases while still traversing into the CTE query.
    """
    cte_query = dummy_ast.Select(from_table=dummy_ast.Identifier(["db1", "table1"]), targets=[])
    cte = types.SimpleNamespace(name=dummy_ast.Identifier(["cte1"]), query=cte_query)
    query = dummy_ast.Select(
        from_table=dummy_ast.Join(
            left=dummy_ast.Identifier(["cte1"]),
            right=dummy_ast.Identifier(["db2", "table2"]),
            condition=None,
        ),
        targets=[],
        cte=[cte],
    )

    engines_by_db = {"db1": "postgres", "db2": "mssql"}

    def resolve_engine(db_name: str) -> str:
        assert db_name != "cte1"
        return engines_by_db[db_name]

    mindsdb_client = types.SimpleNamespace(
        databases=types.SimpleNamespace(get=lambda db_name: types.SimpleNamespace(engine=resolve_engine(db_name)))
    )
    database_engines = mindsdb_module.extract_database_engines_from_select(
        query, mindsdb_client, require_qualified_table=False, exclude_cte_names=True
    )

    assert set(database_engines) == {"postgres", "mssql"}


def test_extract_databases_from_select_collects_unique_db_names(dummy_ast):
    query = dummy_ast.Select(
        from_table=dummy_ast.Join(
            left=dummy_ast.Identifier(["db1", "table1"]),
            right=dummy_ast.Identifier(["db2", "table2"]),
            condition=None,
        ),
        targets=[],
    )

    databases = mindsdb_module.extract_databases_from_select(query)
    assert set(databases) == {"db1", "db2"}
    assert len(databases) == 2


def test_extract_databases_from_select_ignores_unqualified_tables_by_default(dummy_ast):
    query = dummy_ast.Select(from_table=dummy_ast.Identifier(["table_only"]), targets=[])
    assert mindsdb_module.extract_databases_from_select(query) == []


def test_extract_databases_from_select_allows_unqualified_tables_when_configured(dummy_ast):
    query = dummy_ast.Select(from_table=dummy_ast.Identifier(["db1"]), targets=[])
    assert mindsdb_module.extract_databases_from_select(query, require_qualified_table=False) == ["db1"]


def test_extract_databases_from_select_excludes_cte_names(dummy_ast):
    cte_query = dummy_ast.Select(from_table=dummy_ast.Identifier(["db1", "table1"]), targets=[])
    cte = types.SimpleNamespace(name=dummy_ast.Identifier(["cte1"]), query=cte_query)
    query = dummy_ast.Select(
        from_table=dummy_ast.Join(
            left=dummy_ast.Identifier(["cte1"]),
            right=dummy_ast.Identifier(["db2", "table2"]),
            condition=None,
        ),
        targets=[],
        cte=[cte],
    )

    databases = mindsdb_module.extract_databases_from_select(
        query, exclude_cte_names=True, require_qualified_table=False
    )
    assert set(databases) == {"db1", "db2"}
    assert len(databases) == 2


def test_query_traversal_set_operation_replaces_left_and_right_and_sets_parent_query(dummy_ast):
    left = _Leaf("left")
    right = _Leaf("right")
    left2 = _Leaf("left2")
    right2 = _Leaf("right2")
    root = dummy_ast.Union(left=left, right=right)

    calls = []

    def callback(node, **kwargs):
        calls.append(
            {
                "node": node,
                "parent_query": kwargs.get("parent_query"),
                "callstack": kwargs.get("callstack"),
                "is_table": kwargs.get("is_table"),
                "is_target": kwargs.get("is_target"),
            }
        )
        if node is left:
            return left2
        if node is right:
            return right2
        return None

    out = mindsdb_module.query_traversal(root, callback)

    # Root is mutated in-place; traversal returns None to keep original node.
    assert out is None
    assert root.left is left2
    assert root.right is right2

    # Child callbacks should receive `parent_query` as the set-operation node.
    left_call = next(c for c in calls if c["node"] is left)
    right_call = next(c for c in calls if c["node"] is right)
    assert left_call["parent_query"] is root
    assert right_call["parent_query"] is root

    # Child callbacks should see the parent in the callstack.
    assert left_call["callstack"] == [root]
    assert right_call["callstack"] == [root]


def test_query_traversal_function_traverses_args_and_from_arg(dummy_ast):
    arg1 = _Leaf("a1")
    arg2 = _Leaf("a2")
    from_arg = _Leaf("from")
    arg2_replaced = _Leaf("a2r")
    from_replaced = _Leaf("fromr")
    root = dummy_ast.Function(args=[arg1, arg2], from_arg=from_arg)

    calls = []

    def callback(node, **kwargs):
        calls.append((node, kwargs.get("parent_query"), kwargs.get("callstack")))
        if node is arg2:
            return arg2_replaced
        if node is from_arg:
            return from_replaced
        return None

    out = mindsdb_module.query_traversal(root, callback)

    assert out is None
    assert root.args == [arg1, arg2_replaced]
    assert root.from_arg is from_replaced

    # For args/from_arg traversal, `parent_query` is forwarded from the caller (None at root),
    # but callstack still includes the parent node.
    arg2_call = next(c for c in calls if c[0] is arg2)
    from_call = next(c for c in calls if c[0] is from_arg)
    assert arg2_call[1] is None
    assert from_call[1] is None
    assert arg2_call[2] == [root]
    assert from_call[2] == [root]


def test_query_traversal_select_traverses_all_children_and_expands_target_lists(dummy_ast):
    from_table = _Leaf("from")
    t1 = _Leaf("t1")
    t2 = _Leaf("t2")
    where = _Leaf("where")
    g1 = _Leaf("g1")
    having = _Leaf("having")
    ob1 = _Leaf("ob1")
    cte_query = _Leaf("cte_query")
    cte = _Cte(query=cte_query)

    select = dummy_ast.Select(
        from_table=from_table,
        targets=[t1, t2],
        cte=[cte],
        where=where,
        group_by=[g1],
        having=having,
        order_by=[ob1],
    )

    t1a = _Leaf("t1a")
    t1b = _Leaf("t1b")

    calls = []

    def callback(node, **kwargs):
        calls.append(
            (
                node,
                kwargs.get("is_table"),
                kwargs.get("is_target"),
                kwargs.get("parent_query"),
                kwargs.get("callstack"),
            )
        )
        if node is t1:
            # Ensure Select.targets supports callback returning a list (expanded into targets)
            return [t1a, t1b]
        return None

    out = mindsdb_module.query_traversal(select, callback)
    assert out is None

    # target list should be expanded
    assert select.targets == [t1a, t1b, t2]

    # Ensure from_table traversal sets is_table=True and parent_query=select
    from_call = next(c for c in calls if c[0] is from_table)
    assert from_call[1] is True
    assert from_call[2] is False
    assert from_call[3] is select
    assert from_call[4] == [select]

    # Ensure targets traversal sets is_target=True and parent_query=select
    t2_call = next(c for c in calls if c[0] is t2)
    assert t2_call[1] is False
    assert t2_call[2] is True
    assert t2_call[3] is select
    assert t2_call[4] == [select]

    # Ensure other fields are traversed (where/group_by/having/order_by/cte.query)
    assert any(c[0] is where for c in calls)
    assert any(c[0] is g1 for c in calls)
    assert any(c[0] is having for c in calls)
    assert any(c[0] is ob1 for c in calls)
    assert any(c[0] is cte_query for c in calls)


def test_query_traversal_join_traverses_left_right_as_tables_and_condition(dummy_ast):
    left = _Leaf("left")
    right = _Leaf("right")
    condition = _Leaf("cond")
    join = dummy_ast.Join(left=left, right=right, condition=condition)

    left2 = _Leaf("left2")
    right2 = _Leaf("right2")
    cond2 = _Leaf("cond2")

    calls = []

    def callback(node, **kwargs):
        calls.append((node, kwargs.get("is_table"), kwargs.get("parent_query"), kwargs.get("callstack")))
        if node is left:
            return left2
        if node is right:
            return right2
        if node is condition:
            return cond2
        return None

    # parent_query passed into Join children should be forwarded from caller.
    parent = _Leaf("parent_query")
    out = mindsdb_module.query_traversal(join, callback, parent_query=parent)
    assert out is None
    assert join.left is left2
    assert join.right is right2
    assert join.condition is cond2

    left_call = next(c for c in calls if c[0] is left)
    right_call = next(c for c in calls if c[0] is right)
    cond_call = next(c for c in calls if c[0] is condition)

    assert left_call[1] is True
    assert right_call[1] is True
    assert left_call[2] is parent
    assert right_call[2] is parent
    assert cond_call[2] is parent
    assert left_call[3] == [join]
    assert right_call[3] == [join]
    assert cond_call[3] == [join]


def test_query_traversal_window_function_traverses_children_and_ignores_function_replacement(dummy_ast):
    func = _Leaf("func")
    p1 = _Leaf("p1")
    ob1 = _Leaf("ob1")
    wf = dummy_ast.WindowFunction(function=func, partition=[p1], order_by=[ob1])

    func2 = _Leaf("func2")
    p2 = _Leaf("p2")

    def callback(node, **_kwargs):
        if node is func:
            # replacement should be ignored (code doesn't assign)
            return func2
        if node is p1:
            return p2
        return None

    out = mindsdb_module.query_traversal(wf, callback)
    assert out is None
    assert wf.function is func
    assert wf.partition == [p2]
    assert wf.order_by == [ob1]


def test_query_traversal_type_cast_replaces_arg(dummy_ast):
    arg = _Leaf("arg")
    cast = dummy_ast.TypeCast(arg=arg)
    arg2 = _Leaf("arg2")

    def callback(node, **_kwargs):
        return arg2 if node is arg else None

    out = mindsdb_module.query_traversal(cast, callback)
    assert out is None
    assert cast.arg is arg2


def test_query_traversal_tuple_replaces_items(dummy_ast):
    a = _Leaf("a")
    b = _Leaf("b")
    tup = dummy_ast.Tuple(items=[a, b])
    b2 = _Leaf("b2")

    def callback(node, **_kwargs):
        return b2 if node is b else None

    out = mindsdb_module.query_traversal(tup, callback)
    assert out is None
    assert tup.items == [a, b2]


def test_query_traversal_insert_traverses_table_values_and_from_select(dummy_ast):
    table = _Leaf("table")
    v11 = _Leaf("v11")
    v12 = _Leaf("v12")
    from_select = _Leaf("from_select")
    ins = dummy_ast.Insert(table=table, values=[[v11, v12]], from_select=from_select)

    table2 = _Leaf("table2")
    v12_2 = _Leaf("v12_2")
    from_select2 = _Leaf("from_select2")

    def callback(node, **_kwargs):
        if node is table:
            return table2
        if node is v12:
            return v12_2
        if node is from_select:
            return from_select2
        return None

    out = mindsdb_module.query_traversal(ins, callback)
    assert out is None
    assert ins.table is table2
    assert ins.values == [[v11, v12_2]]
    assert ins.from_select is from_select2


def test_query_traversal_update_traverses_table_where_update_columns_and_from_select(dummy_ast):
    table = _Leaf("table")
    where = _Leaf("where")
    v1 = _Leaf("v1")
    v2 = _Leaf("v2")
    from_select = _Leaf("from_select")
    upd = dummy_ast.Update(table=table, where=where, update_columns={"a": v1, "b": v2}, from_select=from_select)

    v1r = _Leaf("v1r")
    from_select2 = _Leaf("from_select2")

    def callback(node, **_kwargs):
        if node is v1:
            return v1r
        if node is from_select:
            return from_select2
        return None

    out = mindsdb_module.query_traversal(upd, callback)
    assert out is None
    assert upd.update_columns == {"a": v1r, "b": v2}
    assert upd.from_select is from_select2


def test_query_traversal_create_table_traverses_columns_name_and_from_select(dummy_ast):
    col1 = _Leaf("col1")
    name = _Leaf("name")
    from_select = _Leaf("from_select")
    ct = dummy_ast.CreateTable(columns=[col1], name=name, from_select=from_select)

    name2 = _Leaf("name2")

    calls = []

    def callback(node, **kwargs):
        calls.append((node, kwargs.get("is_table")))
        if node is name:
            return name2
        return None

    out = mindsdb_module.query_traversal(ct, callback)
    assert out is None
    assert ct.columns == [col1]
    assert ct.name is name2
    assert ct.from_select is from_select

    name_call = next(c for c in calls if c[0] is name)
    assert name_call[1] is True


def test_query_traversal_delete_traverses_where(dummy_ast):
    where = _Leaf("where")
    where2 = _Leaf("where2")
    d = dummy_ast.Delete(where=where)

    def callback(node, **_kwargs):
        return where2 if node is where else None

    out = mindsdb_module.query_traversal(d, callback)
    assert out is None
    assert d.where is where2


def test_query_traversal_order_by_traverses_field(dummy_ast):
    f = _Leaf("f")
    f2 = _Leaf("f2")
    ob = dummy_ast.OrderBy(field=f)

    def callback(node, **_kwargs):
        return f2 if node is f else None

    out = mindsdb_module.query_traversal(ob, callback)
    assert out is None
    assert ob.field is f2


def test_query_traversal_case_traverses_rules_and_default(dummy_ast):
    c1 = _Leaf("c1")
    r1 = _Leaf("r1")
    default = _Leaf("default")
    case = dummy_ast.Case(rules=[(c1, r1)], default=default)

    c1r = _Leaf("c1r")
    default2 = _Leaf("default2")

    def callback(node, **_kwargs):
        if node is c1:
            return c1r
        if node is default:
            return default2
        return None

    out = mindsdb_module.query_traversal(case, callback)
    assert out is None
    assert case.rules == [[c1r, r1]]
    assert case.default is default2


def test_query_traversal_list_returns_new_list_and_replaces_items(dummy_ast):
    a = _Leaf("a")
    b = _Leaf("b")
    b2 = _Leaf("b2")

    def callback(node, **_kwargs):
        return b2 if node is b else None

    out = mindsdb_module.query_traversal([a, b], callback)
    assert out == [a, b2]
