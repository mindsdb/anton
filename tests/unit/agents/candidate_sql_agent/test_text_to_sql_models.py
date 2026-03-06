from minds.agents.candidate_sql_agent.text_to_sql_agents.models import (
    AcquiredKnowledge,
    AcquiredKnowledgeItem,
    DataCatalogSubset,
    QueryAttempt,
    QueryPlan,
    QueryPlanStep,
    QueryPlanStepType,
)


def test_query_attempt_to_string_includes_optional_fields():
    a1 = QueryAttempt(query="SELECT 1")
    assert "SELECT 1" in a1.to_string()

    a2 = QueryAttempt(query="SELECT 1", error="boom")
    assert "Error: boom" in a2.to_string()

    a3 = QueryAttempt(query="SELECT 1", result="ok")
    assert "Result: ok" in a3.to_string()


def test_acquired_knowledge_to_string_contains_all_items():
    ak = AcquiredKnowledge()
    ak.add_item(AcquiredKnowledgeItem(step="s1", attempts=[QueryAttempt(query="q1", result="r1")]))
    ak.add_item(AcquiredKnowledgeItem(step="s2", attempts=[QueryAttempt(query="q2", error="e2")]))

    s = ak.to_string()
    assert "Step: s1" in s
    assert "Query: q1" in s
    assert "Result: r1" in s
    assert "Step: s2" in s
    assert "Error: e2" in s


def test_plan_and_subset_to_string_are_stable():
    subset = DataCatalogSubset(datasources=["ds"], tables=["ds.t"])
    assert "Datasources: ds" in subset.to_string()
    assert "Tables: ds.t" in subset.to_string()

    step = QueryPlanStep(description="d", type=QueryPlanStepType.FINAL, data_catalog_subset=subset)
    plan = QueryPlan(steps=[step])

    s = plan.to_string()
    assert "Description: d" in s
    assert "Type: final" in s
    assert "Tables: ds.t" in s
