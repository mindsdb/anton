from __future__ import annotations

from anton.core.estimator import TimeEstimator


class TestTimeEstimator:
    def test_no_history_returns_none(self):
        est = TimeEstimator()
        assert est.estimate("read_file") is None

    def test_single_record(self):
        est = TimeEstimator()
        est.record("read_file", 0.5)
        assert est.estimate("read_file") == 0.5

    def test_average_of_multiple_records(self):
        est = TimeEstimator()
        est.record("read_file", 1.0)
        est.record("read_file", 2.0)
        est.record("read_file", 3.0)
        assert est.estimate("read_file") == 2.0

    def test_separate_skills_tracked_independently(self):
        est = TimeEstimator()
        est.record("read_file", 1.0)
        est.record("write_file", 5.0)
        assert est.estimate("read_file") == 1.0
        assert est.estimate("write_file") == 5.0


class TestEstimatePlan:
    def test_no_history_returns_none(self):
        est = TimeEstimator()
        assert est.estimate_plan(["read_file", "write_file"]) is None

    def test_all_known(self):
        est = TimeEstimator()
        est.record("read_file", 1.0)
        est.record("write_file", 2.0)
        assert est.estimate_plan(["read_file", "write_file"]) == 3.0

    def test_partial_known(self):
        est = TimeEstimator()
        est.record("read_file", 1.0)
        # write_file has no history
        result = est.estimate_plan(["read_file", "write_file"])
        assert result == 1.0

    def test_empty_plan(self):
        est = TimeEstimator()
        assert est.estimate_plan([]) is None
