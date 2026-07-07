"""Unit tests for the eval metrics (eval_ortho.py) — pure functions, no LLM calls."""
import eval_ortho as e


def test_parse_grades_reads_json_array():
    assert e._parse_grades("[2,0,1,2,0]", 5) == [2, 0, 1, 2, 0]


def test_parse_grades_pads_short_replies():
    assert e._parse_grades("grades: 2, 1", 5) == [2, 1, 0, 0, 0]


def test_parse_grades_clamps_out_of_range():
    assert e._parse_grades("[5, -1, 3]", 3) == [2, 0, 2]


def test_parse_grades_handles_no_numbers():
    assert e._parse_grades("no gradeable content", 3) == [0, 0, 0]


def test_ndcg_rewards_ordering_precision_does_not():
    ideal = e._graded_metrics([2, 1, 0])   # most-relevant first
    worst = e._graded_metrics([0, 1, 2])   # same set, reversed
    assert ideal["ndcg_at_k"] == 1.0
    assert worst["ndcg_at_k"] < ideal["ndcg_at_k"]      # nDCG sees the reorder
    assert ideal["precision_at_k"] == worst["precision_at_k"]  # Precision cannot


def test_graded_metrics_all_irrelevant_is_zero():
    assert e._graded_metrics([0, 0, 0]) == {"precision_at_k": 0.0, "ndcg_at_k": 0.0}


def test_hit_and_mrr_from_ranks():
    m = e._metrics([1, 0, 2, 0])   # ranks: 1, miss, 2, miss
    assert m["hit_at_k"] == 0.5
    assert m["mrr"] == round((1 / 1 + 1 / 2) / 4, 3)   # 0.375
