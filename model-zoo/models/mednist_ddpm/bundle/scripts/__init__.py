from __future__ import annotations


def inv_metric_cmp_fn(current_metric: float, prev_best: float) -> bool:
    """
    This inverts comparison for those metrics which reduce like loss values, such that the lower one is better.

    Args:
        current_metric: metric value of current round computation.
        prev_best: the best metric value of previous rounds to compare with.
    """
    return current_metric < prev_best
