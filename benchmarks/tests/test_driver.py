"""Tests for the ladder verdict every external-tool driver is judged by.

The verdict is the one piece of the harness that three tools share and that
nothing else checks: a campaign is days of GPU time, so a rule that quietly
changed meaning would be found in the figures rather than in a run. It is a pure
function of the traces and the thresholds, which is what makes it testable at
all — no solver, no image store, no results file.

The project has no pytest dependency, so this is written to run either way::

    pixi run python tests/test_driver.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchkit.driver import collect_traces, ladder_verdict

LADDER = [1, 10, 100, 1000]
TARGET = 0.001
TIMEOUT = 60.0
TAU_REF = 2.0


def _trace(pairs):
    """One repeat's ladder: ``(knob, tau, time_s)`` for each rung it reached."""
    return [(knob, tau, time_s) for knob, tau, time_s in pairs]


def test_target_reached_at_an_early_rung():
    """A rung inside the target ends the ladder there and reports why."""
    trace = _trace([(1, 3.0, 0.1), (10, 2.0005, 0.5)])
    rows, verdict = ladder_verdict([trace], LADDER, TAU_REF, TARGET, TIMEOUT)

    assert verdict == "target_reached"
    assert [r.knob for r in rows] == [1, 10]
    assert [r.stop_reason for r in rows] == ["", "target_reached"]
    assert rows[1].rel_error == abs(2.0005 - TAU_REF) / TAU_REF
    assert rows[1].time_s == 0.5


def test_timeout_beats_the_remaining_rungs():
    """A rung over the timeout stops the ladder even with rungs left to walk."""
    trace = _trace([(1, 3.0, 1.0), (10, 2.5, TIMEOUT + 1.0), (100, 2.4, 999.0)])
    rows, verdict = ladder_verdict([trace], LADDER, TAU_REF, TARGET, TIMEOUT)

    assert verdict == "timeout"
    assert [r.knob for r in rows] == [1, 10]
    assert rows[-1].stop_reason == "timeout"


def test_ladder_exhausted_at_the_last_rung():
    """Reaching the top of the ladder without the target is its own verdict."""
    trace = _trace([(knob, 3.0, 0.1) for knob in LADDER])
    rows, verdict = ladder_verdict([trace], LADDER, TAU_REF, TARGET, TIMEOUT)

    assert verdict == "ladder_exhausted"
    assert len(rows) == len(LADDER)
    assert rows[-1].stop_reason == "ladder_exhausted"


def test_repeats_diverged_aggregates_only_the_common_prefix():
    """Repeats that stop a rung apart aggregate over the shorter one, and say so."""
    long_run = _trace([(1, 3.0, 0.1), (10, 2.5, 0.5), (100, 2.0005, 1.0)])
    short_run = _trace([(1, 3.0, 0.2), (10, 2.4, 0.6)])
    rows, verdict = ladder_verdict([long_run, short_run], LADDER, TAU_REF, TARGET, TIMEOUT)

    assert verdict == "repeats_diverged"
    assert [r.knob for r in rows] == [1, 10]
    assert rows[-1].tau == 2.45  # median of the two repeats at that rung
    assert rows[-1].time_s == 0.55
    assert rows[-1].repeats == 2


def test_tau_spread_is_nan_with_one_repeat_and_a_ratio_with_several():
    """A single repeat cannot state a spread; several state it relative to tau."""
    one = _trace([(knob, 3.0, 0.1) for knob in LADDER])
    rows, _ = ladder_verdict([one], LADDER, TAU_REF, TARGET, TIMEOUT)
    assert all(r.tau_spread != r.tau_spread for r in rows)  # NaN is not itself
    assert all(r.repeats == 1 for r in rows)

    other = _trace([(knob, 3.2, 0.1) for knob in LADDER])
    third = _trace([(knob, 3.1, 0.1) for knob in LADDER])
    rows, _ = ladder_verdict([one, other, third], LADDER, TAU_REF, TARGET, TIMEOUT)
    assert all(r.tau == 3.1 for r in rows)
    assert all(abs(r.tau_spread - (3.2 - 3.0) / 3.1) < 1e-12 for r in rows)
    assert all(r.repeats == 3 for r in rows)


def test_agreeing_repeats_report_a_spread_of_zero():
    """Zero is a claim that the repeats agreed exactly, not a missing value."""
    trace = _trace([(knob, 3.0, 0.1) for knob in LADDER])
    rows, _ = ladder_verdict([trace, list(trace)], LADDER, TAU_REF, TARGET, TIMEOUT)

    assert all(r.tau_spread == 0.0 for r in rows)


def test_repeats_that_disagree_about_the_ladder_are_a_bug_not_a_verdict():
    """Two repeats reporting different knobs at one rung must not be aggregated."""
    trace = _trace([(1, 3.0, 0.1), (10, 2.5, 0.5)])
    mismatched = _trace([(1, 3.0, 0.1), (100, 2.5, 0.5)])
    try:
        ladder_verdict([trace, mismatched], LADDER, TAU_REF, TARGET, TIMEOUT)
    except AssertionError as exc:
        assert "repeats disagree about the ladder" in str(exc)
    else:
        raise AssertionError("mismatched ladders were aggregated silently")


def test_a_slow_first_repeat_abandons_the_rest():
    """The repeat budget is spent only on cases cheap enough to afford it."""
    calls = []

    def trace():
        calls.append(len(calls))
        return _trace([(1, 3.0, 100.0)])

    assert len(collect_traces(trace, 3, repeat_threshold=10.0)) == 1
    assert len(calls) == 1


def test_a_fast_first_repeat_spends_the_whole_budget():
    """Under the threshold every repeat runs, so the spread means something."""
    def trace():
        return _trace([(1, 3.0, 0.1)])

    assert len(collect_traces(trace, 3, repeat_threshold=10.0)) == 3


def test_an_empty_trace_raises_rather_than_returning():
    """A solve with no checkpoints must reach the caller's error row."""
    try:
        collect_traces(list, 3, repeat_threshold=10.0)
    except RuntimeError as exc:
        assert "no checkpoints" in str(exc)
    else:
        raise AssertionError("an empty trace was accepted")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} passed")
