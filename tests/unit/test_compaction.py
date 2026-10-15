"""Unit tests for harness.compaction — the checkpoint function and the surgery gate (no models)."""

from harness.compaction import (
    Candidate,
    Checkpoint,
    GateResult,
    Message,
    Probe,
    Settings,
    SummaryRequest,
    Trigger,
    ceiling,
    compact,
    due,
    offer_surgery,
    split,
    uses_template,
)


def _turns(n, size=40):
    return [Message("user" if i % 2 == 0 else "assistant", f"turn {i} " + "x" * size) for i in range(n)]


class FakeSummarizer:
    """Replies with the template; `bad` counts replies that ignore it (to exercise the nudge)."""

    def __init__(self, bad=0):
        self.bad = bad
        self.requests: list[SummaryRequest] = []

    def __call__(self, request: SummaryRequest) -> str:
        self.requests.append(request)
        if self.bad > 0:
            self.bad -= 1
            return "a summary with no headings"
        return "## Goal\nkeep the plot\n## Facts\nx\n## Decided\nx\n## Open\nx\n## State\nx"


def test_ceiling_respects_buffer_and_defaults_to_ten_percent():
    assert ceiling(1000) == 900
    assert ceiling(1000, buffer=100) == 900
    assert ceiling(1000, buffer=250) == 750


def test_due_fires_only_below_the_ceiling_and_with_auto_on():
    small = [Message("user", "hi")]
    assert due(small, 1000, Settings(auto=True)) is False
    big = [Message("user", "x" * 40_000)]
    assert due(big, 10_000, Settings(auto=True)) is True
    assert due(big, 10_000, Settings(auto=False)) is False


def test_split_keeps_the_newest_messages_within_keep():
    messages = [Message("user", "a" * 40), Message("assistant", "b" * 40), Message("user", "c" * 40)]
    older, recent = split(messages, keep=20)          # ~10 tokens each at 4 chars/token
    assert [m.text[:1] for m in recent] == ["b", "c"]
    assert [m.text[:1] for m in older] == ["a"]
    older, recent = split(messages, keep=10)
    assert [m.text[:1] for m in recent] == ["c"]
    assert [m.text[:1] for m in older] == ["a", "b"]
    assert split(messages, keep=10_000) is None       # everything fits: nothing to compact
    assert split([Message("user", "one")], keep=1) is None


def test_checkpoint_round_trip():
    checkpoint = Checkpoint(summary="## Goal\nkeep it", recent="user: hi\n\nassistant: hello")
    parsed = Checkpoint.parse(checkpoint.render())
    assert parsed is not None
    assert parsed.summary == checkpoint.summary
    assert parsed.recent == checkpoint.recent
    assert Checkpoint.parse("no checkpoint here") is None


def test_compact_skips_when_auto_is_off_and_the_trigger_is_not_manual():
    outcome = compact(_turns(4), Trigger.auto(), Settings(auto=False), FakeSummarizer())
    assert outcome.status == "skipped"


def test_compact_fails_cleanly_when_nothing_can_be_compacted():
    outcome = compact([Message("user", "only")], Trigger.manual(), Settings(), FakeSummarizer())
    assert outcome.status == "failed"


def test_compact_completes_with_one_nudge_after_a_template_miss():
    summarizer = FakeSummarizer(bad=1)
    outcome = compact(_turns(6), Trigger.manual(), Settings(keep=40), summarizer)
    assert outcome.status == "completed"
    assert outcome.checkpoint is not None and uses_template(outcome.checkpoint.summary)
    assert [r.retry for r in summarizer.requests] == [False, True]


def test_compact_fails_when_the_template_is_ignored_twice():
    summarizer = FakeSummarizer(bad=2)
    outcome = compact(_turns(6), Trigger.manual(), Settings(keep=40), summarizer)
    assert outcome.status == "failed"
    assert len(summarizer.requests) == 2


def test_compact_emits_events_around_the_checkpoint():
    seen: list[str] = []
    outcome = compact(_turns(6), Trigger.manual(), Settings(keep=40), FakeSummarizer(),
                      emit=lambda kind, payload: seen.append(kind))
    assert outcome.status == "completed"
    assert seen == ["compaction.started", "compaction.completed"]


def _candidate(name, probe_ok=True, gate_passed=True, score=0.9, baseline=0.8):
    return Candidate(
        name=name,
        probe=lambda: Probe(ok=probe_ok, model=f"{name}-model", detail="health"),
        gate=lambda checkpoint, engine: GateResult(passed=gate_passed, score=score, baseline=baseline,
                                                   detail="canary"),
    )


def test_surgery_commits_only_when_the_gate_passes():
    switched: list[str] = []
    outcome = offer_surgery("s1", Checkpoint("## Goal\nx", "recent"), current="old",
                            candidates=[_candidate("new", gate_passed=False)],
                            apply=switched.append)
    assert outcome.status == "declined" and switched == []

    outcome = offer_surgery("s1", Checkpoint("## Goal\nx", "recent"), current="old",
                            candidates=[_candidate("new", gate_passed=True)], apply=switched.append)
    assert outcome.status == "committed" and switched == ["new"]
    assert outcome.engine == "new"


def test_surgery_skips_a_failed_probe_and_the_current_engine():
    switched: list[str] = []
    outcome = offer_surgery("s1", Checkpoint("## Goal\nx", "recent"), current="old",
                            candidates=[_candidate("old"), _candidate("new", probe_ok=False)],
                            apply=switched.append)
    assert outcome.status == "declined" and switched == []
    assert [name for name, _, _ in outcome.results] == ["new"]


def test_surgery_respects_an_absolute_score_floor():
    switched: list[str] = []
    outcome = offer_surgery("s1", Checkpoint("## Goal\nx", "recent"), current="old",
                            candidates=[_candidate("new", gate_passed=True, score=0.5)],
                            apply=switched.append, min_score=0.7)
    assert outcome.status == "declined" and switched == []


def test_surgery_reports_unavailable_without_candidates():
    outcome = offer_surgery("s1", Checkpoint("## Goal\nx", "recent"), current="old", candidates=[],
                            apply=lambda name: None)
    assert outcome.status == "unavailable"
