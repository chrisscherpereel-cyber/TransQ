"""A cut-off reply must not throw away the questions it already finished.

The report: every chunk logged "reply cut off, retrying smaller", the run ended
with the truncation advice, and **no questions came back** — despite the model
having written some before each cut. Those items were complete, valid, and
already billed. Discarding them turned a partial success into a total loss and
charged for it twice, since the retry re-sends the whole transcript chunk.

The rule these tests encode: **only a reply that produced nothing is a
failure.** Anything the model finished before running out of room is kept, the
shortfall is what gets retried, and a second cut-off salvages again rather than
discarding both attempts.
"""

from __future__ import annotations

import json

import pytest

from src.diagnostics import PHASE_GENERATE, RunReport
from src.llm import TruncatedResponseError, salvage_array_objects
from src.mcq import generate_questions
from src.schema import Chunk


def question(n: int) -> dict:
    return {
        "stem": f"Which claim about planning topic {_WORDS[n % len(_WORDS)]} holds?",
        "options": ["Alpha", "Beta", "Gamma", "Delta"],
        "correct_index": n % 4,
        "rationale": "Because alpha.",
        "distractor_rationales": ["", "b", "c", "d"],
        "bloom": "Understand",
        "difficulty": "Medium",
        "topic": f"topic {_WORDS[n % len(_WORDS)]}",
        "source_timestamp": "5:00",
        "source_quote": "the lecture said so",
    }


_WORDS = [
    "capacity", "inventory", "forecasting", "scheduling", "bottlenecks",
    "outsourcing", "quality", "logistics", "staffing", "procurement",
]


def cut_off_after(count: int, total: int = 10) -> str:
    """A reply with ``count`` complete questions and one cut mid-write."""
    body = ",".join(json.dumps(question(i)) for i in range(count))
    partial = json.dumps(question(count))[: len(json.dumps(question(count))) // 2]
    return '{"questions": [' + body + ("," + partial if count < total else "")


@pytest.fixture
def chunk() -> list[Chunk]:
    return [Chunk(index=0, start=0.0, end=600.0, text="Operations content. " * 60)]


class TruncatingLLM:
    """Cuts off the first ``fail_times`` replies after writing some questions."""

    def __init__(self, fail_times: int = 1, complete_before_cut: int = 6):
        self.fail_times = fail_times
        self.complete_before_cut = complete_before_cut
        self.calls = 0
        self.budgets: list[int] = []
        self.last_asked = 0

    def complete_json(self, system, user, max_tokens=None):
        self.calls += 1
        self.budgets.append(max_tokens or 0)
        asked = int(user.split("exactly ")[1].split(" ")[0])
        self.last_asked = asked
        if self.calls <= self.fail_times:
            raise TruncatedResponseError(
                "cut off", raw=cut_off_after(self.complete_before_cut)
            )
        return {"questions": [question(100 + i) for i in range(asked)]}


# --------------------------------------------------------------------------- #
# The reported loss
# --------------------------------------------------------------------------- #


def test_questions_written_before_the_cut_are_kept(chunk):
    """The whole complaint: six finished questions were thrown away."""
    client = TruncatingLLM(fail_times=1, complete_before_cut=6)
    report = RunReport()
    produced = generate_questions(client, chunk, [10], report=report)

    assert len(produced) >= 6, "the completed items must survive the cut"


def test_a_run_cut_off_every_time_still_returns_what_it_wrote(chunk):
    """The exact failure reported: cut off on the retry too, and the run ended
    with nothing. Two truncations must still leave the salvage."""
    client = TruncatingLLM(fail_times=99, complete_before_cut=5)
    report = RunReport()
    produced = generate_questions(client, chunk, [10], report=report)

    assert produced, "a doubly-truncated chunk must not come back empty"
    assert len(produced) >= 5


def test_only_a_reply_that_produced_nothing_counts_as_a_failure(chunk):
    client = TruncatingLLM(fail_times=99, complete_before_cut=0)
    report = RunReport()
    produced = generate_questions(client, chunk, [6], report=report)

    assert produced == []
    assert report.phase_failed_entirely(PHASE_GENERATE)


def test_a_salvaged_chunk_is_not_recorded_as_a_total_failure(chunk):
    """It has to be visible as degraded, not as dead — the difference decides
    whether the app tells you the run failed."""
    client = TruncatingLLM(fail_times=99, complete_before_cut=5)
    report = RunReport()
    generate_questions(client, chunk, [10], report=report)

    assert not report.phase_failed_entirely(PHASE_GENERATE)


def test_the_report_says_how_many_were_rescued(chunk):
    client = TruncatingLLM(fail_times=1, complete_before_cut=4)
    report = RunReport()
    generate_questions(client, chunk, [10], report=report)

    detail = " ".join(s.detail for s in report.phase_steps(PHASE_GENERATE))
    assert "kept 4" in detail, f"the count of rescued items should be stated: {detail}"


# --------------------------------------------------------------------------- #
# The retry asks only for what is missing
# --------------------------------------------------------------------------- #


def test_the_retry_asks_for_the_shortfall_not_the_whole_batch(chunk):
    """Re-asking for all ten after salvaging six pays for six twice."""
    client = TruncatingLLM(fail_times=1, complete_before_cut=6)
    generate_questions(client, chunk, [10])

    assert client.calls == 2
    asked_again = int(client.last_asked)
    assert asked_again <= 4, "only the shortfall should be re-requested"
    # Fewer questions, but far more room each — which is what stops the retry
    # being cut off in the same place.
    assert client.budgets[1] / asked_again > client.budgets[0] / 10


def test_no_retry_when_the_salvage_already_covers_the_request(chunk):
    """Nine of ten rescued does not justify another full round trip."""
    client = TruncatingLLM(fail_times=1, complete_before_cut=10)
    generate_questions(client, chunk, [10])

    assert client.calls == 1, "asking again for nothing wastes the input tokens"


def test_the_first_budget_has_room_for_reasoning_models(chunk):
    """Truncation on *every* chunk points at an under-sized budget, not a long
    lecture: reasoning models spend the allowance thinking before writing."""
    client = TruncatingLLM(fail_times=0)
    generate_questions(client, chunk, [3])

    assert client.budgets[0] >= 4000


# --------------------------------------------------------------------------- #
# The salvage itself
# --------------------------------------------------------------------------- #


def test_salvage_recovers_complete_items_and_stops_at_the_cut():
    items = salvage_array_objects(cut_off_after(3), "questions")
    assert len(items) == 3
    assert all(i["stem"] for i in items)


def test_salvage_handles_a_reply_that_never_reached_the_array():
    assert salvage_array_objects('{"quest', "questions") == []
    assert salvage_array_objects("", "questions") == []
    assert salvage_array_objects("not json at all", "questions") == []


def test_salvage_reads_a_complete_reply_too():
    whole = json.dumps({"questions": [question(0), question(1)]})
    assert len(salvage_array_objects(whole, "questions")) == 2


def test_salvage_survives_a_code_fence():
    fenced = "```json\n" + cut_off_after(2)
    assert len(salvage_array_objects(fenced, "questions")) == 2


def test_salvage_is_not_confused_by_braces_inside_strings():
    tricky = (
        '{"questions": [{"stem": "What does {x} mean?", "options": ["a"], '
        '"correct_index": 0, "source_quote": "he said \\"{y}\\" then", '
        '"source_timestamp": "1:00", "rationale": "r"}, {"stem": "cut'
    )
    items = salvage_array_objects(tricky, "questions")
    assert len(items) == 1
    assert "{x}" in items[0]["stem"]


def test_salvage_ignores_a_different_key():
    assert salvage_array_objects(cut_off_after(3), "reviews") == []


def test_the_error_carries_the_text_it_was_given():
    exc = TruncatedResponseError("cut", raw='{"questions": [')
    assert exc.raw == '{"questions": ['
    assert TruncatedResponseError("cut").raw == ""
