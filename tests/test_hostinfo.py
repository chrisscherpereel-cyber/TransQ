"""The memory guard.

A Whisper model that does not fit does not raise — the container is SIGKILLed
partway through, with no error to show anyone. The only place to catch that is
*before* the weights load, which makes this guard the difference between "use
base on this server" and a progress bar that stops at 80% forever.

Two ways to get this wrong, and the tests pin both. Too generous and it fails to
prevent the crash it exists for. Too strict and it blocks `small` on a 1 GB host
— a combination people run successfully every day — which would be a worse bug
than the one being fixed.
"""

from __future__ import annotations

import pytest

from src import hostinfo


@pytest.fixture
def host(monkeypatch):
    """Pretend the server has a given amount of memory."""

    def set_gb(value):
        monkeypatch.setattr(hostinfo, "available_memory_gb", lambda: value)

    return set_gb


# --------------------------------------------------------------------------- #
# Detecting the ceiling
# --------------------------------------------------------------------------- #


def test_a_cgroup_limit_beats_the_hosts_physical_memory(monkeypatch):
    """The failure this ordering prevents: a container on a 256 GB machine
    reporting 256 GB while actually being capped at one."""
    monkeypatch.setattr(hostinfo, "cgroup_limit_gb", lambda: 1.0)
    monkeypatch.setattr(hostinfo, "physical_memory_gb", lambda: 256.0)
    assert hostinfo.available_memory_gb() == 1.0


def test_physical_memory_is_used_when_there_is_no_cgroup(monkeypatch):
    monkeypatch.setattr(hostinfo, "cgroup_limit_gb", lambda: None)
    monkeypatch.setattr(hostinfo, "physical_memory_gb", lambda: 16.0)
    assert hostinfo.available_memory_gb() == 16.0


def test_an_unlimited_cgroup_is_not_mistaken_for_a_ceiling(tmp_path, monkeypatch):
    """cgroup v2 writes the literal string "max"; v1 writes a huge sentinel."""
    for content in ("max", str(2 ** 63 - 1)):
        path = tmp_path / "memory.max"
        path.write_text(content)
        monkeypatch.setattr(hostinfo, "CGROUP_V2", str(path))
        monkeypatch.setattr(hostinfo, "CGROUP_V1", str(tmp_path / "absent"))
        assert hostinfo.cgroup_limit_gb() is None


def test_a_real_cgroup_limit_is_read(tmp_path, monkeypatch):
    path = tmp_path / "memory.max"
    path.write_text(str(int(2 * hostinfo.GB)))
    monkeypatch.setattr(hostinfo, "CGROUP_V2", str(path))
    assert hostinfo.cgroup_limit_gb() == pytest.approx(2.0)


def test_unreadable_or_nonsense_cgroup_files_are_ignored(tmp_path, monkeypatch):
    path = tmp_path / "memory.max"
    path.write_text("not a number")
    monkeypatch.setattr(hostinfo, "CGROUP_V2", str(path))
    monkeypatch.setattr(hostinfo, "CGROUP_V1", str(tmp_path / "absent"))
    assert hostinfo.cgroup_limit_gb() is None


# --------------------------------------------------------------------------- #
# The verdicts
# --------------------------------------------------------------------------- #


def test_a_1gb_host_refuses_medium_and_large(host):
    host(1.0)
    for model in ("medium", "large-v3"):
        verdict = hostinfo.check_model_fits(model)
        assert verdict.level == "refused"
        assert not verdict.allowed


def test_a_1gb_host_allows_small_but_says_it_is_tight(host):
    """The line this guard must not cross. `small` on Community Cloud works
    often enough that refusing it would break a supported setup."""
    host(1.0)
    verdict = hostinfo.check_model_fits("small")
    assert verdict.level == "tight"
    assert verdict.allowed


def test_a_roomy_host_says_nothing_at_all(host):
    host(16.0)
    for model in hostinfo.MODEL_MEMORY_GB:
        verdict = hostinfo.check_model_fits(model)
        assert verdict.level == "ok"
        assert verdict.message == ""


def test_a_refusal_names_a_model_that_would_work(host):
    host(1.0)
    verdict = hostinfo.check_model_fits("large-v3")
    assert verdict.suggestion in hostinfo.MODEL_MEMORY_GB
    assert verdict.suggestion in verdict.message
    assert "large-v3" in verdict.message


def test_a_refusal_explains_why_there_will_be_no_error_message(host):
    """Without this, the advice reads as arbitrary — the whole point is that the
    alternative to refusing is a silent death, not a caught exception."""
    host(1.0)
    message = hostinfo.check_model_fits("medium").message
    assert "killed" in message or "terminated" in message


def test_a_tiny_host_refuses_everything_and_suggests_importing(host):
    host(0.3)
    verdict = hostinfo.check_model_fits("small")
    assert verdict.level == "refused"
    assert "import" in verdict.message.lower()


def test_an_unknown_memory_limit_permits_rather_than_blocks(host):
    """A guard that cannot read the limit must never be the reason a working
    deployment stops working."""
    host(None)
    verdict = hostinfo.check_model_fits("large-v3")
    assert verdict.allowed and verdict.level == "ok"


def test_wider_compute_types_need_more_memory(host):
    host(2.5)
    assert hostinfo.check_model_fits("medium", "int8").level != "refused"
    assert hostinfo.check_model_fits("medium", "float32").level == "refused"


def test_an_unknown_model_name_falls_back_to_a_middling_estimate():
    assert hostinfo.model_memory_gb("some-future-model") == hostinfo.MODEL_MEMORY_GB[
        "small"
    ]


# --------------------------------------------------------------------------- #
# Picking a model
# --------------------------------------------------------------------------- #


def test_largest_model_that_fits_grows_with_the_host():
    assert hostinfo.largest_model_that_fits(0.2) is None
    assert hostinfo.largest_model_that_fits(1.0) == "base"
    assert hostinfo.largest_model_that_fits(2.7) == "medium"
    assert hostinfo.largest_model_that_fits(16.0) == "large-v3"


def test_the_suggestion_is_never_the_model_that_was_refused(host):
    for budget in (0.5, 1.0, 2.0, 3.0):
        host(budget)
        for model in hostinfo.MODEL_MEMORY_GB:
            verdict = hostinfo.check_model_fits(model)
            assert verdict.suggestion != model


def test_describe_host_is_a_readable_one_liner(host):
    host(1.0)
    assert "1.0 GB" in hostinfo.describe_host()
    host(None)
    assert "unknown" in hostinfo.describe_host().lower()
