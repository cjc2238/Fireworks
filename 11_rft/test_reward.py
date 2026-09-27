"""Unit tests for the reward: always test full credit, partial credit, zero, and adversarial cases."""

from reward import score_triage

GOLD = "team=inference; sev=S2; tag=cold-start"


def test_perfect():
    assert score_triage(GOLD, GOLD) == (1.0, "all correct")


def test_partial_credit():
    s, reason = score_triage("team=inference; sev=S3; tag=cold-start", GOLD)
    assert s == 0.6 and "sev" in reason


def test_all_wrong_but_valid():
    assert score_triage("team=billing; sev=S4; tag=invoice", GOLD)[0] == 0.0


def test_malformed():
    assert score_triage("The team is inference and severity S2", GOLD)[0] == 0.0


def test_hack_extra_text():
    # model tries to append reasoning or hedge with several answers
    assert score_triage(GOLD + "\nor maybe team=billing", GOLD)[0] == 0.0


def test_hack_invalid_team():
    assert score_triage("team=everyone; sev=S2; tag=cold-start", GOLD)[0] == 0.0


def test_too_long():
    assert score_triage("x" * 500, GOLD)[0] == 0.0
