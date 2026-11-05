"""Offline tests: no API key or network needed. Run:  python test_offline.py

Uses a fake client to check the validate-and-retry loop and the escalation rules.
"""
from types import SimpleNamespace as NS

from ticket_router import (ClassificationError, classify_ticket,
                           escalation_reason, validate_classification)

GOOD = {"reasoning": "Outage affecting all agents.", "category": "technical_issue",
        "urgency": "critical", "sentiment": "frustrated", "confidence": 0.95,
        "summary": "Contact center login screen is blank for all agents."}


class FakeClient:
    """Returns scripted tool_use outputs, one per call, and records every request."""
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []
        self.messages = NS(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        data = self.outputs[len(self.calls) - 1]
        block = NS(type="tool_use", id=f"toolu_{len(self.calls)}", name="classify_ticket", input=data)
        return NS(content=[block])


def test_validation_catches_bad_enum_and_confidence():
    bad = {**GOOD, "category": "billingg", "confidence": 1.7}
    errors = validate_classification(bad)
    assert len(errors) == 2, errors


def test_valid_first_try_makes_one_call():
    client = FakeClient([GOOD])
    assert classify_ticket(client, "x")["category"] == "technical_issue"
    assert len(client.calls) == 1


def test_retry_feeds_error_back_then_succeeds():
    client = FakeClient([{**GOOD, "urgency": "super-urgent"}, GOOD])
    result = classify_ticket(client, "x")
    assert result["urgency"] == "critical"
    assert len(client.calls) == 2
    feedback = client.calls[1]["messages"][-1]["content"][0]
    assert feedback["type"] == "tool_result" and feedback["is_error"] is True
    assert "urgency" in feedback["content"]


def test_gives_up_after_max_retries():
    client = FakeClient([{**GOOD, "category": "nope"}] * 3)
    try:
        classify_ticket(client, "x")
        raise AssertionError("expected ClassificationError")
    except ClassificationError:
        assert len(client.calls) == 3


def test_escalation_rules():
    assert escalation_reason({**GOOD}) == "Critical urgency."
    assert escalation_reason({**GOOD, "urgency": "low", "category": "cancellation"})
    assert escalation_reason({**GOOD, "urgency": "low", "sentiment": "angry"}) == "Angry customer."
    assert escalation_reason({**GOOD, "urgency": "low", "sentiment": "neutral", "confidence": 0.4})
    assert escalation_reason({**GOOD, "urgency": "low", "sentiment": "neutral"}) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
