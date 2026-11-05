"""Week 1 starter: support-ticket router built on the Claude API.

Patterns you are practicing (these map to the exam's domains):
  * Prompt chaining      classify -> route -> draft reply
  * Routing              the category decides which prompt (or a human) handles it
  * Structured output    forced tool use returns JSON that matches a schema
  * Validate and retry   never trust the model's JSON; check it and feed errors back

Run:  python ticket_router.py            (needs ANTHROPIC_API_KEY)
"""
import json
import os
import sys
from pathlib import Path

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_RETRIES = 2
CONFIDENCE_FLOOR = 0.6

CATEGORIES = ["billing", "technical_issue", "account_access",
              "refund_request", "feature_request", "cancellation", "other"]
URGENCIES = ["low", "medium", "high", "critical"]
SENTIMENTS = ["positive", "neutral", "frustrated", "angry"]


# ---------------------------------------------------------------------------
# Step 1: classification with structured output (forced tool use)
# ---------------------------------------------------------------------------

CLASSIFY_TOOL = {
    "name": "classify_ticket",
    "description": "Record the classification of one customer support ticket.",
    "input_schema": {
        "type": "object",
        "properties": {
            # reasoning comes first so the model thinks before it labels
            "reasoning": {"type": "string",
                          "description": "Two sentences max: the evidence for your labels."},
            "category": {"type": "string", "enum": CATEGORIES},
            "urgency": {"type": "string", "enum": URGENCIES},
            "sentiment": {"type": "string", "enum": SENTIMENTS},
            "confidence": {"type": "number",
                           "description": "0.0 to 1.0. Lower it when the ticket is ambiguous."},
            "summary": {"type": "string", "description": "One sentence, 25 words max."},
        },
        "required": ["reasoning", "category", "urgency", "sentiment",
                     "confidence", "summary"],
    },
}

# Explicit criteria + few-shot examples beat vague instructions.
CLASSIFY_SYSTEM = """You triage customer support tickets for a B2B software company.
Always answer by calling the classify_ticket tool.

Urgency criteria:
- critical: outage, data loss, security issue, or the customer cannot operate their business right now
- high: a major feature is broken or a deadline/renewal is at risk, but a workaround may exist
- medium: a real problem or request with a normal timeline
- low: questions, suggestions, or anything with no time pressure

Category notes:
- Use cancellation for any ticket about not renewing or leaving, even if it also mentions cost.
- Use refund_request only when the customer explicitly wants money back.
- Use other only when nothing else fits.

Examples:
Ticket: "Dashboard is blank for all 50 of our users since 9am, we're live with customers."
-> category=technical_issue, urgency=critical, sentiment=frustrated

Ticket: "Could you add dark mode someday? No rush."
-> category=feature_request, urgency=low, sentiment=positive
"""


class ClassificationError(Exception):
    """Raised when the model cannot produce a valid classification after retries."""


def validate_classification(data: dict) -> list[str]:
    """Return a list of problems. An empty list means the data is valid."""
    errors = []
    if not isinstance(data, dict):
        return ["Output must be an object."]
    for field, allowed in (("category", CATEGORIES), ("urgency", URGENCIES),
                           ("sentiment", SENTIMENTS)):
        if data.get(field) not in allowed:
            errors.append(f"{field} must be one of {allowed}, got {data.get(field)!r}.")
    conf = data.get("confidence")
    if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not 0 <= conf <= 1:
        errors.append(f"confidence must be a number from 0 to 1, got {conf!r}.")
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        errors.append("summary must be a non-empty string.")
    elif len(summary.split()) > 25:
        errors.append("summary must be 25 words or fewer.")
    if not isinstance(data.get("reasoning"), str) or not data.get("reasoning", "").strip():
        errors.append("reasoning must be a non-empty string.")
    return errors


def classify_ticket(client, ticket_text: str) -> dict:
    """Classify a ticket, retrying with validation feedback if the output is bad."""
    messages = [{"role": "user", "content": f"Ticket:\n{ticket_text}"}]
    last_errors: list[str] = []

    for attempt in range(MAX_RETRIES + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=600,
            system=CLASSIFY_SYSTEM,
            tools=[CLASSIFY_TOOL],
            tool_choice={"type": "tool", "name": "classify_ticket"},
            messages=messages,
        )
        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use is None:
            raise ClassificationError("Model did not call the classify_ticket tool.")

        last_errors = validate_classification(tool_use.input)
        if not last_errors:
            return dict(tool_use.input)

        # Feed the errors back as a tool_result so the model can correct itself.
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "is_error": True,
            "content": "Validation failed: " + " ".join(last_errors)
                       + " Call classify_ticket again with corrected values.",
        }]})

    raise ClassificationError(f"Invalid after {MAX_RETRIES + 1} attempts: {last_errors}")


# ---------------------------------------------------------------------------
# Step 2: routing. The category chooses the drafting prompt, or a human.
# ---------------------------------------------------------------------------

BASE_REPLY_RULES = ("Write a short, warm, professional reply (under 120 words). "
                    "Do not invent account details, prices, or policies. "
                    "If you need information, ask for it. Sign off as 'Customer Success Team'.")

ROUTE_PROMPTS = {
    "billing": "You are a billing support specialist. Acknowledge the issue, say you are "
               "reviewing the invoice, and ask for any missing invoice numbers.",
    "technical_issue": "You are a technical support engineer. Acknowledge impact, ask for "
                       "timestamps, error messages, and affected users, and set expectations.",
    "account_access": "You are an account support specialist. Never ask for passwords. "
                      "Explain you will verify identity before restoring access.",
    "refund_request": "You are a support specialist. Acknowledge the request and say it is "
                      "being reviewed. Do NOT promise or deny a refund.",
    "feature_request": "You are a product-minded support specialist. Thank them, confirm you "
                       "have logged the request, and do not promise a delivery date.",
    "other": "You are a support specialist. Answer helpfully or say who will follow up.",
}


def escalation_reason(c: dict) -> str | None:
    """Return why a human should handle this ticket, or None if the AI can draft."""
    if c["category"] == "cancellation":
        return "Cancellation or churn risk: needs a human CSM."
    if c["urgency"] == "critical":
        return "Critical urgency."
    if c["sentiment"] == "angry":
        return "Angry customer."
    if c["confidence"] < CONFIDENCE_FLOOR:
        return f"Low classifier confidence ({c['confidence']:.2f})."
    return None


# ---------------------------------------------------------------------------
# Step 3: draft a reply (only when the AI is allowed to handle it)
# ---------------------------------------------------------------------------

def draft_reply(client, ticket_text: str, c: dict) -> str:
    system = f"{ROUTE_PROMPTS[c['category']]}\n\n{BASE_REPLY_RULES}"
    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=system,
        messages=[{"role": "user", "content": f"Customer ticket:\n{ticket_text}"}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def route_ticket(client, ticket: dict) -> dict:
    c = classify_ticket(client, ticket["text"])
    reason = escalation_reason(c)
    if reason:
        return {"id": ticket["id"], "classification": c, "action": "escalate_to_human",
                "reason": reason,
                "handoff_note": f"[{c['urgency'].upper()}] {c['summary']} (sentiment: {c['sentiment']})"}
    return {"id": ticket["id"], "classification": c, "action": "auto_draft",
            "reply": draft_reply(client, ticket["text"], c)}


def main() -> None:
    path = Path(__file__).with_name("sample_tickets.json")
    tickets = json.loads(path.read_text())
    import anthropic  # imported here so the offline tests run without the SDK
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    for t in tickets:
        try:
            result = route_ticket(client, t)
        except ClassificationError as e:
            result = {"id": t["id"], "action": "escalate_to_human", "reason": str(e)}
        print(json.dumps(result, indent=2))
        print("-" * 60)


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first (see README.md).")
    main()
