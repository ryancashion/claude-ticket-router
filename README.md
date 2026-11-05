# Week 1: Support Ticket Router (Claude API)

A small workflow that classifies a support ticket, decides who handles it, and drafts a reply.

    ticket -> classify (structured output) -> route -> draft reply  OR  escalate to a human

## Setup

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=your-key

Get a key at console.anthropic.com. Set a spend limit before you start.

## Run it

    python3 test_offline.py     # no API key needed; checks validation, retry, escalation rules
    python3 ticket_router.py    # routes the 8 sample tickets
    python3 eval.py             # scores category and urgency accuracy

Change models with `CLAUDE_MODEL` (default `claude-sonnet-5-5`).

## What each part demonstrates

| Concept | Where |
|---|---|
| Forced tool use for JSON output | `CLASSIFY_TOOL`, `tool_choice` |
| Explicit criteria and few-shot examples | `CLASSIFY_SYSTEM` |
| Validate, then retry with error feedback | `validate_classification`, `classify_ticket` |
| Routing by category | `ROUTE_PROMPTS` |
| Human escalation rules | `escalation_reason` |
| Measuring quality | `eval.py` |

## Results

_Add your eval numbers here after each improvement: change made, accuracy before, accuracy after._

## Notes

- Schema enums guide the model but do not guarantee valid output, which is why outputs are validated.
- Replies are drafts for human review. Never auto-send replies about money or access.
