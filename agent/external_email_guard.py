"""Opt-in check that an external-email completion claim matches a real tool receipt.

Enabled with ``external_email.enforce_completion_claims``. While it is on, the
turn holds its stream so a claim is not delivered before this check runs.

A claim is tied to tool results from the current turn only (after the latest
user message). The only success is a structured receipt::

    {"hermes_external_email": {
        "version": 1,
        "outcome": "accepted",
        "acceptance": "provider",
        "recipient": "person@example.com",
        "transport": "smtp"
    }}

``acceptance`` is provider acceptance (the sender's provider took the message).
It is not inbox delivery. ``outcome`` ``failed`` or ``ambiguous``, a non-zero
exit, a timeout, a missing receipt, or a receipt whose command does not match
``external_email.receipt_command_prefixes`` is not reported as sent and is not
retried. This function only rewrites the reply. It never calls a tool.

Limits, on purpose:

* Claim wording is a small phrase list, not a proof. A paraphrase that does not
  look like those phrases is left unchanged.
* With no ``receipt_command_prefixes``, a printed receipt is not verification.
  Arbitrary command output cannot be trusted to mean a send happened, so the
  check fails closed until a prefix identifies the sender.
* Prose such as "message sent" is not a receipt. A command that prints the
  JSON but does not match a configured prefix is not verification either.
* Evidence from an earlier turn does not verify a new claim.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

RECEIPT_KEY = "hermes_external_email"
GUARD_MARKER = "[external-email status]"

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_ENV_ASSIGN = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)+")
_AFFIRM = (
    re.compile(r"\b(sent|e-?mailed)\b.{0,48}\b(e-?mails?|mail)\b", re.IGNORECASE),
    re.compile(r"\b(e-?mails?|mail)\b.{0,48}\b(has been sent|was sent|is sent)\b", re.IGNORECASE),
    re.compile(
        r"\b(i['’]ve|i have)\s+sent\b.{0,48}\b(e-?mails?|mail|message)\b", re.IGNORECASE,
    ),
    re.compile(r"\b(e-?mails?|mail)\b.{0,32}\b(is on its way|went out)\b", re.IGNORECASE),
)
_INBOX = re.compile(r"\b(inbox|delivered to|they received|delivery confirmed)\b", re.IGNORECASE)
_NEG = re.compile(
    r"\b(not sent|wasn['’]?t sent|was not sent|did not send|didn['’]?t send|"
    r"unable to send|failed to send|not been sent|not reported as sent|"
    r"nothing was sent|it was not sent|no external-email send)\b",
    re.IGNORECASE,
)
_REASONS = {
    "absent": "No external-email send was invoked in this turn. Nothing was sent.",
    "failed": "The external-email send failed. It was not sent.",
    "ambiguous": "The external-email outcome is ambiguous. It is not reported as sent.",
    "mismatch": (
        "No provider-acceptance receipt matches the claimed recipient. "
        "It is not reported as sent."
    ),
    "unverified": (
        "No verifiable provider-acceptance receipt was found. It is not reported as sent."
    ),
}


@dataclass(frozen=True)
class Receipt:
    outcome: str
    acceptance: str
    recipients: tuple[str, ...]


@dataclass(frozen=True)
class EmailEvent:
    classification: str
    recipients: tuple[str, ...]


@dataclass(frozen=True)
class Claim:
    inbox: bool
    recipients: frozenset[str]


@dataclass(frozen=True)
class Decision:
    allow: bool
    reason: str
    recipients: frozenset[str]


def completion_guard_enabled() -> bool:
    """True only when this profile's config opts in. Unreadable config stays off."""
    try:
        from hermes_cli.config import load_config

        section = load_config().get("external_email") or {}
    except Exception:
        logger.exception("external email completion guard could not read config")
        return False
    return isinstance(section, dict) and section.get("enforce_completion_claims") is True


def configured_prefixes() -> tuple[str, ...]:
    try:
        from hermes_cli.config import load_config

        section = load_config().get("external_email") or {}
        raw = section.get("receipt_command_prefixes") if isinstance(section, dict) else ()
    except Exception:
        logger.exception("external email completion guard could not read prefixes")
        return ()
    if not isinstance(raw, list):
        return ()
    return tuple(item.strip() for item in raw if isinstance(item, str) and item.strip())


def guard_reply(
    text: str,
    messages: list | None,
    *,
    enabled: bool | None = None,
    prefixes: tuple[str, ...] | None = None,
) -> str:
    """Return ``text``, or a rewrite that does not report an unverified send."""
    if enabled is None:
        enabled = completion_guard_enabled()
    if not enabled or not isinstance(text, str) or GUARD_MARKER in text:
        return text
    claim = find_claim(text)
    if claim is None:
        return text
    if prefixes is None:
        prefixes = configured_prefixes()
    decision = decide(claim, collect_events(messages or [], prefixes))
    return apply_decision(text, claim, decision)


def find_claim(text: str) -> Claim | None:
    sentences = _sentences(text)
    send = [s for s in sentences if _affirmative_send(s)]
    inbox = [s for s in sentences if _INBOX.search(s) and not _NEG.search(s)]
    if not send and not inbox:
        return None
    blob = " ".join(send + inbox)
    recipients = frozenset(_normalize(match) for match in _EMAIL.findall(blob))
    return Claim(inbox=bool(inbox), recipients=recipients)


def collect_events(messages: list, prefixes: tuple[str, ...]) -> list[EmailEvent]:
    calls = _tool_calls(_turn_messages(messages))
    events: list[EmailEvent] = []
    for message in _turn_messages(messages):
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        name, args = calls.get(
            str(message.get("tool_call_id") or ""),
            (str(message.get("name") or message.get("tool_name") or ""), {}),
        )
        content = _content_text(message.get("content"))
        receipts = extract_receipts(content)
        command = _command_text(name, args)
        prefix_ok, prefix_hit = _prefix_status(name, command, prefixes)
        classification, recipients = _classify(
            _tool_outcome(content), receipts, prefix_ok, prefix_hit,
        )
        if classification != "ignore":
            events.append(EmailEvent(classification, recipients))
    return events


def decide(claim: Claim, events: list[EmailEvent]) -> Decision:
    accepted = {addr for event in events if event.classification == "accepted" for addr in event.recipients}
    if not events:
        return Decision(False, "absent", frozenset())
    if claim.recipients and not claim.recipients <= accepted:
        if not accepted:
            return Decision(False, _worst(events), frozenset())
        return Decision(False, "mismatch", frozenset(accepted))
    if not claim.recipients:
        if any(event.classification != "accepted" for event in events) or not accepted:
            return Decision(False, _worst(events), frozenset(accepted))
        return Decision(True, "accepted", frozenset(accepted))
    if _same_recipient_unresolved(events, claim.recipients):
        return Decision(False, _worst(events), frozenset(accepted))
    return Decision(True, "accepted", frozenset(claim.recipients))


def apply_decision(text: str, claim: Claim, decision: Decision) -> str:
    if decision.allow and not claim.inbox:
        return text
    if decision.allow:
        return _downgrade_inbox(text, decision.recipients)
    kept = [sentence for sentence in _sentences(text) if not _blocked_sentence(sentence)]
    status = _status(decision)
    body = " ".join(kept).strip()
    return f"{body}\n\n{status}" if body else status


def extract_receipts(text: str) -> list[Receipt]:
    found: list[Receipt] = []
    _walk_receipts(text, found, 0)
    return found


def _turn_messages(messages: list) -> list:
    last_user = -1
    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            last_user = index
    return list(messages[last_user + 1:])


def _tool_calls(messages: list) -> dict[str, tuple[str, dict]]:
    calls: dict[str, tuple[str, dict]] = {}
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = str(function.get("name") or call.get("name") or "")
            calls[str(call.get("id") or "")] = (name, _parse_args(function.get("arguments", call.get("arguments"))))
    return calls


def _parse_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _command_text(name: str, args: dict) -> str:
    if name == "terminal":
        return str(args.get("command") or "")
    if name == "execute_code":
        return str(args.get("code") or "")
    return ""


def _prefix_status(name: str, command: str, prefixes: tuple[str, ...]) -> tuple[bool, bool]:
    # No configured sender: a tool can print a receipt without sending. Fail closed.
    if not prefixes:
        return False, False
    hit = _prefix_hit(name, command, prefixes)
    return hit, hit


def _prefix_hit(name: str, command: str, prefixes: tuple[str, ...]) -> bool:
    text = command.strip()
    if name == "terminal":
        text = _ENV_ASSIGN.sub("", text).strip()
        return any(text.startswith(prefix) for prefix in prefixes)
    if name == "execute_code":
        if any(text.startswith(prefix) for prefix in prefixes):
            return True
        return any(f'"{prefix}"' in command or f"'{prefix}'" in command for prefix in prefixes)
    return False


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def _tool_outcome(content: str) -> str:
    data = _first_object(content)
    if not isinstance(data, dict):
        return "ambiguous"
    status = str(data.get("status") or "")
    code = data.get("exit_code")
    if status == "timeout" or code == 124:
        return "timeout"
    if status == "interrupted":
        return "interrupted"
    if status in {"error", "degraded", "blocked"}:
        return "failed"
    if isinstance(code, int) and code != 0:
        return "failed"
    if status == "yielded_to_background":
        return "ambiguous"
    if status == "success" or code == 0:
        return "success"
    if code is None and not status:
        return "ambiguous"
    return "ambiguous"


def _classify(
    outcome: str, receipts: list[Receipt], prefix_ok: bool, prefix_hit: bool,
) -> tuple[str, tuple[str, ...]]:
    if not receipts and not prefix_hit:
        return "ignore", ()
    recipients = tuple(dict.fromkeys(addr for receipt in receipts for addr in receipt.recipients))
    if any(receipt.outcome == "ambiguous" for receipt in receipts) or outcome in {
        "timeout", "interrupted", "ambiguous",
    }:
        return "ambiguous", recipients
    if outcome == "failed" or any(receipt.outcome == "failed" for receipt in receipts):
        return "failed", recipients
    if not prefix_ok:
        return "unverified", recipients
    accepted = [
        receipt for receipt in receipts
        if receipt.outcome == "accepted" and receipt.acceptance == "provider" and receipt.recipients
    ]
    if accepted:
        verified = tuple(dict.fromkeys(addr for receipt in accepted for addr in receipt.recipients))
        return "accepted", verified
    return "unverified", recipients


def _worst(events: list[EmailEvent]) -> str:
    kinds = {event.classification for event in events}
    for kind in ("ambiguous", "failed", "unverified"):
        if kind in kinds:
            return kind
    return "unverified"


def _same_recipient_unresolved(events: list[EmailEvent], claimed: frozenset[str]) -> bool:
    for event in events:
        if event.classification == "accepted":
            continue
        if not event.recipients or set(event.recipients) & claimed:
            return True
    return False


def _walk_receipts(text: str, found: list[Receipt], depth: int) -> None:
    if depth > 4 or not text or RECEIPT_KEY not in text:
        return
    data = _first_object(text)
    if isinstance(data, dict):
        _take_receipt(data, found)
        for value in data.values():
            if isinstance(value, str):
                _walk_receipts(value, found, depth + 1)
            elif isinstance(value, dict):
                _take_receipt(value, found)
        return
    decoder = json.JSONDecoder()
    start = 0
    while True:
        pos = text.find(RECEIPT_KEY, start)
        if pos < 0:
            return
        brace = text.rfind("{", 0, pos)
        if brace < 0:
            start = pos + len(RECEIPT_KEY)
            continue
        try:
            obj, end = decoder.raw_decode(text[brace:])
        except json.JSONDecodeError:
            start = pos + len(RECEIPT_KEY)
            continue
        if isinstance(obj, dict):
            _take_receipt(obj, found)
        start = brace + max(end, 1)


def _take_receipt(obj: dict, found: list[Receipt]) -> None:
    if RECEIPT_KEY not in obj:
        return
    parsed = _parse_receipt(obj[RECEIPT_KEY])
    found.append(parsed if parsed is not None else Receipt("ambiguous", "none", ()))


def _parse_receipt(value: Any) -> Receipt | None:
    if not isinstance(value, dict) or value.get("version") != 1:
        return None
    outcome = value.get("outcome")
    if outcome not in {"accepted", "failed", "ambiguous"}:
        return None
    acceptance = value.get("acceptance") or "none"
    if not isinstance(acceptance, str):
        return None
    raw = value.get("recipient", value.get("recipients", ()))
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return None
    recipients = []
    for item in raw:
        addr = _normalize(str(item))
        if addr and addr not in recipients:
            recipients.append(addr)
    return Receipt(outcome, acceptance, tuple(recipients))


def _first_object(text: str) -> dict | None:
    stripped = text.strip()
    start = stripped.find("{")
    if start < 0:
        return None
    try:
        obj, _end = json.JSONDecoder().raw_decode(stripped[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _affirmative_send(sentence: str) -> bool:
    if _NEG.search(sentence):
        return False
    return any(pattern.search(sentence) for pattern in _AFFIRM)


def _blocked_sentence(sentence: str) -> bool:
    if _NEG.search(sentence):
        return False
    return _affirmative_send(sentence) or bool(_INBOX.search(sentence))


def _downgrade_inbox(text: str, recipients: frozenset[str]) -> str:
    kept = [
        sentence for sentence in _sentences(text)
        if not (_INBOX.search(sentence) and not _NEG.search(sentence))
    ]
    who = ", ".join(sorted(recipients)) if recipients else "the addressed recipient"
    note = (
        f"{GUARD_MARKER} The provider accepted the message for {who}. "
        "Inbox delivery was not verified."
    )
    body = " ".join(kept).strip()
    return f"{body}\n\n{note}" if body else note


def _status(decision: Decision) -> str:
    sentence = _REASONS.get(decision.reason, _REASONS["unverified"])
    return (
        f"{GUARD_MARKER} {sentence} "
        "Provider acceptance is not inbox delivery. This was not retried."
    )


def _normalize(value: str) -> str:
    text = value.strip().lower()
    if "@" not in text or "." not in text.split("@", 1)[-1]:
        return ""
    return text
