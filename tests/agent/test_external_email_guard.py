"""Opt-in external-email completion check. Transport is mocked; nothing is sent."""

from __future__ import annotations

import json

from agent.external_email_guard import GUARD_MARKER, guard_reply

PREFIX = ("/opt/mail/send",)
RECIPIENT = "a@example.com"


def _receipt(outcome="accepted", acceptance="provider", recipient=RECIPIENT):
    return json.dumps({
        "hermes_external_email": {
            "version": 1,
            "outcome": outcome,
            "acceptance": acceptance,
            "recipient": recipient,
            "transport": "smtp",
        },
    })


def _tool_result(output, *, exit_code=0, status="success"):
    return json.dumps({"output": output, "exit_code": exit_code, "status": status, "error": None})


def _turn(command, result, *, name="terminal"):
    if name == "execute_code":
        args = {"code": command}
    else:
        args = {"command": command}
    return [
        {"role": "user", "content": "send the note"},
        {"role": "assistant", "tool_calls": [{
            "id": "c1",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]},
        {"role": "tool", "tool_call_id": "c1", "name": name, "content": result},
    ]


def _guard(text, messages, prefixes=PREFIX):
    return guard_reply(text, messages, enabled=True, prefixes=prefixes)


def test_final_transform_rewrites_an_unverified_claim_when_enabled(monkeypatch):
    from types import SimpleNamespace

    from agent.turn_finalizer import apply_llm_output_transform

    monkeypatch.setattr("agent.external_email_guard.completion_guard_enabled", lambda: True)
    agent = SimpleNamespace(
        session_id="s", model="m", platform="cli", _hold_stream=False, _held_transforms=None,
    )
    text = f"I sent the email to {RECIPIENT}."
    final, transformed, pre = apply_llm_output_transform(
        agent, text, turn_id="t-email", messages=[{"role": "user", "content": "hi"}],
    )
    assert transformed is True
    assert pre == text
    assert "Nothing was sent" in final
    assert "I sent the email" not in final


def test_guard_off_leaves_a_claim_unchanged():
    text = f"I sent the email to {RECIPIENT}."
    assert guard_reply(text, [], enabled=False, prefixes=PREFIX) == text


def test_absent_invocation_is_not_reported_sent():
    out = _guard(f"I sent the email to {RECIPIENT}.", [{"role": "user", "content": "hi"}])
    assert "I sent the email" not in out
    assert "Nothing was sent" in out
    assert "not retried" in out
    assert "inbox delivery" in out.lower()


def test_send_failure_is_not_reported_sent():
    messages = _turn(
        f"/opt/mail/send {RECIPIENT}",
        _tool_result(_receipt(outcome="failed", acceptance="none"), exit_code=1, status="error"),
    )
    out = _guard(f"I sent the email to {RECIPIENT}.", messages)
    assert "failed" in out.lower()
    assert "was not sent" in out
    assert "not retried" in out
    assert "I sent the email" not in out


def test_timeout_is_ambiguous_and_not_retried():
    messages = _turn(
        f"/opt/mail/send {RECIPIENT}",
        _tool_result(_receipt(), exit_code=124, status="timeout"),
    )
    out = _guard(f"I sent the email to {RECIPIENT}.", messages)
    assert "ambiguous" in out.lower()
    assert "not reported as sent" in out
    assert "not retried" in out
    assert "I sent the email" not in out


def test_fabricated_success_without_a_matching_command_is_rejected():
    forged = _receipt()
    messages = _turn(f"printf '%s' '{forged}'", _tool_result(forged))
    out = _guard(f"I sent the email to {RECIPIENT}.", messages)
    assert "not reported as sent" in out
    assert "not retried" in out
    assert "I sent the email" not in out


def test_receipt_pasted_into_the_reply_is_not_a_tool_event():
    text = f"I sent the email to {RECIPIENT}. {_receipt()}"
    out = _guard(text, [{"role": "user", "content": "hi"}])
    assert "Nothing was sent" in out
    assert GUARD_MARKER in out


def test_verified_matching_provider_acceptance_keeps_the_claim():
    text = f"I sent the email to {RECIPIENT}."
    messages = _turn(f"/opt/mail/send {RECIPIENT}", _tool_result(_receipt()))
    assert _guard(text, messages) == text


def test_verified_execute_code_sender_keeps_the_claim():
    text = f"The email to {RECIPIENT} has been sent."
    code = 'open("/opt/mail/send")\n'
    messages = _turn(code, _tool_result(_receipt()), name="execute_code")
    assert _guard(text, messages) == text


def test_inbox_wording_is_downgraded_to_provider_acceptance():
    text = f"I sent the email to {RECIPIENT}. It was delivered to their inbox."
    messages = _turn(f"/opt/mail/send {RECIPIENT}", _tool_result(_receipt()))
    out = _guard(text, messages)
    assert "delivered to their inbox" not in out
    assert "I sent the email" in out
    assert "provider accepted" in out.lower()
    assert "Inbox delivery was not verified" in out


def test_acceptance_other_than_provider_is_not_delivery():
    messages = _turn(
        f"/opt/mail/send {RECIPIENT}",
        _tool_result(_receipt(acceptance="delivered")),
    )
    out = _guard(f"I sent the email to {RECIPIENT}.", messages)
    assert "not reported as sent" in out
    assert "I sent the email" not in out


def test_empty_prefixes_do_not_treat_a_printed_receipt_as_sent():
    messages = _turn(f"/opt/mail/send {RECIPIENT}", _tool_result(_receipt()))
    out = _guard(f"I sent the email to {RECIPIENT}.", messages, prefixes=())
    assert "not reported as sent" in out
    assert "I sent the email" not in out


def test_unrelated_reply_is_unchanged():
    text = "I updated the script and left the image queue alone."
    assert _guard(text, []) == text


def test_mixed_reply_drops_only_the_unverified_send_claim():
    messages = [{"role": "user", "content": "status"}]
    out = _guard(f"I updated the script. I sent the email to {RECIPIENT}.", messages)
    assert "updated the script" in out
    assert "I sent the email" not in out
    assert "Nothing was sent" in out


def test_negated_send_is_not_rewritten():
    text = "I did not send the email."
    assert _guard(text, []) == text


def test_earlier_turn_receipt_does_not_verify_this_turn():
    prior = _turn(f"/opt/mail/send {RECIPIENT}", _tool_result(_receipt()))
    messages = prior + [{"role": "user", "content": "did it go?"}]
    out = _guard(f"I sent the email to {RECIPIENT}.", messages)
    assert "Nothing was sent" in out


def test_recipient_mismatch_is_not_reported_sent():
    messages = _turn(
        "/opt/mail/send b@example.com",
        _tool_result(_receipt(recipient="b@example.com")),
    )
    out = _guard(f"I sent the email to {RECIPIENT}.", messages)
    assert "claimed recipient" in out
    assert "I sent the email" not in out


def test_rewrite_is_idempotent_and_does_not_ask_for_a_retry():
    out = _guard(f"I sent the email to {RECIPIENT}.", [])
    assert guard_reply(out, [], enabled=True, prefixes=PREFIX) == out
    assert "resend" not in out.lower()
    assert "not retried" in out
