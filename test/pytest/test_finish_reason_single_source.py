# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Pin the engine's finish-reason vocabulary and each interface's mapping.

``FinishReason`` says why a request ended; every HTTP interface maps it to its
own wire values. These tests hold the enum, the single guard
(``require_reportable``) and the five mappings to the same contract, so one
interface cannot silently report a different story for the same engine state.
"""

import asyncio
import inspect
import json
from types import SimpleNamespace

import pytest

import chitu.dp_token_router as router_module
import chitu.serve.anthropic_api as anthropic
import chitu.serve.openai_api as chat
import chitu.serve.responses_api as responses
from chitu.dp_token_sender import DPAsyncDataStream, DPTokenSender
from chitu.finish_reason import FinishReason
from chitu.serve.common import require_reportable
from chitu.task import UserRequest

PUBLIC = [
    FinishReason.STOP,
    FinishReason.LENGTH,
    FinishReason.TOOL_CALLS,
    FinishReason.CONTENT_FILTER,
]
INTERNAL = [
    FinishReason.UNKNOWN,
    FinishReason.ABORTED,
    FinishReason.EVICTED,
    FinishReason.PREFILL_ONLY,
]
# UNKNOWN and ERROR never reach a mapping: the guard turns both into errors.
REPORTABLE = PUBLIC + [
    FinishReason.ABORTED,
    FinishReason.EVICTED,
    FinishReason.PREFILL_ONLY,
]


class StaticStream:
    """Minimal async stream: yields the given (text, state, (logprobs, tokens))."""

    def __init__(self, items=(), error_message=None):
        self.items = items
        self.tokens_len = len(items)
        self.reasoning_tokens = 0
        self.input_cached_tokens = None
        self.error_message = error_message
        self.output_parser = SimpleNamespace(
            params=SimpleNamespace(enable_reasoning=False, separated_tool_calling=False)
        )

    async def wait_input_cached_tokens(self):
        return self.input_cached_tokens

    def __aiter__(self):
        self.iterator = iter(self.items)
        return self

    async def __anext__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            raise StopAsyncIteration


def fake_request(reason, *, error_message=None):
    return SimpleNamespace(
        request_id="req-finish-reason",
        prompt_len=4,
        finish_reason=reason,
        tool_call_params=None,
        response_format_params=None,
        logprobs=False,
        top_logprobs=None,
        output="hi",
        async_stream=StaticStream((("hi", False, (None, None)),), error_message),
        num_hit_tokens=0,
    )


def sse_events(chunks):
    result = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if line.startswith("data: ") and line[6:] != "[DONE]":
                result.append(json.loads(line[6:]))
    return result


async def collect(stream):
    return [chunk async for chunk in stream]


# --------------------------------------------------------------------------
# the enum itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason", REPORTABLE + [FinishReason.ERROR, FinishReason.UNKNOWN]
)
def test_every_member_knows_its_spelling_and_whether_it_is_reportable(reason):
    assert str(reason) == reason.value
    assert reason.is_reportable() is (reason in PUBLIC)


def test_only_the_protocol_values_are_reportable():
    assert {reason for reason in FinishReason if reason.is_reportable()} == set(PUBLIC)


def test_every_member_is_classified_exactly_once():
    # Every member is reported to the client, ends the request with an error, or
    # is logged and mapped to the interface's closest value. A new member that
    # skips all three buckets fails here instead of silently reporting "stop".
    reported = {reason for reason in FinishReason if reason.is_reportable()}
    failed = {reason for reason in FinishReason if reason.is_error()}
    logged = {
        FinishReason.ABORTED,
        FinishReason.EVICTED,
        FinishReason.PREFILL_ONLY,
    }
    assert reported | failed | logged == set(FinishReason)
    assert not (reported & failed)
    assert not (reported & logged)
    assert not (failed & logged)


def test_the_wire_spelling_round_trips_to_the_member():
    assert FinishReason("length") is FinishReason.LENGTH
    with pytest.raises(ValueError):
        FinishReason("stoped")


# --------------------------------------------------------------------------
# the single guard
# --------------------------------------------------------------------------


def test_unknown_is_reported_as_an_error(caplog):
    # A request that never recorded a reason carries UNKNOWN, so nothing
    # downstream needs a `None` branch; the client gets an error (we cannot say
    # why it ended) and the log says which request it was.
    with caplog.at_level("ERROR", logger="chitu.serve.common"):
        with pytest.raises(RuntimeError, match="without a recorded finish reason"):
            require_reportable(fake_request(FinishReason.UNKNOWN))
    errors = [
        r
        for r in caplog.records
        if r.levelname == "ERROR" and r.name == "chitu.serve.common"
    ]
    assert len(errors) == 1
    assert "req-finish-reason" in errors[0].getMessage()


def test_unknown_prefers_the_message_the_engine_recorded():
    with pytest.raises(RuntimeError, match="engine went away"):
        require_reportable(
            fake_request(FinishReason.UNKNOWN, error_message="engine went away")
        )


@pytest.mark.parametrize(
    "reason", [FinishReason.ABORTED, FinishReason.EVICTED, FinishReason.PREFILL_ONLY]
)
def test_other_internal_states_log_an_error(reason, caplog):
    with caplog.at_level("ERROR", logger="chitu.serve.common"):
        assert require_reportable(fake_request(reason)) is reason
    errors = [
        r
        for r in caplog.records
        if r.levelname == "ERROR" and r.name == "chitu.serve.common"
    ]
    assert len(errors) == 1
    assert reason.value in errors[0].getMessage()


@pytest.mark.parametrize("reason", PUBLIC)
def test_public_values_pass_through_quietly(reason, caplog):
    with caplog.at_level("ERROR", logger="chitu.serve.common"):
        assert require_reportable(fake_request(reason)) is reason
    assert [r for r in caplog.records if r.name == "chitu.serve.common"] == []


def test_engine_error_raises_with_the_message_the_engine_recorded():
    with pytest.raises(RuntimeError, match="TTFT timeout"):
        require_reportable(
            fake_request(FinishReason.ERROR, error_message="TTFT timeout")
        )


def test_engine_error_without_a_message_stays_generic():
    # The request id is an internal identifier: it belongs in the log, not in
    # what the client is told.
    with pytest.raises(RuntimeError) as caught:
        require_reportable(fake_request(FinishReason.ERROR))
    assert "req-finish-reason" not in str(caught.value)
    assert "internal engine error" in str(caught.value)


# --------------------------------------------------------------------------
# one mapping per interface
# --------------------------------------------------------------------------


CHAT = {
    FinishReason.LENGTH: "length",
    FinishReason.TOOL_CALLS: "tool_calls",
    FinishReason.CONTENT_FILTER: "content_filter",
}
COMPLETIONS = {
    FinishReason.LENGTH: "length",
    FinishReason.CONTENT_FILTER: "content_filter",
}
MESSAGES = {FinishReason.LENGTH: "max_tokens"}
COMPLETE = {FinishReason.LENGTH: "max_tokens"}
RESPONSES = {FinishReason.LENGTH: "incomplete"}


@pytest.mark.parametrize("reason", REPORTABLE)
def test_chat_maps_every_reason_into_its_protocol(reason):
    response = chat.AsyncResponse(fake_request(reason), response_model="m", created=1)
    assert response._finish_reason(has_tool_calls=False) == CHAT.get(reason, "stop")


@pytest.mark.parametrize("reason", REPORTABLE)
def test_completions_maps_every_reason_into_its_protocol(reason):
    body = asyncio.run(
        chat.CompletionAsyncResponse(
            fake_request(reason), response_model="m"
        ).full_generator()
    ).model_dump()
    assert body["choices"][0]["finish_reason"] == COMPLETIONS.get(reason, "stop")


@pytest.mark.parametrize("reason", REPORTABLE)
def test_anthropic_maps_every_reason_into_its_protocol(reason):
    assert anthropic.map_finish_reason_to_stop_reason(reason) == MESSAGES.get(
        reason, "end_turn"
    )
    assert anthropic.map_finish_reason_to_completion_stop_reason(
        reason
    ) == COMPLETE.get(reason, "stop_sequence")


@pytest.mark.parametrize("reason", REPORTABLE)
def test_responses_maps_every_reason_into_its_protocol(reason):
    status, details = responses._response_status_and_incomplete_details(
        fake_request(reason)
    )
    assert status == RESPONSES.get(reason, "completed")
    assert details == (
        {"reason": "max_output_tokens"} if status == "incomplete" else None
    )


def test_a_tool_call_outranks_stop_but_not_a_truncation():
    response = chat.AsyncResponse(
        fake_request(FinishReason.STOP), response_model="m", created=1
    )
    assert response._finish_reason(has_tool_calls=True) == "tool_calls"
    truncated = chat.AsyncResponse(
        fake_request(FinishReason.LENGTH), response_model="m", created=1
    )
    assert truncated._finish_reason(has_tool_calls=True) == "length"


# --------------------------------------------------------------------------
# the engine error path: nothing is reported as a normal completion
# --------------------------------------------------------------------------


def test_every_interface_raises_for_an_engine_error():
    with pytest.raises(RuntimeError, match="engine went away"):
        chat.AsyncResponse(
            fake_request(FinishReason.ERROR, error_message="engine went away"),
            response_model="m",
            created=1,
        )._finish_reason(has_tool_calls=False)
    with pytest.raises(RuntimeError, match="engine went away"):
        responses._response_status_and_incomplete_details(
            fake_request(FinishReason.ERROR, error_message="engine went away")
        )


def test_chat_stream_ends_with_an_error_chunk_carrying_our_message():
    response = chat.AsyncResponse(
        fake_request(FinishReason.ERROR, error_message="TTFT timeout"),
        response_model="m",
        created=1,
    )
    raw = asyncio.run(collect(response.stream_generator(include_usage=False)))
    assert raw[-1] != "data: [DONE]\n\n"
    assert json.loads(raw[-1][len("data: ") :])["detail"] == "TTFT timeout"


def test_completions_stream_ends_with_an_error_chunk_carrying_our_message():
    response = chat.CompletionAsyncResponse(
        fake_request(FinishReason.ERROR, error_message="TTFT timeout"),
        response_model="m",
    )
    raw = asyncio.run(collect(response.stream_generator(include_usage=False)))
    assert raw[-1] != "data: [DONE]\n\n"
    assert json.loads(raw[-1][len("data: ") :])["detail"] == "TTFT timeout"


def test_anthropic_stream_ends_with_an_error_event_carrying_our_message():
    raw = asyncio.run(
        collect(
            anthropic.anthropic_stream_from_async_stream(
                user_req=fake_request(FinishReason.ERROR, error_message="TTFT timeout"),
                response_model="m",
            )
        )
    )
    last = sse_events(raw)[-1]
    assert last["type"] == "error"
    assert last["error"]["message"] == "TTFT timeout"


def test_responses_stream_ends_with_an_error_event_carrying_our_message():
    raw = asyncio.run(
        collect(
            responses.responses_stream_from_async_stream(
                user_req=fake_request(FinishReason.ERROR, error_message="TTFT timeout"),
                request=responses.ResponsesCreateRequest(input="hi"),
                response_model="m",
                public_tools=[],
                public_tool_choice="auto",
                created_at=1,
            )
        )
    )
    last = sse_events(raw)[-1]
    assert last["type"] == "error"
    assert last["message"] == "TTFT timeout"


# --------------------------------------------------------------------------
# engine side: ending a stream requires a reason, the default is UNKNOWN
# --------------------------------------------------------------------------


def bare_request():
    """A UserRequest with only what stopping a stream touches."""
    stream = SimpleNamespace(
        stop_signal=False,
        input_cached_tokens=None,
        seqs=[],
        send_stop_signal=lambda error=None: None,
        set_input_cached_tokens=lambda value: None,
    )
    req = UserRequest.__new__(UserRequest)
    req.request_id = "req-bare"
    req.finish_reason = FinishReason.UNKNOWN
    req.output = ""
    req.completion_time = 0.0
    req.save_trace_dir = None
    req.num_hit_tokens = 0
    req.async_stream = stream
    return req


def test_a_new_request_starts_at_unknown_so_no_reader_sees_none():
    # Built through the real constructor: this pins __post_init__'s default,
    # which a hand-filled object would only assert back to itself.
    req = UserRequest.create_mock(
        input_len=4, request_id="req-default", enable_thinking=False
    )
    assert req.finish_reason is FinishReason.UNKNOWN


def test_stop_stream_takes_the_reason_as_a_required_argument():
    params = inspect.signature(UserRequest.stop_stream).parameters
    assert params["reason"].default is inspect.Parameter.empty


def test_a_real_reason_replaces_unknown():
    req = bare_request()
    req.stop_stream(FinishReason.LENGTH)
    assert req.finish_reason is FinishReason.LENGTH


@pytest.mark.parametrize("first", [FinishReason.LENGTH, FinishReason.STOP])
@pytest.mark.parametrize("late", [FinishReason.ABORTED, FinishReason.EVICTED])
def test_a_late_reason_never_relabels_the_request(first, late):
    # Covers both the late abort and the eviction mark: a request that already
    # ended for its own reason keeps it.
    req = bare_request()
    req.stop_stream(first)
    req.stop_stream(late)
    assert req.finish_reason is first


def test_set_finish_reason_needs_a_member():
    with pytest.raises(ValueError):
        FinishReason("stoped")


# --------------------------------------------------------------------------
# transport layer: the frame always carries a spelling
# --------------------------------------------------------------------------


def test_send_finish_takes_the_reason_as_a_required_argument():
    params = inspect.signature(DPTokenSender.send_finish).parameters
    assert params["finish_reason"].default is inspect.Parameter.empty


@pytest.mark.parametrize(
    "reason", [FinishReason.LENGTH, FinishReason.ABORTED, FinishReason.UNKNOWN]
)
def test_dp_finish_frame_carries_the_spelling(reason):
    task = SimpleNamespace(
        req=SimpleNamespace(
            request_id="req-dp",
            finish_reason=reason,
            num_hit_tokens=0,
            save_trace_dir=None,
        )
    )
    sender = DPTokenSender()
    frames = []
    sender._send_data = frames.append
    stream = DPAsyncDataStream.build(
        enable_thinking=False, task=task, token_sender=sender
    )
    stream.input_cached_tokens = None

    stream.send_stop_signal()

    finish_frames = [frame for frame in frames if frame.get("type") == "finish"]
    assert len(finish_frames) == 1
    assert finish_frames[0]["finish_reason"] == reason.value


def test_token_router_rejects_a_frame_without_a_reason(monkeypatch):
    router = router_module.TokenRouter(SimpleNamespace())
    req = bare_request()
    req.request_id = "req-no-key"
    req.completion_time = 0.0
    req.start_time = 0.0
    router.active_requests[req.request_id] = req

    with pytest.raises(KeyError):
        asyncio.run(
            router._process_token_data({"type": "finish", "request_id": req.request_id})
        )
