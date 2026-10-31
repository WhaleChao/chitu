# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Why a request ended.

``UNKNOWN`` is the default until some path records a reason. Whether a reason
may be shown to a client is a property of the reason itself: see
:meth:`FinishReason.is_reportable` and :meth:`FinishReason.is_error`.
"""

from enum import Enum

__all__ = ["FinishReason"]


class FinishReason(str, Enum):
    """Why a request stopped."""

    UNKNOWN = "unknown"

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"

    ABORTED = "aborted"
    EVICTED = "evicted"
    PREFILL_ONLY = "prefill_only"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value

    def is_reportable(self) -> bool:
        """Whether a client may see this reason as a value of its own."""
        return self in _REPORTABLE

    def is_error(self) -> bool:
        """Whether the request must end with an error response."""
        return self in _ERRORS


_REPORTABLE = frozenset(
    {
        FinishReason.STOP,
        FinishReason.LENGTH,
        FinishReason.TOOL_CALLS,
        FinishReason.CONTENT_FILTER,
    }
)

# Reasons the client must see as a failed request rather than a finished one.
_ERRORS = frozenset({FinishReason.UNKNOWN, FinishReason.ERROR})
