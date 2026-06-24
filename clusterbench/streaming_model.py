"""StreamingLitellmModel — drop-in replacement for mini-swe-agent's LitellmModel.

Passed to mini-extra via --model-class when streaming=True so that each LLM
call goes through litellm in streaming mode. This makes TTFT and cache-miss
metrics visible on the LiteLLM proxy (matching real coding-agent behaviour
like opencode / Claude Code which always stream).

The only change from LitellmModel is _query():
  - calls litellm.completion(..., stream=True)
  - collects all chunks and reassembles them with stream_chunk_builder()
  - returns the same ModelResponse shape the rest of query() expects

Everything else (cost tracking, action parsing, retry, format errors) is
inherited unchanged.
"""
from __future__ import annotations

import litellm

from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL


class StreamingLitellmModel(LitellmModel):
    """LitellmModel that forces stream=True on every completion call.

    Reassembles the stream with litellm.stream_chunk_builder() so the
    returned object has the same shape as a non-streaming ModelResponse —
    no changes to the caller (query / _parse_actions / cost tracking)
    required.
    """

    def _query(self, messages: list[dict], **kwargs):
        # Remove any caller-supplied stream= so we always control it here.
        kwargs.pop("stream", None)
        try:
            stream = litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=[BASH_TOOL],
                stream=True,
                **(self.config.model_kwargs | kwargs),
            )
            chunks = list(stream)
        except litellm.exceptions.AuthenticationError as e:
            e.message += (
                " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            )
            raise e

        # Reassemble into a ModelResponse with the same shape as non-streaming.
        response = litellm.stream_chunk_builder(chunks, messages=messages)
        return response
