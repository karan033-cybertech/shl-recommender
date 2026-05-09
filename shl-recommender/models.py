"""
Pydantic models for request/response schemas.

TODO:
- Expand request models to support conversation history, user profile, constraints, etc.
- Add richer recommendation metadata if needed (e.g., score, rationale, tags).
- Validate URLs / enums for `test_type` once catalog schema is finalized.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Recommendation(BaseModel):
    """
    Represents a single SHL assessment recommendation.

    IMPORTANT: Field names must match the required response schema exactly.
    """

    name: str = Field(..., description="Assessment name")
    url: str = Field(..., description="Assessment URL")
    test_type: str = Field(..., description="Assessment/test type (e.g., cognitive, personality)")


class ChatMessage(BaseModel):
    """One turn in a stateless conversation (full history sent each request)."""

    role: Literal["user", "assistant"] = Field(..., description="Speaker role")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    """
    Chat request: either `messages` (full history) or legacy single `message`.
    """

    message: Optional[str] = Field(
        default=None,
        description="Latest user message only (legacy; use `messages` when possible)",
    )
    messages: List[ChatMessage] = Field(
        default_factory=list,
        description="Full conversation history in order (stateless API)",
    )

    @model_validator(mode="after")
    def _require_some_input(self) -> ChatRequest:
        if self.messages:
            return self
        if self.message is not None and str(self.message).strip():
            return self
        raise ValueError("Provide non-empty `messages` or a non-empty `message`.")

    def resolved_messages(self) -> List[ChatMessage]:
        if self.messages:
            return list(self.messages)
        return [ChatMessage(role="user", content=str(self.message).strip())]


class ChatResponse(BaseModel):
    """
    Response schema for POST /chat.

    MUST be exactly:
    {
      "reply": "string",
      "recommendations": [{"name": "...", "url": "...", "test_type": "..."}],
      "end_of_conversation": boolean
    }
    """

    reply: str = Field(..., description="Assistant reply message")
    recommendations: List[Recommendation] = Field(
        default_factory=list, description="List of recommended assessments"
    )
    end_of_conversation: bool = Field(
        False, description="Whether the conversation should end"
    )

