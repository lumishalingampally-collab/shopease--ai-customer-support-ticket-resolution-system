"""Pydantic schemas for Conversation Messages."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field
from app.models.message import SenderType


class MessageBase(BaseModel):
    """Base fields for a message."""
    sender: SenderType
    content: str = Field(..., min_length=1, description="Message text content")
    language: str = Field(default="en", max_length=10, description="Language code")
    context_used: Optional[str] = Field(default=None, description="Serialized context/sources used")


class MessageCreate(MessageBase):
    """Schema for creating a new message."""
    pass


class MessageResponse(MessageBase):
    """Schema for returning message details."""
    id: str
    ticket_id: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
