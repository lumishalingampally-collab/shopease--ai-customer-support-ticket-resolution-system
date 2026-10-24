"""Pydantic schemas for Support Tickets."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict, Field, EmailStr
from app.models.ticket import (
    TicketStatus,
    TicketPriority,
    TicketCategory,
    SentimentType,
)
from app.schemas.message import MessageResponse


class TicketBase(BaseModel):
    """Base fields for a support ticket."""
    customer_name: Optional[str] = Field(default=None, max_length=100)
    customer_email: Optional[str] = Field(default=None, max_length=100)
    category: TicketCategory = TicketCategory.GENERAL
    priority: TicketPriority = TicketPriority.MEDIUM
    intent: Optional[str] = Field(default=None, max_length=100)
    sentiment: SentimentType = SentimentType.NEUTRAL
    detected_language: str = Field(default="en", max_length=10)
    requires_human: bool = False
    escalation_reason: Optional[str] = None


class TicketCreate(BaseModel):
    """Schema for creating a ticket."""
    customer_name: Optional[str] = Field(default=None, max_length=100)
    customer_email: Optional[str] = Field(default=None, max_length=100)
    category: Optional[TicketCategory] = TicketCategory.GENERAL
    priority: Optional[TicketPriority] = TicketPriority.MEDIUM
    initial_message: Optional[str] = Field(
        default=None,
        description="Optional initial user query to log as first conversation message"
    )
    intent: Optional[str] = None
    sentiment: Optional[SentimentType] = SentimentType.NEUTRAL
    detected_language: Optional[str] = "en"
    requires_human: Optional[bool] = False
    escalation_reason: Optional[str] = None


class TicketUpdate(BaseModel):
    """Schema for updating an existing ticket."""
    status: Optional[TicketStatus] = None
    priority: Optional[TicketPriority] = None
    category: Optional[TicketCategory] = None
    requires_human: Optional[bool] = None
    escalation_reason: Optional[str] = None
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None


class TicketResponse(TicketBase):
    """Schema for returning ticket metadata."""
    id: str
    ticket_number: str
    status: TicketStatus
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TicketDetailResponse(TicketResponse):
    """Schema for ticket with full message history."""
    messages: List[MessageResponse] = []

    model_config = ConfigDict(from_attributes=True)
