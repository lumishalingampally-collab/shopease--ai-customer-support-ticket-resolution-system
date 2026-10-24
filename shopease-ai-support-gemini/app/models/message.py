"""SQLAlchemy model and enum for Conversation Messages."""

import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum
from sqlalchemy import Column, String, DateTime, Text, Enum, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base


class SenderType(str, PyEnum):
    """Origin of a conversation message."""
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    HUMAN_AGENT = "HUMAN_AGENT"


class Message(Base):
    """Conversation Message database model."""

    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    ticket_id = Column(
        String(36),
        ForeignKey("tickets.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    sender = Column(Enum(SenderType), nullable=False)
    content = Column(Text, nullable=False)
    language = Column(String(10), default="en", nullable=False)
    context_used = Column(Text, nullable=True)  # Store JSON-serialized KB sources retrieved
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    ticket = relationship("Ticket", back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message {self.id[:8]} ({self.sender.value}) for Ticket {self.ticket_id[:8]}>"
