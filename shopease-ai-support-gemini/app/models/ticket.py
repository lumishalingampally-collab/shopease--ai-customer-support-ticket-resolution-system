"""SQLAlchemy models and enums for Support Tickets."""

import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum
from sqlalchemy import Column, String, Boolean, DateTime, Text, Enum
from sqlalchemy.orm import relationship
from app.database import Base


class TicketStatus(str, PyEnum):
    """Lifecycle states of a ticket."""
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"


class TicketPriority(str, PyEnum):
    """Urgency priority levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"


class TicketCategory(str, PyEnum):
    """Classification categories."""
    PAYMENT = "PAYMENT"
    RETURN_REFUND = "RETURN_REFUND"
    SHIPPING = "SHIPPING"
    ORDER = "ORDER"
    ACCOUNT = "ACCOUNT"
    PROMOTION = "PROMOTION"
    PRODUCT = "PRODUCT"
    OTHER = "OTHER"
    GENERAL = "OTHER"


class SentimentType(str, PyEnum):
    """Customer sentiment states."""
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    FRUSTRATED = "FRUSTRATED"


def generate_ticket_number() -> str:
    """Generate a unique human-friendly ticket identifier."""
    unique_suffix = uuid.uuid4().hex[:6].upper()
    return f"TKT-{unique_suffix}"


class Ticket(Base):
    """Support Ticket database model."""

    __tablename__ = "tickets"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    ticket_number = Column(String(20), unique=True, index=True, default=generate_ticket_number, nullable=False)
    customer_name = Column(String(100), nullable=True)
    customer_email = Column(String(100), nullable=True)

    status = Column(Enum(TicketStatus), default=TicketStatus.OPEN, nullable=False, index=True)
    category = Column(Enum(TicketCategory), default=TicketCategory.GENERAL, nullable=False)
    priority = Column(Enum(TicketPriority), default=TicketPriority.MEDIUM, nullable=False)

    intent = Column(String(100), nullable=True)
    sentiment = Column(Enum(SentimentType), default=SentimentType.NEUTRAL, nullable=False)
    detected_language = Column(String(10), default="en", nullable=False)

    requires_human = Column(Boolean, default=False, nullable=False)
    escalation_reason = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Relationship to conversation messages
    messages = relationship(
        "Message",
        back_populates="ticket",
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )

    def __repr__(self) -> str:
        return f"<Ticket {self.ticket_number} - {self.status.value}>"
