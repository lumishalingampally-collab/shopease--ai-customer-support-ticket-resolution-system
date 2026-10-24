"""Unit tests for SQLite database models and relationships."""

import pytest
from app.models.ticket import (
    Ticket,
    TicketStatus,
    TicketPriority,
    TicketCategory,
    SentimentType,
)
from app.models.message import Message, SenderType


def test_create_ticket_defaults(db_session):
    """Test that a Ticket initializes with correct defaults."""
    ticket = Ticket(
        customer_name="Rahul Verma",
        customer_email="rahul@example.com"
    )
    db_session.add(ticket)
    db_session.commit()
    db_session.refresh(ticket)

    assert ticket.id is not None
    assert ticket.ticket_number.startswith("TKT-")
    assert ticket.status == TicketStatus.OPEN
    assert ticket.priority == TicketPriority.MEDIUM
    assert ticket.category == TicketCategory.GENERAL
    assert ticket.sentiment == SentimentType.NEUTRAL
    assert ticket.requires_human is False
    assert ticket.created_at is not None
    assert ticket.updated_at is not None


def test_ticket_message_relationship_and_cascade(db_session):
    """Test message creation and cascade deletion with ticket."""
    ticket = Ticket(
        customer_name="Priya Patel",
        customer_email="priya@example.com",
        category=TicketCategory.PAYMENT,
        priority=TicketPriority.URGENT,
        requires_human=True,
        escalation_reason="Duplicate payment charged"
    )
    db_session.add(ticket)
    db_session.commit()
    db_session.refresh(ticket)

    msg1 = Message(
        ticket_id=ticket.id,
        sender=SenderType.USER,
        content="I was debited twice for order #9921",
        language="en"
    )
    msg2 = Message(
        ticket_id=ticket.id,
        sender=SenderType.ASSISTANT,
        content="We are looking into this payment dispute.",
        language="en"
    )
    db_session.add_all([msg1, msg2])
    db_session.commit()

    db_session.refresh(ticket)
    assert len(ticket.messages) == 2
    assert ticket.messages[0].sender == SenderType.USER
    assert ticket.messages[1].sender == SenderType.ASSISTANT
    assert msg1.ticket.ticket_number == ticket.ticket_number

    # Test cascade delete
    db_session.delete(ticket)
    db_session.commit()

    remaining_msgs = db_session.query(Message).filter(Message.ticket_id == ticket.id).all()
    assert len(remaining_msgs) == 0


def test_sqlite_file_persists_across_new_engine(tmp_path):
    """Requirement 7: ticket/history survive closing the DB engine and reopening it."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database import Base

    db_file = tmp_path / "persistent_shopease.db"
    url = f"sqlite:///{db_file.as_posix()}"
    engine1 = create_engine(url, connect_args={"check_same_thread": False})
    Session1 = sessionmaker(bind=engine1, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine1)
    s1 = Session1()
    ticket = Ticket(customer_name="Persistence Test", category=TicketCategory.SHIPPING)
    s1.add(ticket)
    s1.flush()
    s1.add(Message(ticket_id=ticket.id, sender=SenderType.USER, content="Where is my order?", language="en"))
    s1.commit()
    ticket_number = ticket.ticket_number
    s1.close()
    engine1.dispose()

    engine2 = create_engine(url, connect_args={"check_same_thread": False})
    Session2 = sessionmaker(bind=engine2, autocommit=False, autoflush=False)
    s2 = Session2()
    reopened = s2.query(Ticket).filter(Ticket.ticket_number == ticket_number).first()
    assert reopened is not None
    assert len(reopened.messages) == 1
    assert reopened.messages[0].content == "Where is my order?"
    s2.close()
    engine2.dispose()
