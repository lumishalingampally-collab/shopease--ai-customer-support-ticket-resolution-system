"""FastAPI router for Support Ticket CRUD endpoints."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.ticket import Ticket, TicketStatus, TicketPriority, TicketCategory
from app.models.message import Message, SenderType
from app.schemas.message import MessageCreate, MessageResponse
from app.schemas.ticket import (
    TicketCreate,
    TicketUpdate,
    TicketResponse,
    TicketDetailResponse,
)

router = APIRouter(prefix="/tickets", tags=["Tickets"])


@router.post(
    "",
    response_model=TicketResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new support ticket"
)
def create_ticket(
    ticket_in: TicketCreate,
    db: Session = Depends(get_db)
):
    """Create a new support ticket and optionally record an initial message."""
    db_ticket = Ticket(
        customer_name=ticket_in.customer_name,
        customer_email=ticket_in.customer_email,
        category=ticket_in.category or TicketCategory.GENERAL,
        priority=ticket_in.priority or TicketPriority.MEDIUM,
        intent=ticket_in.intent,
        sentiment=ticket_in.sentiment,
        detected_language=ticket_in.detected_language or "en",
        requires_human=ticket_in.requires_human or False,
        escalation_reason=ticket_in.escalation_reason,
    )
    db.add(db_ticket)
    db.flush()  # Populates db_ticket.id and defaults

    # If an initial user message was provided, store it
    if ticket_in.initial_message:
        initial_msg = Message(
            ticket_id=db_ticket.id,
            sender=SenderType.USER,
            content=ticket_in.initial_message,
            language=ticket_in.detected_language or "en",
        )
        db.add(initial_msg)

    db.commit()
    db.refresh(db_ticket)
    return db_ticket


@router.get(
    "",
    response_model=List[TicketResponse],
    summary="List support tickets with optional filtering"
)
def list_tickets(
    status_filter: Optional[TicketStatus] = Query(None, alias="status", description="Filter by ticket status"),
    priority: Optional[TicketPriority] = Query(None, description="Filter by priority level"),
    category: Optional[TicketCategory] = Query(None, description="Filter by category"),
    requires_human: Optional[bool] = Query(None, description="Filter by human escalation requirement"),
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(50, ge=1, le=100, description="Pagination limit"),
    db: Session = Depends(get_db)
):
    """Retrieve tickets with optional filtering and pagination."""
    query = db.query(Ticket)

    if status_filter is not None:
        query = query.filter(Ticket.status == status_filter)
    if priority is not None:
        query = query.filter(Ticket.priority == priority)
    if category is not None:
        query = query.filter(Ticket.category == category)
    if requires_human is not None:
        query = query.filter(Ticket.requires_human == requires_human)

    tickets = query.order_by(Ticket.created_at.desc()).offset(skip).limit(limit).all()
    return tickets


@router.get(
    "/{ticket_id}",
    response_model=TicketDetailResponse,
    summary="Get ticket details and conversation history"
)
def get_ticket(
    ticket_id: str,
    db: Session = Depends(get_db)
):
    """Retrieve a single ticket by its UUID or ticket number with its message history."""
    ticket = db.query(Ticket).filter(
        (Ticket.id == ticket_id) | (Ticket.ticket_number == ticket_id)
    ).first()

    if not ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket '{ticket_id}' not found."
        )

    return ticket




@router.post(
    "/{ticket_id}/messages",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Append a message to an existing ticket"
)
def add_ticket_message(
    ticket_id: str,
    message_in: MessageCreate,
    db: Session = Depends(get_db)
):
    """Persist a customer/assistant/human message without keeping it only in UI memory."""
    ticket = db.query(Ticket).filter(
        (Ticket.id == ticket_id) | (Ticket.ticket_number == ticket_id)
    ).first()
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Ticket '{ticket_id}' not found.")
    message = Message(
        ticket_id=ticket.id,
        sender=message_in.sender,
        content=message_in.content,
        language=message_in.language,
        context_used=message_in.context_used,
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


@router.get(
    "/{ticket_id}/messages",
    response_model=List[MessageResponse],
    summary="Get persistent conversation messages for a ticket"
)
def get_ticket_messages(
    ticket_id: str,
    db: Session = Depends(get_db)
):
    ticket = db.query(Ticket).filter(
        (Ticket.id == ticket_id) | (Ticket.ticket_number == ticket_id)
    ).first()
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Ticket '{ticket_id}' not found.")
    return ticket.messages

@router.patch(
    "/{ticket_id}",
    response_model=TicketResponse,
    summary="Update ticket metadata or status"
)
def update_ticket(
    ticket_id: str,
    ticket_update: TicketUpdate,
    db: Session = Depends(get_db)
):
    """Update ticket attributes such as status, priority, or escalation flags."""
    ticket = db.query(Ticket).filter(
        (Ticket.id == ticket_id) | (Ticket.ticket_number == ticket_id)
    ).first()

    if not ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket '{ticket_id}' not found."
        )

    update_data = ticket_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(ticket, field, value)

    db.commit()
    db.refresh(ticket)
    return ticket
