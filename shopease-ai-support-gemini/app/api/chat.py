"""FastAPI router for end-to-end grounded AI customer chat."""

import json
import logging
from typing import List, Dict
from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.ticket import Ticket, TicketStatus, TicketCategory
from app.models.message import Message, SenderType
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.classifier_service import ClassifierService, get_classifier_service
from app.services.rag_service import RAGService, get_rag_service
from app.services.response_service import ResponseService, get_response_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["AI Customer Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Multi-turn grounded AI customer support chat"
)
def chat_with_support(
    request: ChatRequest,
    db: Session = Depends(get_db),
    classifier_service: ClassifierService = Depends(get_classifier_service),
    rag_service: RAGService = Depends(get_rag_service),
    response_service: ResponseService = Depends(get_response_service)
):
    """End-to-end AI chat pipeline:

    1. Retrieve or create ticket in SQLite.
    2. Extract recent conversation message history.
    3. Classify customer query and detect intent/sentiment/language.
    4. Retrieve relevant knowledge base policy chunks via ChromaDB.
    5. Generate strictly grounded response via Gemini (with fallback).
    6. Persist customer and assistant messages to database.
    7. Update ticket lifecycle status and return structured response.
    """
    user_query = request.query.strip()

    # 1. Retrieve or create ticket
    ticket: Ticket
    if request.ticket_id:
        ticket = db.query(Ticket).filter(
            (Ticket.id == request.ticket_id) | (Ticket.ticket_number == request.ticket_id)
        ).first()
        if not ticket:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ticket '{request.ticket_id}' not found."
            )
    else:
        ticket = Ticket(
            customer_name=request.customer_name or "Valued Customer",
            customer_email=request.customer_email,
            status=TicketStatus.OPEN,
            category=TicketCategory.OTHER,
        )
        db.add(ticket)
        db.flush()

    # 2. Extract recent conversation history from SQLite
    conversation_history: List[Dict[str, str]] = []
    if ticket.messages:
        for msg in ticket.messages[-6:]:
            conversation_history.append({
                "sender": msg.sender.value,
                "content": msg.content
            })

    # 3. Resolve whether this is a genuine follow-up or a new topic.
    # A short query such as "How much does it cost?" needs the previous topic,
    # but a new explicit topic such as "Can I get same-day delivery in Hyderabad?"
    # must not inherit an earlier payment conversation.
    def _has_explicit_topic(text: str) -> bool:
        q = text.lower()
        topic_markers = (
            "payment", "paid", "card", "upi", "refund", "return", "cashback", "coupon",
            "delivery", "shipping", "ship", "tracking", "track", "courier", "package",
            "parcel", "same-day", "same day", "express", "order", "password", "login",
            "account", "otp", "damaged", "broken", "wrong item", "defective", "cancel",
            "address", "discount", "promo", "voucher",
            "भुगतान", "रिफंड", "रिटर्न", "डिलीवरी", "ट्रैकिंग", "ऑर्डर",
            "చెల్లింపు", "రిఫండ్", "రిటర్న్", "డెలివరీ", "ఆర్డర్",
        )
        return any(marker in q for marker in topic_markers)

    def _is_follow_up(text: str) -> bool:
        q = text.lower().strip()
        if _has_explicit_topic(q):
            return False
        markers = (
            "what about", "how much", "how many", "what is the cost", "what's the cost",
            "how much does it cost", "when is the cutoff", "what is the cutoff",
            "what's the cutoff", "when", "where", "why", "and then", "what should i do",
            "what about it", "what about that", "can i", "is it", "does it", "will it",
            "how long", "how long should i wait", "how long to wait",
        )
        return len(q.split()) <= 10 and (any(q.startswith(m) for m in markers) or q in {"how much?", "how many?", "when?", "where?", "why?", "cost?", "cutoff?"})

    classification_query = user_query
    retrieval_query = user_query
    previous_user = None
    is_follow_up = False
    if conversation_history:
        topic_user = next(
            (m["content"] for m in reversed(conversation_history) if m.get("sender") == "USER" and _has_explicit_topic(m["content"])),
            None,
        )
        previous_user = topic_user or next(
            (m["content"] for m in reversed(conversation_history) if m.get("sender") == "USER"),
            None,
        )

    if previous_user and _is_follow_up(user_query):
        is_follow_up = True
        classification_query = f"Previous customer topic: {previous_user}\nFollow-up customer question: {user_query}"
        retrieval_query = classification_query

    classification = classifier_service.classify_query(query=classification_query)
    rag_result = rag_service.search(query=retrieval_query, top_k=3, classification=classification)
    retrieved_chunks = rag_result.results
    unique_sources = list(dict.fromkeys([chunk.source for chunk in retrieved_chunks]))
    policy_references = [response_service._policy_reference(chunk) for chunk in retrieved_chunks]
    retrieved_evidence = [
        {
            "source": chunk.source,
            "section": chunk.topic,
            "reference_id": ref.reference_id,
            "relevance_score": ref.relevance_score,
            "content": "\n".join(response_service._select_evidence(retrieval_query, chunk.content, max_lines=8)) or chunk.content,
        }
        for chunk, ref in zip(retrieved_chunks, policy_references)
    ]

    # 5. Generate Grounded Response
    # A deliberately vague issue should trigger a clarification question rather
    # than a random semantically similar policy.
    if classification.intent == "needs_clarification":
        clarification = response_service.clarification_response(classification.language)
        generated_resp = clarification
    else:
        generated_resp = response_service.generate_response(
        query=user_query,
        classification=classification,
        retrieved_chunks=retrieved_chunks,
            conversation_history=(conversation_history[-6:] if is_follow_up else [])
        )

    # 6. Update Ticket Lifecycle and Escalation State
    should_escalate = (
        classification.requires_human
        or generated_resp.needs_human_review
        or not generated_resp.sufficient_information
    )

    if should_escalate:
        ticket.status = TicketStatus.ESCALATED
        ticket.requires_human = True
        ticket.escalation_reason = (
            classification.escalation_reason
            or generated_resp.reason
            or "Flagged for human support review."
        )
    elif ticket.status == TicketStatus.OPEN:
        ticket.status = TicketStatus.IN_PROGRESS

    # Update metadata fields
    ticket.category = classification.category
    ticket.priority = classification.priority
    ticket.intent = classification.intent
    ticket.sentiment = classification.sentiment
    ticket.detected_language = generated_resp.language or classification.language

    # 7. Persist User and Assistant Messages in SQLite
    user_message = Message(
        ticket_id=ticket.id,
        sender=SenderType.USER,
        content=user_query,
        language=classification.language
    )
    db.add(user_message)

    assistant_message = Message(
        ticket_id=ticket.id,
        sender=SenderType.ASSISTANT,
        content=generated_resp.response,
        language=generated_resp.language,
        context_used=json.dumps({
            "original_query": user_query,
            "retrieval_query": retrieval_query,
            "retrieved_evidence": retrieved_evidence,
            "llm_used": generated_resp.llm_used,
            "generation_method": generated_resp.generation_method,
            "generation_model": generated_resp.generation_model,
            "grounding_verified": generated_resp.grounding_verified,
            "sufficient_information": generated_resp.sufficient_information,
            "escalated": ticket.requires_human,
            "escalation_reason": ticket.escalation_reason,
        }, ensure_ascii=False) if retrieved_evidence or generated_resp.generation_method else None
    )
    db.add(assistant_message)

    db.commit()
    db.refresh(ticket)

    logger.info(
        "RAG Trace: original_query='%s' | retrieval_query='%s' | retrieved_chunks=%d | "
        "llm_used=%s | model='%s' | method='%s' | grounding_verified=%s | "
        "sufficient_info=%s | escalate=%s | reason='%s'",
        user_query,
        retrieval_query,
        len(retrieved_chunks),
        generated_resp.llm_used,
        generated_resp.generation_model or "none",
        generated_resp.generation_method,
        generated_resp.grounding_verified,
        generated_resp.sufficient_information,
        ticket.requires_human,
        ticket.escalation_reason or "none"
    )

    # 8. Return Structured API Response
    return ChatResponse(
        ticket_id=ticket.id,
        ticket_number=ticket.ticket_number,
        response=generated_resp.response,
        reply=generated_resp.response,
        classification=classification,
        sources=unique_sources,
        policy_references=generated_resp.policy_references or policy_references,
        retrieved_evidence=retrieved_evidence,
        used_knowledge_base=generated_resp.used_knowledge_base,
        sufficient_information=generated_resp.sufficient_information,
        needs_human_review=ticket.requires_human,
        status=ticket.status,
        llm_used=generated_resp.llm_used,
        generation_method=generated_resp.generation_method,
        generation_model=generated_resp.generation_model,
        grounding_verified=generated_resp.grounding_verified,
        retrieval_query=retrieval_query,
    )
