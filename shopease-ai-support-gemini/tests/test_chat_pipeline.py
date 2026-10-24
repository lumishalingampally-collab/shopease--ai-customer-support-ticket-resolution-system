"""Unit and integration tests for Grounded AI Response Generation & Chat Pipeline."""

from unittest.mock import patch, MagicMock
import pytest
from app.models.ticket import Ticket, TicketStatus, TicketPriority, TicketCategory
from app.models.message import Message, SenderType
from app.schemas.chat import GeneratedResponse, ClassificationResult
from app.services.rag_service import get_rag_service


@pytest.fixture(autouse=True)
def ensure_kb_indexed():
    """Ensure knowledge base is indexed for RAG retrieval during chat tests."""
    s = get_rag_service()
    if s.collection.count() == 0:
        s.index_documents(force_reindex=False)


# -------------------------------------------------------------
# 1. Successful payment response
# -------------------------------------------------------------
def test_1_successful_payment_response(client):
    """Test 1: General payment query receives grounded response with payment policy sources."""
    with patch("app.services.llm_service.LLMService.is_available", return_value=True), \
         patch("app.services.llm_service.LLMService.generate_structured") as mock_gen:
        mock_gen.side_effect = [
            # Grounded response; issue analysis is deterministic for this clear intent.
            GeneratedResponse(
                response="ShopEase accepts UPI, Credit/Debit cards, Net Banking, and Cash on Delivery for orders under ₹10,000.",
                language="English",
                used_knowledge_base=True,
                sufficient_information=True,
                needs_human_review=False
            )
        ]
        resp = client.post("/api/v1/chat", json={"query": "What payment methods are supported on ShopEase?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["used_knowledge_base"] is True
        assert data["sufficient_information"] is True
        assert "UPI" in data["response"]
        assert any("payment" in s.lower() for s in data["sources"])


# -------------------------------------------------------------
# 2. Successful refund response
# -------------------------------------------------------------
def test_2_successful_refund_response(client):
    """Test 2: Refund timeline question returns grounded return policy timelines."""
    with patch("app.services.llm_service.LLMService.is_available", return_value=True), \
         patch("app.services.llm_service.LLMService.generate_structured") as mock_gen:
        mock_gen.side_effect = [
            GeneratedResponse(
                response="Refunds to the original payment card take 5 to 7 business days, while UPI refunds take 24 to 48 hours.",
                language="English",
                used_knowledge_base=True,
                sufficient_information=True,
                needs_human_review=False
            )
        ]
        resp = client.post("/api/v1/chat", json={"query": "How long will my refund take to reach my card?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "5 to 7" in data["response"]
        assert data["needs_human_review"] is False
        assert any("returns" in s.lower() for s in data["sources"])


# -------------------------------------------------------------
# 3. Shipping response
# -------------------------------------------------------------
def test_3_shipping_response(client):
    """Test 3: Shipping tiers inquiry grounded in shipping policy."""
    with patch("app.services.llm_service.LLMService.is_available", return_value=True), \
         patch("app.services.llm_service.LLMService.generate_structured") as mock_gen:
        mock_gen.side_effect = [
            GeneratedResponse(
                response="Standard delivery takes 3 to 5 business days and is free on orders above ₹499.",
                language="English",
                used_knowledge_base=True,
                sufficient_information=True,
                needs_human_review=False
            )
        ]
        resp = client.post("/api/v1/chat", json={"query": "How long does standard delivery take?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "3 to 5" in data["response"]
        assert data["classification"]["category"] == "SHIPPING"


# -------------------------------------------------------------
# 4. Duplicate payment escalation
# -------------------------------------------------------------
def test_4_duplicate_payment_escalation(client):
    """Test 4: Duplicate payment is escalated and flagged for human specialist."""
    resp = client.post("/api/v1/chat", json={"query": "My card was charged twice for the same order."})
    assert resp.status_code == 200
    data = resp.json()

    assert data["classification"]["category"] == "PAYMENT"
    assert data["classification"]["intent"] == "duplicate_charge"
    assert data["classification"]["requires_human"] is True
    assert data["needs_human_review"] is True
    assert data["status"] == "ESCALATED"
    assert "duplicate" in data["response"].lower() or "human" in data["response"].lower() or "specialist" in data["response"].lower()


# -------------------------------------------------------------
# 5. Explicit human request
# -------------------------------------------------------------
def test_5_explicit_human_request(client):
    """Test 5: Explicit request to talk to a human agent marks ticket as ESCALATED."""
    resp = client.post("/api/v1/chat", json={"query": "I want to speak to a human agent right now."})
    assert resp.status_code == 200
    data = resp.json()

    assert data["needs_human_review"] is True
    assert data["status"] == "ESCALATED"
    assert data["classification"]["intent"] == "talk_to_agent"


# -------------------------------------------------------------
# 6. Insufficient KB information (Anti-Hallucination)
# -------------------------------------------------------------
def test_6_insufficient_kb_information(client):
    """Test 6: Unsupported query (e.g. ₹10,000 compensation) signals insufficient info and does not invent policies."""
    with patch("app.services.llm_service.LLMService.is_available", return_value=True), \
         patch("app.services.llm_service.LLMService.generate_structured") as mock_gen:
        mock_gen.side_effect = [
            ClassificationResult(
                category=TicketCategory.OTHER,
                priority=TicketPriority.MEDIUM,
                intent="general_question",
                sentiment="NEUTRAL",
                language="English",
                requires_human=False
            ),
            GeneratedResponse(
                response="The available ShopEase policies do not specify a ₹10,000 compensation for delays. I have flagged your request for human review.",
                language="English",
                used_knowledge_base=True,
                sufficient_information=False,  # Signal that KB lacks information!
                needs_human_review=True,
                reason="Policy does not provide ₹10,000 compensation."
            )
        ]
        resp = client.post("/api/v1/chat", json={"query": "Will ShopEase compensate me ₹10,000 for my delayed order?"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["sufficient_information"] is False
        assert data["needs_human_review"] is True
        assert data["status"] == "ESCALATED"


# -------------------------------------------------------------
# 7. Unknown / out-of-domain query
# -------------------------------------------------------------
def test_7_unknown_out_of_domain_query(client):
    """Test 7: Out-of-domain query produces no sufficient context and does not guess."""
    # When query is out of domain (distance > threshold), system escalates and signals insufficient info
    resp = client.post("/api/v1/chat", json={"query": "Can I adopt a blue penguin and feed it space candy?"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["needs_human_review"] is True
    assert data["status"] == "ESCALATED"


# -------------------------------------------------------------
# 8. Telugu response
# -------------------------------------------------------------
def test_8_telugu_response(client):
    """Test 8: Telugu query receives a response matching the Telugu language."""
    telugu_q = "నా ఆర్డర్ ఇంకా రాలేదు డెలివరీ ఎప్పుడు వస్తుంది?"
    resp = client.post("/api/v1/chat", json={"query": telugu_q})
    assert resp.status_code == 200
    data = resp.json()

    assert data["classification"]["language"] == "Telugu"
    assert data["classification"]["category"] == "SHIPPING"


# -------------------------------------------------------------
# 9. Hindi response
# -------------------------------------------------------------
def test_9_hindi_response(client):
    """Test 9: Hindi query receives a response matching the Hindi language."""
    hindi_q = "मुझे अपना सामान वापस करना है क्या नियम है?"
    resp = client.post("/api/v1/chat", json={"query": hindi_q})
    assert resp.status_code == 200
    data = resp.json()

    assert data["classification"]["language"] == "Hindi"
    assert data["classification"]["category"] == "RETURN_REFUND"


# -------------------------------------------------------------
# 10. Multi-turn conversation follow-up using previous history
# -------------------------------------------------------------
def test_10_conversation_followup_using_previous_history(client):
    """Test 10: Turn 2 resolves context ('it') from Turn 1 under the same ticket."""
    # Turn 1:
    turn1_resp = client.post(
        "/api/v1/chat",
        json={"query": "My order for shoes has not arrived yet."}
    )
    assert turn1_resp.status_code == 200
    ticket_id = turn1_resp.json()["ticket_id"]

    # Turn 2 using ticket_id with patched LLM:
    with patch("app.services.llm_service.LLMService.is_available", return_value=True), \
         patch("app.services.llm_service.LLMService.generate_structured") as mock_gen:
        mock_gen.side_effect = [
            GeneratedResponse(
                response="For your delayed shoes order, standard delivery takes 3 to 5 business days. If it has been more than 4 days without updates, we investigate with the courier.",
                language="English",
                used_knowledge_base=True,
                sufficient_information=True,
                needs_human_review=False
            )
        ]
        turn2_resp = client.post(
            "/api/v1/chat",
            json={
                "query": "How long should I wait for it?",
                "ticket_id": ticket_id
            }
        )
        assert turn2_resp.status_code == 200
        data = turn2_resp.json()
        assert data["ticket_id"] == ticket_id
        assert "shoes" in data["response"].lower() or "delayed" in data["response"].lower()


# -------------------------------------------------------------
# 11. Gemini API failure fallback
# -------------------------------------------------------------
def test_11_gemini_api_failure_fallback(client):
    """Test 11: When Gemini API raises an exception, system uses safe fallback without crashing."""
    with patch("app.services.llm_service.LLMService.is_available", return_value=True), \
         patch("app.services.llm_service.LLMService.generate_structured", side_effect=Exception("API connection timeout")):
        resp = client.post("/api/v1/chat", json={"query": "What is the return window for electronics?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] is not None
        assert len(data["response"]) > 0


# -------------------------------------------------------------
# 12. Empty/invalid model response fallback
# -------------------------------------------------------------
def test_12_empty_or_invalid_model_response(client):
    """Test 12: Empty response returned from LLM triggers safe fallback."""
    with patch("app.services.llm_service.LLMService.is_available", return_value=True), \
         patch("app.services.llm_service.LLMService.generate_structured", return_value=None):
        resp = client.post("/api/v1/chat", json={"query": "What are your customer support hours?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] is not None
        assert len(data["response"]) > 0


# -------------------------------------------------------------
# 13. Response is saved to SQLite
# -------------------------------------------------------------
def test_13_response_saved_to_sqlite(client, db_session):
    """Test 13: Verifies that both customer query and assistant response are saved in SQLite."""
    resp = client.post(
        "/api/v1/chat",
        json={"query": "I would like to know about coupon codes."}
    )
    assert resp.status_code == 200
    ticket_id = resp.json()["ticket_id"]

    # Verify messages in SQLite
    msgs = db_session.query(Message).filter(Message.ticket_id == ticket_id).order_by(Message.created_at).all()
    assert len(msgs) == 2
    assert msgs[0].sender == SenderType.USER
    assert msgs[0].content == "I would like to know about coupon codes."
    assert msgs[1].sender == SenderType.ASSISTANT
    assert len(msgs[1].content) > 0


# -------------------------------------------------------------
# 14. Assistant message is linked to correct ticket
# -------------------------------------------------------------
def test_14_assistant_message_linked_to_correct_ticket(client, db_session):
    """Test 14: Verifies foreign key relationship and ticket message back_populates."""
    resp = client.post(
        "/api/v1/chat",
        json={"query": "How do I change my shipping address?"}
    )
    assert resp.status_code == 200
    ticket_id = resp.json()["ticket_id"]

    ticket = db_session.query(Ticket).filter(Ticket.id == ticket_id).first()
    assert ticket is not None
    assert len(ticket.messages) == 2
    assert ticket.messages[1].ticket_id == ticket.id
    assert ticket.messages[1].ticket == ticket

# -------------------------------------------------------------
# 12. Payment-pending intent must not be confused with shipping
# -------------------------------------------------------------
def test_12_payment_pending_is_payment_not_shipping(client):
    resp = client.post("/api/v1/chat", json={"query": "My payment was successful but my order still says payment pending."})
    assert resp.status_code == 200
    data = resp.json()
    assert data["classification"]["category"] == "PAYMENT"
    assert data["classification"]["intent"] == "payment_status_pending"
    assert data["status"] == "ESCALATED"
    assert "pending" in data["response"].lower()


# -------------------------------------------------------------
# 13. Exact KB evidence is exposed to the UI
# -------------------------------------------------------------
def test_13_retrieved_evidence_is_returned(client):
    resp = client.post("/api/v1/chat", json={"query": "How long does express delivery take?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["retrieved_evidence"]
    assert data["retrieved_evidence"][0]["source"] == "shipping_and_delivery.md"
    assert "Express Shipping" in data["retrieved_evidence"][0]["content"]


# -------------------------------------------------------------
# 14. Follow-up keeps the previous topic
# -------------------------------------------------------------
def test_14_followup_context_is_persistent(client):
    first = client.post("/api/v1/chat", json={"query": "Can I get same-day delivery in Hyderabad?"})
    assert first.status_code == 200
    ticket_id = first.json()["ticket_id"]
    second = client.post("/api/v1/chat", json={"query": "How much does it cost?", "ticket_id": ticket_id})
    assert second.status_code == 200
    data = second.json()
    assert data["ticket_id"] == ticket_id
    assert data["classification"]["category"] == "SHIPPING"
    assert "149" in data["response"] or "149" in " ".join(x["content"] for x in data["retrieved_evidence"])


def test_15_same_day_followups_keep_topic_and_retrieve_relevant_evidence(client):
    """Follow-up questions inherit the same-day topic without leaking other delivery tiers."""
    first = client.post("/api/v1/chat", json={"query": "Can I get same-day delivery in Hyderabad?"})
    assert first.status_code == 200
    ticket_id = first.json()["ticket_id"]

    second = client.post("/api/v1/chat", json={"query": "How much does it cost?", "ticket_id": ticket_id})
    assert second.status_code == 200
    data = second.json()
    assert "149" in data["response"]
    assert data["retrieved_evidence"]
    assert data["retrieved_evidence"][0]["source"] == "shipping_and_delivery.md"

    third = client.post("/api/v1/chat", json={"query": "What is the cutoff time?", "ticket_id": ticket_id})
    assert third.status_code == 200
    data = third.json()
    assert "11:00" in data["response"]
    assert data["retrieved_evidence"][0]["source"] == "shipping_and_delivery.md"


def test_16_new_explicit_shipping_topic_does_not_inherit_payment_topic(client):
    """An explicit new topic must override the previous payment conversation."""
    first = client.post("/api/v1/chat", json={"query": "My payment is pending."})
    assert first.status_code == 200
    ticket_id = first.json()["ticket_id"]

    second = client.post("/api/v1/chat", json={
        "query": "Can I get same-day delivery in Hyderabad?",
        "ticket_id": ticket_id,
    })
    assert second.status_code == 200
    data = second.json()
    assert data["classification"]["category"] == "SHIPPING"
    assert data["classification"]["intent"] == "same_day_delivery"
    assert data["retrieved_evidence"][0]["source"] == "shipping_and_delivery.md"


def test_17_unsupported_five_percent_cashback_is_not_answered_from_unrelated_numbers(client):
    """5% must not match unrelated 5-to-7-day refund numbers in another document."""
    resp = client.post("/api/v1/chat", json={"query": "Do you offer 5% cashback on every order?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["sufficient_information"] is False
    assert data["needs_human_review"] is True
    assert "5%" not in data["response"] or "cannot confirm" in data["response"].lower() or "does not confirm" in data["response"].lower()


def test_chat_response_exposes_llm_grounding_trace(client):
    from unittest.mock import patch
    from app.schemas.chat import GeneratedResponse
    with patch("app.services.llm_service.LLMService.is_available", return_value=True), \
         patch("app.services.llm_service.LLMService.generate_structured") as mock_gen:
        mock_gen.return_value = GeneratedResponse(
            response="Express delivery takes 1 to 2 business days.",
            language="English", sufficient_information=True, used_knowledge_base=True
        )
        resp = client.post("/api/v1/chat", json={"query": "How long does express delivery take?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["llm_used"] is True
        assert data["generation_method"] == "gemini"
        assert data["grounding_verified"] is True
        assert data["retrieved_evidence"]
        prompt = mock_gen.call_args.kwargs["prompt"]
        assert "RETRIEVED POLICY CHUNK (AUTHORITATIVE)" in prompt
        assert "Express Shipping" in prompt
