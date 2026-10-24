"""Comprehensive unit and integration tests for AI Query Classification and Escalation."""

from unittest.mock import MagicMock
import pytest

from app.models.ticket import TicketCategory, TicketPriority, SentimentType
from app.schemas.chat import ClassificationResult
from app.services.classifier_service import ClassifierService
from app.services.llm_service import LLMService


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service for testing deterministic and LLM flows."""
    mock = MagicMock(spec=LLMService)
    mock.is_available.return_value = True
    return mock


@pytest.fixture
def classifier(mock_llm_service):
    """Classifier service with mocked LLM dependency."""
    return ClassifierService(llm_service=mock_llm_service)


# -------------------------------------------------------------
# Test Cases (1 to 12)
# -------------------------------------------------------------

def test_1_failed_payment(classifier, mock_llm_service):
    """Test 1: Failed payment query."""
    mock_llm_service.generate_structured.return_value = ClassificationResult(
        category=TicketCategory.PAYMENT,
        priority=TicketPriority.MEDIUM,
        intent="payment_failed",
        sentiment=SentimentType.NEGATIVE,
        language="English",
        requires_human=False,
        escalation_reason=None
    )
    result = classifier.classify_query("My UPI payment failed but money was deducted from my account.")
    assert result.category == TicketCategory.PAYMENT
    assert result.intent == "payment_failed"
    assert result.priority in [TicketPriority.MEDIUM, TicketPriority.HIGH]


def test_2_duplicate_payment(classifier, mock_llm_service):
    """Test 2: Duplicate payment triggers HIGH priority and human escalation."""
    mock_llm_service.generate_structured.return_value = ClassificationResult(
        category=TicketCategory.PAYMENT,
        priority=TicketPriority.HIGH,
        intent="duplicate_charge",
        sentiment=SentimentType.FRUSTRATED,
        language="English",
        requires_human=True,
        escalation_reason="Duplicate payment requires human review"
    )
    result = classifier.classify_query("My card was charged twice for the same order and I want my money back.")
    assert result.category == TicketCategory.PAYMENT
    assert result.intent == "duplicate_charge"
    assert result.requires_human is True
    assert result.priority in [TicketPriority.HIGH, TicketPriority.URGENT]
    assert "duplicate" in result.escalation_reason.lower()


def test_3_unauthorized_transaction(classifier, mock_llm_service):
    """Test 3: Unauthorized transaction / fraud triggers URGENT and human escalation."""
    mock_llm_service.generate_structured.return_value = ClassificationResult(
        category=TicketCategory.PAYMENT,
        priority=TicketPriority.URGENT,
        intent="unauthorized_transaction",
        sentiment=SentimentType.FRUSTRATED,
        language="English",
        requires_human=True,
        escalation_reason="Suspected card fraud"
    )
    result = classifier.classify_query("Someone placed an unauthorized transaction on my card! My card is stolen.")
    assert result.category == TicketCategory.PAYMENT
    assert result.priority == TicketPriority.URGENT
    assert result.requires_human is True
    assert result.intent == "unauthorized_transaction"
    assert "unauthorized" in result.escalation_reason.lower() or "fraud" in result.escalation_reason.lower()


def test_4_refund_request(classifier, mock_llm_service):
    """Test 4: Standard return and refund request."""
    mock_llm_service.generate_structured.return_value = ClassificationResult(
        category=TicketCategory.RETURN_REFUND,
        priority=TicketPriority.MEDIUM,
        intent="return_request",
        sentiment=SentimentType.NEUTRAL,
        language="English",
        requires_human=False,
        escalation_reason=None
    )
    result = classifier.classify_query("The dress size is too small, I would like to initiate a return.")
    assert result.category == TicketCategory.RETURN_REFUND
    assert result.intent == "return_request"
    assert result.requires_human is False


def test_5_shipping_delay(classifier, mock_llm_service):
    """Test 5: Shipping tracking and courier delay inquiry."""
    mock_llm_service.generate_structured.return_value = ClassificationResult(
        category=TicketCategory.SHIPPING,
        priority=TicketPriority.MEDIUM,
        intent="delayed_delivery",
        sentiment=SentimentType.NEGATIVE,
        language="English",
        requires_human=False,
        escalation_reason=None
    )
    result = classifier.classify_query("My courier has been delayed for 2 days. Where is my tracking update?")
    assert result.category == TicketCategory.SHIPPING
    assert result.intent in ["delayed_delivery", "order_tracking"]


def test_6_password_reset(classifier, mock_llm_service):
    """Test 6: Account security and password reset."""
    mock_llm_service.generate_structured.return_value = ClassificationResult(
        category=TicketCategory.ACCOUNT,
        priority=TicketPriority.MEDIUM,
        intent="password_reset",
        sentiment=SentimentType.NEUTRAL,
        language="English",
        requires_human=False,
        escalation_reason=None
    )
    result = classifier.classify_query("I forgot my login password and the reset OTP is not arriving.")
    assert result.category == TicketCategory.ACCOUNT
    assert result.intent in ["password_reset", "account_security"]


def test_7_telugu_query(classifier, mock_llm_service):
    """Test 7: Support query submitted in Telugu."""
    telugu_text = "నా ఆర్డర్ ఇంకా రాలేదు డెలివరీ చాలా ఆలస్యం అయింది"
    mock_llm_service.generate_structured.return_value = ClassificationResult(
        category=TicketCategory.SHIPPING,
        priority=TicketPriority.MEDIUM,
        intent="delayed_delivery",
        sentiment=SentimentType.NEGATIVE,
        language="Telugu",
        requires_human=False,
        escalation_reason=None
    )
    result = classifier.classify_query(telugu_text)
    assert result.language == "Telugu"
    assert result.category == TicketCategory.SHIPPING


def test_8_hindi_query(classifier, mock_llm_service):
    """Test 8: Support query submitted in Hindi."""
    hindi_text = "मेरे खाते से दो बार पैसे कट गए हैं कृपया वापस करें"
    mock_llm_service.generate_structured.return_value = ClassificationResult(
        category=TicketCategory.PAYMENT,
        priority=TicketPriority.HIGH,
        intent="duplicate_charge",
        sentiment=SentimentType.FRUSTRATED,
        language="Hindi",
        requires_human=True,
        escalation_reason="Duplicate payment"
    )
    result = classifier.classify_query(hindi_text)
    assert result.language == "Hindi"
    assert result.category == TicketCategory.PAYMENT
    assert result.requires_human is True


def test_9_explicit_talk_to_human(classifier, mock_llm_service):
    """Test 9: Explicit request to speak with a human agent."""
    mock_llm_service.generate_structured.return_value = ClassificationResult(
        category=TicketCategory.OTHER,
        priority=TicketPriority.HIGH,
        intent="talk_to_agent",
        sentiment=SentimentType.NEUTRAL,
        language="English",
        requires_human=True,
        escalation_reason="Customer requested a human agent"
    )
    result = classifier.classify_query("I do not want to talk to a bot, please connect me to a human agent immediately.")
    assert result.requires_human is True
    assert result.intent == "talk_to_agent"
    assert "human" in result.escalation_reason.lower() or "agent" in result.escalation_reason.lower()


def test_10_unknown_unclear_query(classifier, mock_llm_service):
    """Test 10: Gibberish or incomprehensible input escalates for human clarity."""
    mock_llm_service.generate_structured.return_value = ClassificationResult(
        category=TicketCategory.OTHER,
        priority=TicketPriority.LOW,
        intent="unknown",
        sentiment=SentimentType.NEUTRAL,
        language="English",
        requires_human=True,
        escalation_reason="Query unclear"
    )
    result = classifier.classify_query("?? !!! zzzzqqqq 12389")
    assert result.category == TicketCategory.OTHER
    assert result.requires_human is True
    assert result.intent == "unknown"


def test_11_gemini_api_failure_fallback(mock_llm_service):
    """Test 11: When Gemini API raises an exception, the system uses safe heuristic fallback and does not crash."""
    mock_llm_service.is_available.return_value = True
    mock_llm_service.generate_structured.side_effect = RuntimeError("Gemini API connection timed out!")

    classifier = ClassifierService(llm_service=mock_llm_service)
    # Query should still be classified cleanly without throwing an exception
    result = classifier.classify_query("My payment was charged twice for order 8812")

    assert result is not None
    assert result.category == TicketCategory.PAYMENT
    assert result.intent == "duplicate_charge"
    assert result.requires_human is True
    assert result.priority in [TicketPriority.HIGH, TicketPriority.URGENT]


def test_12_deterministic_escalation_overrides_unsafe_llm(classifier, mock_llm_service):
    """Test 12: Deterministic rules override an unsafe/hallucinatory LLM response that missed escalation."""
    # Suppose LLM erroneously thought a legal notice / unauthorized card charge didn't need a human
    mock_llm_service.generate_structured.return_value = ClassificationResult(
        category=TicketCategory.OTHER,
        priority=TicketPriority.LOW,
        intent="general_question",
        sentiment=SentimentType.NEUTRAL,
        language="English",
        requires_human=False,  # Unsafe LLM output!
        escalation_reason=None
    )

    query = "You stole my money and I am filing a consumer court case against ShopEase!"
    result = classifier.classify_query(query)

    # Deterministic rule MUST override the LLM's unsafe output
    assert result.requires_human is True
    assert result.priority == TicketPriority.URGENT
    assert "legal" in result.escalation_reason.lower() or "court" in result.escalation_reason.lower()


# -------------------------------------------------------------
# Integration Test for API Endpoint
# -------------------------------------------------------------

def test_api_classify_endpoint(client):
    """Test POST /api/v1/classify REST endpoint."""
    payload = {"query": "I was charged twice for order #54321"}
    response = client.post("/api/v1/classify", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "PAYMENT"
    assert data["intent"] == "duplicate_charge"
    assert data["requires_human"] is True
    assert data["priority"] in ["HIGH", "URGENT"]
    assert "language" in data
