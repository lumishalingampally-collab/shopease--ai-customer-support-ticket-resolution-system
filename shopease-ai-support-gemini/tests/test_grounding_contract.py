"""Contract tests for the anti-hallucination and traceability layer."""

from unittest.mock import Mock

from app.schemas.chat import ClassificationResult, GeneratedResponse, PolicyReference
from app.schemas.kb import KBSearchResult
from app.services.response_service import ResponseService


def chunk(content, source="shipping_and_delivery.md", topic="1. Shipping Tiers and Service Timelines", score=0.92):
    return KBSearchResult(
        content=content,
        source=source,
        category=source.rsplit(".", 1)[0],
        topic=topic,
        chunk_id="shipping_and_delivery_sec_1",
        distance=0.1,
        similarity_score=score,
        metadata={},
    )


def test_generated_answer_with_supported_numbers_passes_grounding():
    evidence = chunk("Express Shipping: Delivery window: 1 to 2 business days. Cost: Additional ₹99 express surcharge.")
    output = GeneratedResponse(
        response="Express Shipping takes 1–2 business days and has a ₹99 surcharge.\n\nKB reference: KB-SHIP-01",
        policy_references=[PolicyReference(reference_id="KB-SHIP-01", source=evidence.source, section=evidence.topic, relevance_score=0.92)],
    )
    assert ResponseService._validate_grounding(output, [evidence]) is True


def test_generated_answer_with_unsupported_number_is_rejected():
    evidence = chunk("Express Shipping has a ₹99 surcharge.")
    output = GeneratedResponse(
        response="Express Shipping has a ₹499 surcharge.",
        policy_references=[PolicyReference(reference_id="KB-SHIP-01", source=evidence.source, section=evidence.topic, relevance_score=0.92)],
    )
    assert ResponseService._validate_grounding(output, [evidence]) is False


def test_maximum_refund_question_is_not_supported_by_generic_refund_evidence():
    from app.services.rag_service import RAGService
    assert RAGService._specificity_gate(
        "What is the maximum refund amount ShopEase allows?",
        "Returns initiated due to defective items are free of charge. Standard accounts may incur a ₹49 reverse logistics fee.",
    ) is False


def test_five_percent_cashback_is_not_supported_without_the_number_in_evidence():
    from app.services.rag_service import RAGService
    assert RAGService._specificity_gate(
        "Does ShopEase offer 5% cashback?",
        "Payment gateway cashback campaigns are credited by the payment partner within 7 to 14 business days.",
    ) is False


def test_long_generated_answer_is_rejected():
    evidence = chunk("Express Shipping: Delivery window: 1 to 2 business days. Cost: Additional ₹99 express surcharge.")
    output = GeneratedResponse(
        response="Express Shipping is available. " + ("This is extra text. " * 100),
        policy_references=[PolicyReference(reference_id="KB-SHIP-01", source=evidence.source, section=evidence.topic, relevance_score=0.92)],
    )
    assert ResponseService._validate_grounding(output, [evidence]) is False
