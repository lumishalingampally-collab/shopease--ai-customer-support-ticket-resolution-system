"""Regression tests for issue analysis and semantic-RAG architecture."""
from pathlib import Path

from app.services.classifier_service import ClassifierService
from app.services.rag_service import RAGService
from app.models.ticket import TicketCategory


class _NoLLM:
    def is_available(self):
        return False


def _services():
    classifier = ClassifierService(llm_service=_NoLLM())
    rag = RAGService.__new__(RAGService)
    rag.kb_dir = Path(__file__).resolve().parents[1] / "knowledge_base"
    return classifier, rag


def test_generic_package_problem_requires_clarification():
    classifier, _ = _services()
    result = classifier.classify_query("I have a problem with my package")
    assert result.intent == "needs_clarification"
    assert result.category == TicketCategory.OTHER


def test_damaged_return_analysis_is_specific():
    classifier, _ = _services()
    result = classifier.classify_query("My package is totally damaged and I don't know how to put it return.")
    assert result.intent == "damaged_return_request"
    assert result.category == TicketCategory.RETURN_REFUND


def test_payment_pending_analysis_stays_in_payment_category():
    classifier, _ = _services()
    result = classifier.classify_query("My payment was successful but my order still says payment pending.")
    assert result.intent == "payment_status_pending"
    assert result.category == TicketCategory.PAYMENT


def test_order_tracking_analysis_is_shipping():
    classifier, _ = _services()
    result = classifier.classify_query("I am not understanding where my order is.")
    assert result.intent == "order_tracking"
    assert result.category == TicketCategory.SHIPPING


def test_cancellation_analysis_is_order():
    classifier, _ = _services()
    result = classifier.classify_query("I paid for a product but I don't want that now.")
    assert result.intent == "order_cancellation"
    assert result.category == TicketCategory.ORDER


def test_cod_mismatch_is_escalated_without_inventing_a_workflow():
    classifier, _ = _services()
    query = "I paid during order placement but the delivery agent is asking me for COD cash."
    result = classifier.classify_query(query)
    assert result.intent == "cod_payment_mismatch"
    assert result.requires_human is True


def test_replacement_status_is_escalated_when_kb_lacks_tracking_workflow():
    classifier, _ = _services()
    query = "I kept an order replacement but did not receive it till now."
    result = classifier.classify_query(query)
    assert result.intent == "replacement_status"
    assert result.requires_human is True


def test_rag_query_builder_uses_analysis_as_semantic_hint_not_answer():
    rag = RAGService.__new__(RAGService)
    from app.schemas.chat import ClassificationResult
    result = ClassificationResult(category=TicketCategory.PAYMENT, intent="duplicate_charge")
    query = rag._query_for_retrieval("My card was billed two times", result)
    assert "My card was billed two times" in query
    assert "support category payment" in query.lower()
    assert "customer intent duplicate charge" in query.lower()
