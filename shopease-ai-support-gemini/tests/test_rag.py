"""Comprehensive unit and integration tests for Knowledge Base and RAG retrieval layer."""

import pytest
from app.services.rag_service import get_rag_service


@pytest.fixture(scope="module")
def rag_service():
    """Shared RAG service fixture with pre-indexed knowledge base."""
    service = get_rag_service()
    # Force reindex to ensure clean state
    service.index_documents(force_reindex=True)
    return service


def test_indexing_knowledge_base_documents(rag_service):
    """Test 1: Verify indexing loads all policy documents and splits them into chunks."""
    status = rag_service.get_status()
    assert status["is_indexed"] is True
    assert status["total_chunks"] >= 30, f"Expected at least 30 chunks, got {status['total_chunks']}"
    assert status["collection_name"] == "shopease_knowledge_base"


def test_retrieve_payment_query(rag_service):
    """Test 2: Retrieve policy for duplicate charges and payment disputes."""
    query = "My account was debited twice for order 12345"
    response = rag_service.search(query=query, top_k=3)

    assert response.has_sufficient_context is True
    assert response.results_count > 0

    top_result = response.results[0]
    # Check that payment or dispute chunks were prioritized
    assert any(
        kw in top_result.content.lower()
        for kw in ["duplicate", "charge", "debit", "payment", "dispute"]
    )
    assert top_result.category in ["payment_and_billing", "human_escalation_policy"]
    assert top_result.similarity_score >= 0.60


def test_retrieve_refund_query(rag_service):
    """Test 3: Retrieve refund policy and return window timelines."""
    query = "What is the return window for electronics and clothing?"
    response = rag_service.search(query=query, top_k=3)

    assert response.has_sufficient_context is True
    assert response.results_count > 0

    top_sources = [r.source for r in response.results]
    assert "returns_and_refunds.md" in top_sources

    # Check for specific return numbers from policy (30 days standard, 14 days electronics)
    found_timeline = any("30" in r.content or "14" in r.content for r in response.results)
    assert found_timeline is True


def test_retrieve_shipping_query(rag_service):
    """Test 4: Retrieve shipping tiers and express delivery time."""
    query = "How long does standard delivery take and is it free over 499?"
    response = rag_service.search(query=query, top_k=3)

    assert response.has_sufficient_context is True
    assert response.results_count > 0

    top_sources = [r.source for r in response.results]
    assert "shipping_and_delivery.md" in top_sources

    # Verify standard delivery details (3 to 5 business days, 499 threshold)
    found_details = any("3 to 5" in r.content or "499" in r.content for r in response.results)
    assert found_details is True


def test_unrelated_query_produces_no_sufficient_context(rag_service):
    """Test 5: An unrelated out-of-domain query produces no valid results when threshold is enforced."""
    unrelated_query = "Can I adopt an Antarctic penguin and feed it strawberry ice cream in space?"

    # Use strict threshold to verify out-of-domain rejection
    response = rag_service.search(query=unrelated_query, top_k=3, threshold=0.85)

    assert response.results_count == 0
    assert response.has_sufficient_context is False


def test_metadata_returned_correctly(rag_service):
    """Test 6: Verify all required metadata fields are properly populated on retrieved results."""
    query = "How do I report damaged goods or wrong item received?"
    response = rag_service.search(query=query, top_k=2)

    assert response.has_sufficient_context is True
    for item in response.results:
        assert item.source.endswith(".md")
        assert len(item.category) > 0
        assert len(item.topic) > 0
        assert len(item.chunk_id) > 0
        assert 0.0 <= item.similarity_score <= 1.0
        assert isinstance(item.metadata, dict)
        assert "doc_title" in item.metadata


def test_api_reindex_and_search_endpoints(client):
    """Test 7: Test the FastAPI KB REST endpoints (/kb/status, /kb/reindex, /kb/search)."""
    # Test GET /api/v1/kb/status
    status_resp = client.get("/api/v1/kb/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert "is_indexed" in status_data
    assert "total_chunks" in status_data

    # Test POST /api/v1/kb/search
    search_payload = {
        "query": "How can I reset my account password?",
        "top_k": 3
    }
    search_resp = client.post("/api/v1/kb/search", json=search_payload)
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert search_data["results_count"] > 0
    assert search_data["has_sufficient_context"] is True
    assert any("password" in r["content"].lower() for r in search_data["results"])


def test_exact_standard_delivery_prefers_shipping_tier(rag_service):
    """Regression: exact standard-delivery question must rank shipping tiers first."""
    response = rag_service.search("How long does standard delivery take?", top_k=3)
    assert response.has_sufficient_context is True
    assert response.results[0].source == "shipping_and_delivery.md"
    assert "3 to 5 business days" in response.results[0].content


def test_exact_express_delivery_prefers_shipping_tier(rag_service):
    """Regression: express-delivery timing must rank the shipping-tier section first."""
    response = rag_service.search("How long does express delivery take?", top_k=3)
    assert response.has_sufficient_context is True
    assert response.results[0].source == "shipping_and_delivery.md"
    assert "1 to 2 business days" in response.results[0].content


def test_same_day_delivery_prefers_shipping_tier(rag_service):
    """Regression: same-day delivery availability must retrieve shipping tiers."""
    response = rag_service.search("Do you offer same-day delivery?", top_k=3)
    assert response.has_sufficient_context is True
    assert response.results[0].source == "shipping_and_delivery.md"
    assert "Same-Day Priority Delivery" in response.results[0].content


def test_greeting_does_not_retrieve_policy(rag_service):
    """Greetings are conversational and should never be treated as a KB gap."""
    response = rag_service.search("Hi", top_k=3)
    assert response.results_count == 0
    assert response.has_sufficient_context is False


def _assert_topic(rag_service, query, expected_section, expected_phrase=None):
    response = rag_service.search(query, top_k=3)
    assert response.has_sufficient_context is True, f"No context for: {query}"
    assert response.results_count >= 1
    top = response.results[0]
    assert expected_section.lower() in top.topic.lower(), (
        f"Wrong top section for {query!r}: {top.topic!r}"
    )
    if expected_phrase:
        assert expected_phrase.lower() in top.content.lower(), (
            f"Expected evidence {expected_phrase!r} not found for {query!r}"
        )


@pytest.mark.parametrize("query,section,phrase", [
    ("How long does standard delivery take?", "Shipping Tiers and Service Timelines", "3 to 5 business days"),
    ("How long does express delivery take?", "Shipping Tiers and Service Timelines", "1 to 2 business days"),
    ("Do you offer same-day delivery?", "Shipping Tiers and Service Timelines", "11:00 AM IST"),
    ("What is your return policy?", "Return Window and Eligibility Overview", "30 calendar days"),
    ("What is the return window for electronics?", "Return Window and Eligibility Overview", "14-calendar-day"),
    ("How long will my refund take if I paid by card?", "Inspection and Refund Timelines", "5 to 7 business days"),
    ("What is the refund time for UPI?", "Inspection and Refund Timelines", "24 to 48 business hours"),
    ("My card was charged twice for the same order.", "Duplicate Charges and Double Deductions", "HIGH PRIORITY / HUMAN ESCALATION"),
    ("My tracking hasn't moved for four consecutive business days.", "Delayed Deliveries and Escalation Protocol", "Stalled / Potential Lost in Transit"),
    ("My order says delivered but I didn't receive it.", "Package Marked Delivered but Not Received", "4 hours"),
    ("Do you offer cashback?", "Bank and Payment Partner Offers", "7 to 14 business days"),
    ("How do I apply a coupon?", "Applying Discount Promo Codes", "Apply Coupon"),
])
def test_common_customer_questions_retrieve_exact_policy_section(rag_service, query, section, phrase):
    _assert_topic(rag_service, query, section, phrase)


def test_return_shipping_fee_retrieves_return_charges(rag_service):
    _assert_topic(
        rag_service,
        "Do I have to pay a return shipping fee?",
        "Return Shipping Charges",
        "₹49"
    )


def test_cancel_order_retrieves_cancellation_window(rag_service):
    _assert_topic(
        rag_service,
        "Can I cancel my order?",
        "Order Cancellation Window and Eligibility",
        "2 hours"
    )


def test_cancellation_refund_retrieves_cancellation_refund_section(rag_service):
    _assert_topic(
        rag_service,
        "How long does a cancellation refund take?",
        "Cancellation Refund Processing",
        "2 to 5 business days"
    )


def test_follow_up_express_question_can_beat_previous_standard_topic(rag_service):
    query = "How long does standard delivery take?\nFollow-up customer question: What about express delivery?"
    response = rag_service.search(query, top_k=3)
    assert response.has_sufficient_context is True
    assert "Shipping Tiers and Service Timelines" in response.results[0].topic
    assert "express shipping" in response.results[0].content.lower()
    assert "1 to 2 business days" in response.results[0].content


def test_unsupported_maximum_refund_has_no_context(rag_service):
    response = rag_service.search("What is the maximum refund amount I can receive?", top_k=3)
    assert response.has_sufficient_context is False
    assert response.results_count == 0


def test_unsupported_five_percent_cashback_has_no_context(rag_service):
    response = rag_service.search("Do you offer 5% cashback on every order?", top_k=3)
    assert response.has_sufficient_context is False
    assert response.results_count == 0


def test_regular_shipment_maps_to_standard_delivery(rag_service):
    response = rag_service.search("When should I expect my regular shipment?", top_k=3)
    assert response.has_sufficient_context is True
    assert response.results[0].topic == "Shipping Tiers and Service Timelines"
    assert "3 to 5 business days" in response.results[0].content


def test_fastest_delivery_maps_to_same_day_tier(rag_service):
    response = rag_service.search("What is the fastest delivery option you have?", top_k=3)
    assert response.has_sufficient_context is True
    assert response.results[0].topic == "Shipping Tiers and Service Timelines"
    assert "Same-Day Priority Delivery" in response.results[0].content


def test_delivery_today_maps_to_same_day_tier(rag_service):
    response = rag_service.search("Can I get my order delivered today?", top_k=3)
    assert response.has_sufficient_context is True
    assert response.results[0].topic == "Shipping Tiers and Service Timelines"
    assert "11:00 AM IST" in response.results[0].content


def test_billed_two_times_maps_to_duplicate_charge(rag_service):
    response = rag_service.search("I was billed two times for one order.", top_k=3)
    assert response.has_sufficient_context is True
    assert response.results[0].topic == "Duplicate Charges and Double Deductions (High Priority)"


def test_delayed_package_maps_to_delay_policy(rag_service):
    response = rag_service.search("My package is delayed. What should I do?", top_k=3)
    assert response.has_sufficient_context is True
    assert response.results[0].topic == "Delayed Deliveries and Escalation Protocol"


def test_delivered_nothing_reached_maps_to_nonreceipt(rag_service):
    response = rag_service.search("Tracking says delivered but nothing reached me.", top_k=3)
    assert response.has_sufficient_context is True
    assert response.results[0].topic == "Package Marked Delivered but Not Received"


def test_return_after_90_days_uses_return_policy(rag_service):
    response = rag_service.search("Can I return a product after 90 days?", top_k=3)
    assert response.has_sufficient_context is True
    assert response.results[0].topic == "Return Window and Eligibility Overview"
    assert "30 calendar days" in response.results[0].content


def test_five_percent_cashback_is_not_supported(rag_service):
    response = rag_service.search("Do you offer 5% cashback on every order?", top_k=3)
    assert response.has_sufficient_context is False
    assert response.results_count == 0
