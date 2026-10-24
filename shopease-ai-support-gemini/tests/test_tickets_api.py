"""Integration tests for FastAPI Ticket endpoints."""

import pytest
from app.models.ticket import TicketStatus, TicketPriority, TicketCategory


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_create_ticket_minimal(client):
    """Test creating a ticket with minimal fields."""
    payload = {
        "customer_name": "Rohan Gupta",
        "customer_email": "rohan@example.com"
    }
    response = client.post("/api/v1/tickets", json=payload)
    assert response.status_code == 201
    data = response.json()

    assert data["id"] is not None
    assert data["ticket_number"].startswith("TKT-")
    assert data["customer_name"] == "Rohan Gupta"
    assert data["status"] == "OPEN"
    assert data["priority"] == "MEDIUM"
    assert data["category"] in ["GENERAL", "OTHER"]


def test_create_ticket_with_initial_message(client):
    """Test creating a ticket along with an initial customer message."""
    payload = {
        "customer_name": "Deepa Rao",
        "customer_email": "deepa@example.com",
        "category": "RETURN_REFUND",
        "priority": "HIGH",
        "initial_message": "Item arrived broken, need a return.",
        "detected_language": "en"
    }
    create_resp = client.post("/api/v1/tickets", json=payload)
    assert create_resp.status_code == 201
    ticket_id = create_resp.json()["id"]

    # Verify message was logged by getting ticket detail
    get_resp = client.get(f"/api/v1/tickets/{ticket_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert len(data["messages"]) == 1
    assert data["messages"][0]["content"] == "Item arrived broken, need a return."
    assert data["messages"][0]["sender"] == "USER"


def test_list_tickets_and_filtering(client):
    """Test listing tickets with status and priority filtering."""
    # Create 3 tickets with different statuses and priorities
    client.post("/api/v1/tickets", json={"customer_name": "User 1", "priority": "LOW", "category": "SHIPPING"})
    client.post("/api/v1/tickets", json={"customer_name": "User 2", "priority": "URGENT", "category": "PAYMENT"})
    client.post("/api/v1/tickets", json={"customer_name": "User 3", "priority": "LOW", "category": "SHIPPING"})

    # Fetch all
    all_resp = client.get("/api/v1/tickets")
    assert all_resp.status_code == 200
    assert len(all_resp.json()) == 3

    # Filter by priority
    urgent_resp = client.get("/api/v1/tickets?priority=URGENT")
    assert urgent_resp.status_code == 200
    assert len(urgent_resp.json()) == 1
    assert urgent_resp.json()[0]["customer_name"] == "User 2"

    # Filter by category
    shipping_resp = client.get("/api/v1/tickets?category=SHIPPING")
    assert shipping_resp.status_code == 200
    assert len(shipping_resp.json()) == 2


def test_get_ticket_not_found(client):
    """Test 404 response for invalid ticket ID."""
    response = client.get("/api/v1/tickets/non-existent-uuid")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_update_ticket_status(client):
    """Test updating ticket status and escalation flags."""
    create_resp = client.post(
        "/api/v1/tickets",
        json={"customer_name": "Ananya Sen", "category": "PAYMENT"}
    )
    ticket_id = create_resp.json()["id"]

    # Update ticket to ESCALATED with reason
    patch_payload = {
        "status": "ESCALATED",
        "priority": "URGENT",
        "requires_human": True,
        "escalation_reason": "Customer reported unauthorized card deduction"
    }
    patch_resp = client.patch(f"/api/v1/tickets/{ticket_id}", json=patch_payload)
    assert patch_resp.status_code == 200
    data = patch_resp.json()
    assert data["status"] == "ESCALATED"
    assert data["priority"] == "URGENT"
    assert data["requires_human"] is True
    assert data["escalation_reason"] == "Customer reported unauthorized card deduction"

    # Update ticket to RESOLVED
    resolve_resp = client.patch(f"/api/v1/tickets/{ticket_id}", json={"status": "RESOLVED"})
    assert resolve_resp.status_code == 200
    assert resolve_resp.json()["status"] == "RESOLVED"


def test_ticket_message_api_persists_messages(client):
    create_resp = client.post("/api/v1/tickets", json={"customer_name": "API Message Test"})
    assert create_resp.status_code == 201
    ticket_id = create_resp.json()["id"]
    post_resp = client.post(f"/api/v1/tickets/{ticket_id}/messages", json={
        "sender": "USER", "content": "How long does express delivery take?", "language": "English"
    })
    assert post_resp.status_code == 201
    get_resp = client.get(f"/api/v1/tickets/{ticket_id}/messages")
    assert get_resp.status_code == 200
    assert get_resp.json()[0]["content"] == "How long does express delivery take?"
