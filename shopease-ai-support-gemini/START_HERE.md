# ShopEase — Start Here

## Run the finished project

Open this folder in VS Code.

### Terminal 1 — FastAPI

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

If your local `.venv` is not inside this folder, use the Python executable from your already-working ShopEase environment.

### Terminal 2 — Streamlit UI

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Open the URL Streamlit prints, normally `http://localhost:8501`.

## Run tests

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected target: all tests pass.

## Demo flow

1. Open **Customer Support**.
2. Try: `My card was charged twice for the same order.`
3. Observe category, priority, sentiment, sources, and human escalation.
4. Try a normal policy question such as `How long does standard delivery take?`.
5. Try an unsupported question such as `Can I adopt an Antarctic penguin?` and verify that the system does not invent a ShopEase policy.
6. Try a Telugu or Hindi support query.
7. Open **Agent Dashboard** to inspect and resolve/escalate tickets.
8. Open **Knowledge Center** to inspect the policy corpus and test retrieval.
9. Refresh/reopen Streamlit, choose a ticket under **Saved conversations**, and click **Resume selected ticket** to demonstrate database-backed conversation history.
10. Open **Agent Dashboard** to show that tickets remain stored independently of the UI session.


## Final RAG demo checklist

1. Create `.env` from `.env.example` and add your own API keys. Never commit or share `.env`.
2. Start FastAPI on port 8000.
3. Start Streamlit in a second terminal.
4. Test: `How long does standard delivery take?` → `shipping_and_delivery.md` and 3–5 business days.
5. Test: `How long does express delivery take?` → `shipping_and_delivery.md` and 1–2 business days.
6. Test: `Do you offer same-day delivery?` → `shipping_and_delivery.md`.
7. Test: `I was charged twice for my order.` → payment KB + human escalation.
8. Test: `What is the capital of France?` → no KB answer / human review, no hallucination.
9. Test: `Hi` → friendly greeting, no escalation.
10. Test a follow-up such as `My package is delayed` → `When should I contact support?`.

## Final interview flow

For the strongest demonstration, use one ticket and ask:

1. `Can I get same-day delivery in Hyderabad?`
2. `How much does it cost?`
3. `What is the cutoff time?`

Then start a separate/new topic and ask:

4. `My payment was successful but my order still says payment pending.`
5. `My card was charged twice for the same order.`

Show the AI Analysis panel and Retrieved KB Evidence panel after each answer. This demonstrates classification, priority, intent, sentiment, human escalation, RAG traceability, and conversation context.

## Final regression scenarios

These cases are intentionally included because they protect against common semantic-RAG false positives:

- `I have a problem with my package` → asks what happened instead of selecting a random shipping policy.
- `My package is totally damaged and I don't know how to put it return.` → Return Request and Pickup Procedure / damaged-return intent.
- `I am not understanding where my order is.` → Order Tracking and Status Definitions.
- `I paid for a product but I don't want that now.` → Order Cancellation Window and Eligibility.
- `I paid during order placement but the delivery agent is asking me for COD cash.` → insufficient KB evidence; no invented policy; human review.
- `I kept an order replacement but did not receive it till now.` → insufficient KB evidence for replacement-status tracking; human review.
- `My payment was successful but my order still says payment pending.` → PAYMENT / payment_status_pending; never treated as shipping.
- `Can I get same-day delivery in Hyderabad?` → Same-Day Priority Delivery evidence.
- In that same conversation: `How much does it cost?` → ₹149 same-day surcharge.
- In that same conversation: `What is the cutoff time?` → 11:00 AM IST.
- A new explicit topic after a payment question must not inherit the payment topic.

The UI should show the retrieved source, section, evidence lines, analysis fields, and ticket number after each query.
