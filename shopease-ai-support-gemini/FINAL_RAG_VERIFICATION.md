# Final RAG Verification

The support pipeline uses the Markdown knowledge base as the source of truth. ChromaDB remains the semantic retrieval layer; a policy-aware topic matcher is applied before semantic fallback so common support intents select the exact KB section. Gemini receives the selected evidence and is instructed to format only supported facts.

## Critical retrieval cases
- Standard / regular delivery -> Shipping Tiers and Service Timelines
- Express / fastest delivery -> Shipping Tiers and Service Timelines
- Same-day / delivery today -> Shipping Tiers and Service Timelines
- Return policy / return window -> Return Window and Eligibility Overview
- Card / UPI refund timing -> Inspection and Refund Timelines
- Duplicate / double charge -> Duplicate Charges and Double Deductions
- Delayed package -> Delayed Deliveries and Escalation Protocol
- Delivered but not received -> Package Marked Delivered but Not Received
- Cashback / bank offer -> Bank and Payment Partner Offers
- Coupon -> Applying Discount Promo Codes

## Grounding behavior
If a requested concrete fact is not supported by the retrieved evidence, the system refuses to invent it and flags the case for human review. Comparative return questions such as a return request after 90 days are answered from the documented return window rather than requiring the queried number itself to appear in the KB.

## Persistence
Tickets and every user/assistant message are stored in SQLite. A ticket can be retrieved by ID or ticket number, and the Streamlit UI provides a Saved Conversations control to resume existing tickets.
