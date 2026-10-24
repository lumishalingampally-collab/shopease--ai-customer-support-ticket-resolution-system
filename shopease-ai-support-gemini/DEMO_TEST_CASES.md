# Interview Demo Test Cases

Run the backend and frontend, then use these in the Customer Support tab.

| Query | Expected analysis | Expected KB | Human? |
|---|---|---|---|
| How long does express delivery take? | SHIPPING / MEDIUM / express_delivery | shipping_and_delivery.md → Shipping Tiers and Service Timelines | No |
| Can I get my order delivered today? | SHIPPING / MEDIUM / same_day_delivery | shipping_and_delivery.md → Shipping Tiers and Service Timelines | No |
| What is the return window for electronics? | RETURN_REFUND / MEDIUM / return_request | returns_and_refunds.md → Return Window and Eligibility Overview | No |
| How long will my card refund take? | RETURN_REFUND / MEDIUM / refund_status | returns_and_refunds.md → Inspection and Refund Timelines | No |
| My card was charged twice for the same order. | PAYMENT / HIGH / duplicate_charge | payment_and_billing.md → Duplicate Charges and Double Deductions | **Yes** |
| My tracking hasn't moved for four consecutive business days. | SHIPPING / HIGH / delayed_delivery | shipping_and_delivery.md → Delayed Deliveries and Escalation Protocol | **Yes** |
| My payment was successful but my order still says payment pending. | PAYMENT / MEDIUM / payment_status_pending | payment_and_billing.md → Failed Payments and Automated Reversals | **Yes: KB gap** |
| Do you offer 5% cashback on every order? | PROMOTION | No sufficient evidence | **Yes** |
| My tracking says delivered but I haven't received my package. | SHIPPING / HIGH / delivered_not_received | shipping_and_delivery.md → Package Marked Delivered but Not Received | **Policy-driven if unresolved** |

## Multi-turn demo

1. `Can I get same-day delivery in Hyderabad?`
2. `How much does it cost?`
3. `And what is the cutoff?`

The second and third questions are sent with the saved ticket ID. The backend loads prior messages from SQLite and uses them as conversation context for classification, semantic retrieval, and Gemini response generation.

## Persistence demo

1. Ask 2–3 questions.
2. Note the ticket number shown in the UI.
3. Close Streamlit and FastAPI.
4. Start them again.
5. Choose the ticket under **Saved conversations → Resume selected ticket**.
6. The stored user/assistant messages and ticket analysis remain available.

## Recommended live demo sequence

1. `How long does express delivery take?` → Express Shipping → 1–2 business days + ₹99.
2. `Can I get same-day delivery in Hyderabad?` → Hyderabad eligibility + 11:00 AM IST cutoff + ₹149.
3. `How much does it cost?` → ₹149 because the previous topic is same-day delivery.
4. `What is the cutoff time?` → 11:00 AM IST because the previous topic is still same-day delivery.
5. `My payment was successful but my order still says payment pending.` → PAYMENT/payment_status_pending, no invented workflow, human review because the KB has no separate pending workflow.
6. `My card was charged twice for the same order.` → PAYMENT/duplicate_charge/HIGH/human.
7. `Do you offer 5% cashback on every order?` → insufficient KB evidence; never invent a 5% offer.
8. Resume the ticket after restarting the app to demonstrate persistent SQLite history.
