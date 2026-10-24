# ShopEase Human Escalation and Specialist Handoff Policy

## 1. Principles of Human Intervention
The ShopEase automated AI support system is designed to provide rapid, accurate, and policy-grounded assistance. However, complex disputes, sensitive financial matters, and safety incidents must be transitioned directly to human customer service specialists without delay.

## 2. Mandatory Human Escalation Triggers
A support ticket MUST be immediately escalated to a human specialist under the following specific conditions:

### A. Financial and Payment Disputes (Priority: URGENT / HIGH)
- **Duplicate Debits**: Customer bank account debited more than once for a single transaction or Order ID.
- **Unauthorized Transactions & Fraud**: Any claims of account takeover, stolen credentials, unauthorized credit/debit card usage, or fraudulent orders.
- **Refund Failures Exceeding SLA**: Refunds that have not reflected in customer accounts beyond the stated 7 business day window.

### B. Logistics Breakdowns and Lost Packages (Priority: HIGH)
- **Stalled Transit**: Tracking shows zero scan updates or movement across hubs for **4 consecutive business days**.
- **Disputed Delivery**: Package marked "Delivered" on the tracking portal, but customer asserts non-receipt after checking with neighbors and building security past 4 hours.

### C. Knowledge Base Gaps and Insufficient Information
- When a customer asks about a scenario, exception, custom contract, or operational edge case that is **NOT explicitly covered** in the ShopEase Knowledge Base.
- The AI system must **NEVER speculate, extrapolate, or invent answers**. If information is lacking, it must trigger escalation with status `ESCALATED` and reason `Insufficient Knowledge Base Information`.

### D. Explicit Customer Request for Human Representative
- When the customer explicitly requests to interact with a human (e.g., "connect me to an agent", "I want to talk to a human", "call me", "speak to a manager", "transfer to support representative").

### E. Severe Sentiment Degradation and Frustration
- When sentiment analysis registers `FRUSTRATED` or when a customer query remains unresolved across multiple consecutive conversation turns.

### F. Legal, Regulatory, or Consumer Court Escalations
- Any mention of consumer court notices, legal action, regulatory complaints, or media escalation must be routed directly to the Executive Escalations and Legal Compliance Desk.

## 3. Human Escalation Service Level Agreement (SLA)
- **Operating Window**: Human specialists review and take ownership of escalated tickets daily between **8:00 AM and 10:00 PM IST**.
- **Initial Human Response SLA**:
  - `URGENT` (Payment fraud, unauthorized charges): **Under 15 minutes**.
  - `HIGH` (Duplicate charges, stalled delivery > 4 days): **Under 30 minutes**.
  - `MEDIUM` (Policy exceptions, complex return disputes): **Under 2 hours**.
- **After-Hours Protocol**: Tickets escalated outside human operating hours are queued by priority and assigned starting at 8:00 AM IST the following morning, with an automated confirmation sent to the customer.
