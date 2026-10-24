# Assignment Compliance Checklist

| Assignment requirement | Implementation |
|---|---|
| Natural-language customer query | Streamlit chat + `POST /api/v1/chat` |
| Category | Structured classifier |
| Priority | Structured classifier + deterministic rules |
| Customer intent | Structured classifier |
| Sentiment | Structured classifier |
| Human intervention decision | Deterministic escalation + insufficient-context guard |
| Provided KB | 9 Markdown policy documents in `knowledge_base/` |
| Retrieval before response | ChromaDB RAG pipeline |
| No fabricated information | Hybrid relevance gate + no-context refusal + grounded Gemini prompt + safe fallback |
| Human escalation | Fraud, duplicate payment, legal threats, explicit agent request, insufficient KB |
| Conversation history | Ticket/message persistence, saved-ticket resume, and recent follow-up context |
| Tickets/history database | SQLite + SQLAlchemy |
| Ticket APIs | FastAPI ticket CRUD + persistent message endpoints (`/tickets`, `/tickets/{id}`, `/tickets/{id}/messages`) |
| Working UI | Streamlit customer + agent + knowledge center interface |
| English/Hindi/Telugu | Script/language detection, multilingual voice translation, and English-policy retrieval bridge |
| Source code | Included |
| README/setup | Included |
| Architecture explanation | `ARCHITECTURE.md` |
| Dependency file | `requirements.txt` |
| Sample tests | `tests/` |


## Persistence verification

A ticket is created when a new conversation sends its first message. Subsequent turns reuse the same ticket ID and append both USER and ASSISTANT messages. The ticket detail API returns the stored history, and the Streamlit UI can resume a saved ticket after a refresh.

## Grounding verification

The retrieval test suite explicitly checks standard, express, same-day, return, refund, duplicate-payment, delayed-delivery, delivered-but-not-received, cashback, coupon, cancellation, and unsupported-fact queries. Unsupported numeric claims such as a universal 5% cashback rate or an unspecified maximum refund are rejected instead of being invented.


## Final requirement verification

### Requirement 2 — Issue Analysis
Each incoming query is classified for **category, priority, intent, sentiment, detected language, and human-intervention requirement**. Gemini performs structured classification when available, with deterministic high-confidence business rules applied afterward for payment, shipping, return/refund, promotion, security, sentiment, and escalation signals.

### Requirement 6 — Conversation History
Every chat turn is persisted as a `Message` row linked to its `Ticket`. The chat API loads recent messages from SQLite and supplies only relevant follow-up history to retrieval/response generation. New explicit topics are not blindly contaminated by older conversation context. The Streamlit sidebar can list saved tickets and resume a selected ticket with its stored messages.

### Requirement 7 — Persistent Backend & Database
Tickets and messages are stored in a **file-backed SQLite database**. The SQLite path is resolved relative to the project root, so starting FastAPI from the project directory or its parent does not create a second database. Closing the application does not delete the database; reopening the project reads the same tickets/history. FastAPI provides `POST /api/v1/tickets`, `GET /api/v1/tickets`, `GET /api/v1/tickets/{ticket_id}` (including message history), `GET/POST /api/v1/tickets/{ticket_id}/messages`, and `PATCH` for ticket lifecycle updates.


## Requirement 4 — RAG evidence is actually supplied to the LLM
For a supported question, ChromaDB retrieves the top verified policy chunks first. `ResponseService._build_grounding_prompt()` then sends those actual retrieved chunks to Gemini with source filename, section, relevance score, and the retrieved policy text. The LLM is explicitly instructed to use only that evidence. The chat API returns `llm_used`, `generation_method`, `generation_model`, `grounding_verified`, `retrieval_query`, and the exact `retrieved_evidence` used for the response.

A local response is explicitly labeled `local_kb_fallback` when Gemini is unavailable or fails validation; the application never presents that fallback as a Gemini-generated answer.
