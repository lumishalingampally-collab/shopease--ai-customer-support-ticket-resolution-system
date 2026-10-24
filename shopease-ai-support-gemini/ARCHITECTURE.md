# ShopEase Architecture

```text
Customer / Voice
      |
      v
Streamlit UI
      |
      v
FastAPI /api/v1/chat
      |
      +--> Conversation + Ticket persistence (SQLite / SQLAlchemy)
      |
      +--> Classifier
      |       +--> category / priority / intent / sentiment / language
      |       +--> deterministic escalation rules
      |
      +--> RAG Service
      |       +--> all 9 Markdown policy documents
      |       +--> section-aware chunking
      |       +--> local all-MiniLM-L6-v2 embeddings
      |       +--> ChromaDB cosine candidate search
      |       +--> semantic + lexical/topic/category/intent reranking
      |       +--> specificity / answerability gate
      |       +--> Top-K verified evidence
      |
      +--> Grounded Response Service
              +--> evidence package with exact retrieved KB chunks
              +--> Gemini structured generation using retrieved evidence only
              +--> numeric/reference grounding validation
              +--> safe evidence-only fallback
              +--> human escalation when evidence is insufficient
```

## Source of truth

`knowledge_base/*.md` is the only source of ShopEase policy facts. Gemini is used to structure and communicate retrieved facts; it is not used as an independent policy database.

## Retrieval

The RAG service searches the complete Chroma collection. It first retrieves a wider candidate set, then reranks candidates using semantic similarity and lexical/topic overlap. A conservative answerability gate rejects weak matches and questions whose requested concrete details (for example a percentage, amount, maximum, or minimum) are absent from the candidate evidence.

## Generation and verification

Only accepted KB evidence is passed to Gemini. The generation prompt explicitly prohibits adding facts not present in the evidence. The generated response is checked after generation: explicit numbers must occur in the retrieved evidence and returned KB reference IDs must correspond to retrieved chunks. Failed validation falls back to a concise evidence-only response.

## Traceability

Each section chunk has a deterministic internal reference such as `KB-SHIP-01`. The UI shows the reference, source Markdown filename, section title, and retrieval relevance score. These identifiers are **KB traceability references**, not invented official policy numbers.

## Failure behavior

- No relevant KB evidence -> no factual Gemini answer; human review.
- Gemini unavailable -> readable, evidence-only fallback; no hallucinated facts.
- Gemini output fails grounding validation -> evidence-only fallback.
- Serious payment/fraud/legal cases -> deterministic human escalation overrides model output.
- Conversation follow-ups use recent ticket history for retrieval context while the original user language is preserved for response generation.


## Ticket lifecycle and persistence

A first message without a `ticket_id` creates a new persistent SQLite ticket. Every subsequent message sent with that ticket ID is appended to the `messages` table. The recent history is loaded before classification and retrieval, allowing follow-up questions to resolve references such as "what about express?" while the customer's original wording remains available to the response generator.

The UI can retrieve a saved ticket through `GET /api/v1/tickets/{ticket_id}` and resume the stored conversation after a browser/Streamlit refresh. The Agent Dashboard uses the ticket-list API to inspect all stored conversations.

## RAG ranking principle

Vector similarity is used for broad candidate discovery, not as the sole definition of relevance. A semantic candidate search is followed by a hybrid reranker using semantic similarity, lexical overlap, document category, and classifier intent/category as ranking signals. There is no direct phrase-to-section answer router. The best verified retrieved chunks are passed to the response generator. This prevents semantically adjacent policies (for example delayed delivery) from outranking an exact Express/Same-Day delivery section.

No policy answer is hardcoded into the response layer: the reranker only selects among the supplied Markdown chunks.
