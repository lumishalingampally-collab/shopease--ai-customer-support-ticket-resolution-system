# Final Verification Notes

This build was modified from the supplied project rather than rebuilt as a toy application.

## Verified in the available environment

- Python compilation for app/frontend/tests.
- Deterministic issue analysis for ambiguous package complaints, damaged-return requests, tracking, cancellation, payment pending, COD mismatch, replacement status, and same-day delivery.
- ChromaDB semantic candidate retrieval across all supplied Markdown KB sections, followed by hybrid semantic/lexical/category/intent reranking.
- Evidence selection for express timing, same-day availability, same-day cost, and same-day cutoff.
- Unsupported 5% cashback is rejected before retrieval can become a false answer.
- SQLite ticket/message persistence smoke test.
- Regression test module added for the previously observed false-positive scenarios.

## Environment limitation

The execution environment has no external package/DNS access. An attempt to install the project's ChromaDB, Sentence Transformers, and Google GenAI dependencies failed because PyPI could not be reached. The project therefore retains the real ChromaDB + `all-MiniLM-L6-v2` RAG implementation and the exact dependencies in `requirements.txt`, but a live Chroma/Gemini integration run could not be honestly claimed here.

## Runtime

Install the supplied requirements in the user's working virtual environment before starting FastAPI/Streamlit. The application keeps a grounded local fallback if Gemini is unavailable; ChromaDB remains the semantic retrieval engine when installed.
