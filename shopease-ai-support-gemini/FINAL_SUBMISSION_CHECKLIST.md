# Final Submission Checklist

## Run

1. Create/activate the virtual environment and install `requirements.txt`.
2. Copy `.env.example` to `.env` and provide valid credentials only locally. Never commit `.env`.
3. Start FastAPI on port 8000.
4. Start Streamlit on port 8501.

## Grounded RAG acceptance tests

### Supported
- `How long does standard shipping take?`
- `How much does express shipping cost?`
- `Do you offer same-day delivery?`
- `What is the return window for electronics?`
- `My card was charged twice for the same order.`
- `How long does a UPI refund take?`

These should use KB evidence and show a KB reference/source section.

### Must refuse to guess
- `What is the maximum refund amount ShopEase allows?`
- `Does ShopEase offer 5% cashback?`
- `What compensation will I get for exactly 2 days of standard-shipping delay?` (unless the KB directly specifies it)
- An unrelated non-ShopEase question.

These should not receive invented factual answers and should be marked for human review when the KB is insufficient.

### Multi-turn
1. `How long does express shipping take?`
2. `And how much does it cost?`

The second turn should use the previous context while grounding the answer in the KB.

## Traceability

Every grounded answer can expose a stable internal KB reference such as `KB-SHIP-01`, the exact Markdown source, section title, and retrieval relevance score. These IDs are traceability references generated from the supplied KB, not claims of official external policy numbering.

## Verification performed on the packaged source

- Python source compilation: passed.
- Secret-pattern scan of the packaged source: no embedded API keys detected.
- The full pytest suite could not be executed in this build environment because external Python packages such as ChromaDB/Sentence Transformers are not installed and network access is unavailable. Run `pytest -q` once in the supplied project virtual environment after dependencies are installed.


### Persistence / requirement 6-7 acceptance test
1. Create a chat ticket and send at least two messages.
2. Note the ticket number shown in the UI.
3. Stop FastAPI and Streamlit.
4. Start them again.
5. Open **Saved conversations**, select the ticket, and choose **Resume selected ticket**.
6. The ticket metadata and all stored user/assistant messages should reappear.

### Issue-analysis acceptance test
For each query, the right-side **AI Analysis** panel should expose category, priority, intent, sentiment, language, and human-review status. High-risk examples such as duplicate payment should be HIGH + human escalation.
