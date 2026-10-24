# ShopEase Voice + Interactive Avatar

## Experience

1. Browser microphone records the customer's issue.
2. Groq Whisper converts speech to text.
3. Hindi/Telugu is translated to English for the ShopEase knowledge-base retrieval layer.
4. The existing FastAPI classification + ChromaDB RAG pipeline remains the source of truth.
5. The grounded answer is translated back to the customer's language.
6. gTTS generates an optional spoken response.
7. The right-side AI Concierge avatar blinks, floats and animates its mouth while the answer audio plays.

## Configuration

Add these values to `.env`:

```text
GROQ_API_KEY=...
GROQ_WHISPER_MODEL=whisper-large-v3
GROQ_TRANSLATION_MODEL=openai/gpt-oss-20b
```

For local NLLB, set `NLLB_MODEL_PATH` to the model directory. The app tries NLLB first and falls back to Groq text translation if NLLB is not configured/available.

The voice layer is deliberately outside the RAG policy layer: translation is never treated as a source of truth. This prevents translated text from becoming invented ShopEase policy.
