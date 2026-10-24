# RAG Interview Demonstration

The interviewer can verify the retrieval path without relying on the Streamlit UI.

From the project root:

```powershell
python scripts/demo_rag.py
```

The utility prints, for each sample query:

1. the customer query;
2. whether sufficient KB context was found;
3. the actual source Markdown file;
4. the actual KB section;
5. the semantic similarity returned by the retrieval layer;
6. the hybrid retrieval score; and
7. a short excerpt from the retrieved evidence.

For an unsupported request such as `Do you offer 5% cashback on every order?`,
the retriever should return no sufficiently supported context because the supplied
KB does not establish a 5% cashback rate.

This utility deliberately stops at retrieval. The normal `/api/v1/chat` endpoint
then passes the retrieved evidence to the Gemini response layer, where grounding
validation is applied before a generated answer is accepted.
