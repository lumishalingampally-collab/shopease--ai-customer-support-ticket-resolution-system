"""Application services package.

Service modules are intentionally imported explicitly by their callers instead of
being eagerly imported here. This keeps lightweight services (for example the
classifier and database layer) usable even when the optional heavy RAG stack is
not installed yet.
"""
