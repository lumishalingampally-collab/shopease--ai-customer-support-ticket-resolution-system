"""Pydantic schemas for Knowledge Base Search and Indexing."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class KBSearchRequest(BaseModel):
    """Payload for querying the knowledge base."""
    query: str = Field(..., min_length=1, description="Search query or customer question")
    top_k: int = Field(default=3, ge=1, le=10, description="Maximum number of relevant chunks to retrieve")
    threshold: Optional[float] = Field(
        default=None,
        description="Optional maximum cosine distance threshold (defaults to settings.retrieval_threshold)"
    )


class KBSearchResult(BaseModel):
    """A single retrieved knowledge base chunk."""
    content: str = Field(..., description="Retrieved chunk text content")
    source: str = Field(..., description="Source markdown document filename")
    category: str = Field(..., description="Document category")
    topic: str = Field(..., description="Specific section heading/topic")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    distance: float = Field(..., description="Vector cosine distance (lower means more similar)")
    similarity_score: float = Field(..., description="Normalized similarity score (0.0 to 1.0, higher is better)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Complete chunk metadata")


class KBSearchResponse(BaseModel):
    """Response containing retrieved relevant chunks."""
    query: str
    results_count: int
    has_sufficient_context: bool = Field(
        ...,
        description="True if at least one retrieved chunk satisfies the relevance threshold"
    )
    results: List[KBSearchResult]


class KBReindexResponse(BaseModel):
    """Response returned upon re-indexing knowledge base documents."""
    status: str
    total_documents: int
    total_chunks: int
    message: str


class KBStatusResponse(BaseModel):
    """Status information for knowledge base vector store."""
    is_indexed: bool
    total_chunks: int
    collection_name: str
    persist_directory: str
    embedding_model: str
    threshold: float
