"""FastAPI router for Knowledge Base management and semantic retrieval."""

from fastapi import APIRouter, Depends, status
from app.schemas.kb import (
    KBSearchRequest,
    KBSearchResponse,
    KBReindexResponse,
    KBStatusResponse,
)
from app.services.rag_service import RAGService, get_rag_service

router = APIRouter(prefix="/kb", tags=["Knowledge Base & RAG"])


@router.post(
    "/reindex",
    response_model=KBReindexResponse,
    status_code=status.HTTP_200_OK,
    summary="Rebuild or update the ChromaDB knowledge base"
)
def reindex_knowledge_base(
    rag_service: RAGService = Depends(get_rag_service)
):
    """Scan the knowledge_base directory, parse markdown files, and re-embed documents into ChromaDB."""
    result = rag_service.index_documents(force_reindex=True)
    return KBReindexResponse(
        status=result.get("status", "success"),
        total_documents=result.get("total_documents", 0),
        total_chunks=result.get("total_chunks", 0),
        message=result.get("message", "Indexing completed.")
    )


@router.get(
    "/status",
    response_model=KBStatusResponse,
    summary="Get knowledge base indexing health and status"
)
def get_kb_status(
    rag_service: RAGService = Depends(get_rag_service)
):
    """Check whether the vector database is initialized and how many chunks are indexed."""
    status_info = rag_service.get_status()
    return KBStatusResponse(**status_info)


@router.post(
    "/search",
    response_model=KBSearchResponse,
    summary="Retrieve relevant knowledge base chunks for a query"
)
def search_knowledge_base(
    request: KBSearchRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Retrieve top-K relevant policy chunks using vector similarity and apply distance threshold filtering."""
    return rag_service.search(
        query=request.query,
        top_k=request.top_k,
        threshold=request.threshold
    )
