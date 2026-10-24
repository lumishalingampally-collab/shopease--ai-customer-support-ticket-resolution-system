"""Main FastAPI application entry point for ShopEase AI Support."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.database import init_db
from app.api.tickets import router as tickets_router
from app.api.kb import router as kb_router
from app.api.classifier import router as classifier_router
from app.api.chat import router as chat_router
from app.services.rag_service import get_rag_service

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to run startup and shutdown events."""
    # Initialize SQLite database schema
    init_db()

    # Reuse the persisted Chroma index when the KB fingerprint is unchanged;
    # automatically rebuild only when the authoritative Markdown KB changes.
    rag_service = get_rag_service()
    rag_service.index_documents(force_reindex=False)

    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="ShopEase AI-powered customer support and ticket resolution backend API.",
    lifespan=lifespan,
)

# Enable CORS for local Streamlit / Frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(tickets_router, prefix="/api/v1")
app.include_router(kb_router, prefix="/api/v1")
app.include_router(classifier_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint to verify system status."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "environment": settings.app_env,
        "version": "1.0.0"
    }


@app.get("/", tags=["Root"])
def root():
    """Root endpoint welcoming users and providing documentation links."""
    return {
        "message": f"Welcome to {settings.app_name} API",
        "docs_url": "/docs",
        "health_url": "/health"
    }
