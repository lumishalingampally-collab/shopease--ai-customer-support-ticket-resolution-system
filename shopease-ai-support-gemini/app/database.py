"""Database connection and session management for SQLite."""

from typing import Generator
from sqlalchemy import create_engine
from pathlib import Path
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.config import get_settings

settings = get_settings()


def _resolve_database_url(database_url: str) -> str:
    """Keep the SQLite database beside the project, regardless of launch directory.

    This is important for requirement 7: tickets and conversation history must
    survive closing/reopening the application. A relative SQLite URL otherwise
    creates different databases when the server is started from different folders.
    """
    if not database_url.startswith("sqlite:///"):
        return database_url
    if database_url.startswith("sqlite:///:memory:"):
        return database_url

    raw_path = database_url[len("sqlite:///"):]
    path = Path(raw_path)
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        path = (project_root / path).resolve()
    return f"sqlite:///{path.as_posix()}"


resolved_database_url = _resolve_database_url(settings.database_url)

# For SQLite, check_same_thread=False allows FastAPI multi-threading
connect_args = {"check_same_thread": False} if resolved_database_url.startswith("sqlite") else {}

engine = create_engine(
    resolved_database_url,
    connect_args=connect_args,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Dependency provider for database session in FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all database tables on application startup."""
    from app.models import Ticket, Message  # Ensure models are imported
    Base.metadata.create_all(bind=engine)
