"""Pydantic schemas for AI chat, grounded response generation, and classification."""

from typing import Optional, List
from pydantic import BaseModel, Field, model_validator
from app.models.ticket import TicketStatus, TicketPriority, TicketCategory, SentimentType


class ClassifyRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Customer issue or question in natural language")


class ClassificationResult(BaseModel):
    category: TicketCategory = Field(default=TicketCategory.OTHER)
    priority: TicketPriority = Field(default=TicketPriority.MEDIUM)
    intent: str = Field(default="general_question")
    sentiment: SentimentType = Field(default=SentimentType.NEUTRAL)
    language: str = Field(default="English")
    requires_human: bool = Field(default=False)
    escalation_reason: Optional[str] = Field(default=None)
    analysis_source: str = Field(default="deterministic_policy_rules")


class PolicyReference(BaseModel):
    """Traceable reference to the exact KB evidence used for an answer."""
    reference_id: str = Field(..., description="Stable internal KB reference, not an external/official policy number")
    source: str = Field(..., description="Markdown KB document filename")
    section: str = Field(..., description="Exact Markdown section heading")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)


class RetrievedEvidence(BaseModel):
    """Exact KB evidence exposed to the UI for auditability/interview demo."""
    source: str
    section: str
    reference_id: str
    relevance_score: float = 0.0
    content: str


class GeneratedResponse(BaseModel):
    """Grounded customer-facing response plus auditable KB references."""
    response: str = Field(..., description="Customer-friendly grounded answer, preferably Markdown")
    language: str = Field(default="English")
    used_knowledge_base: bool = Field(default=True)
    sufficient_information: bool = Field(default=True)
    needs_human_review: bool = Field(default=False)
    reason: Optional[str] = Field(default=None)
    policy_references: List[PolicyReference] = Field(default_factory=list)
    retrieved_evidence: List[RetrievedEvidence] = Field(default_factory=list)
    llm_used: bool = Field(default=False)
    generation_method: str = Field(default="local_kb_fallback")
    generation_model: Optional[str] = None
    grounding_verified: bool = Field(default=False)
    retrieval_query: Optional[str] = None


class ChatRequest(BaseModel):
    query: Optional[str] = Field(default=None)
    message: Optional[str] = Field(default=None)
    ticket_id: Optional[str] = Field(default=None)
    customer_name: Optional[str] = Field(default=None, max_length=100)
    customer_email: Optional[str] = Field(default=None, max_length=100)
    language: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def validate_query_or_message(self):
        effective_query = self.query or self.message
        if not effective_query or not effective_query.strip():
            raise ValueError("Either 'query' or 'message' must be provided with non-empty text.")
        self.query = effective_query.strip()
        self.message = effective_query.strip()
        return self


class ChatResponse(BaseModel):
    ticket_id: str
    ticket_number: str
    response: str
    reply: Optional[str] = None
    classification: ClassificationResult
    sources: List[str] = Field(default_factory=list)
    policy_references: List[PolicyReference] = Field(default_factory=list)
    retrieved_evidence: List[RetrievedEvidence] = Field(default_factory=list)
    used_knowledge_base: bool = True
    sufficient_information: bool = True
    needs_human_review: bool = False
    status: TicketStatus = TicketStatus.IN_PROGRESS
    llm_used: bool = False
    generation_method: str = "local_kb_fallback"
    generation_model: Optional[str] = None
    grounding_verified: bool = False
    retrieval_query: Optional[str] = None

    @model_validator(mode="after")
    def populate_reply_alias(self):
        if not self.reply:
            self.reply = self.response
        return self
