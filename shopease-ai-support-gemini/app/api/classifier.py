"""FastAPI router for AI query classification and issue analysis."""

from fastapi import APIRouter, Depends, status
from app.schemas.chat import ClassifyRequest, ClassificationResult
from app.services.classifier_service import ClassifierService, get_classifier_service

router = APIRouter(tags=["AI Query Classification"])


@router.post(
    "/classify",
    response_model=ClassificationResult,
    status_code=status.HTTP_200_OK,
    summary="Analyze and classify customer support query"
)
def classify_support_query(
    request: ClassifyRequest,
    classifier_service: ClassifierService = Depends(get_classifier_service)
):
    """Analyze customer query, detect category/priority/sentiment/intent/language, and apply escalation rules."""
    return classifier_service.classify_query(query=request.query)
