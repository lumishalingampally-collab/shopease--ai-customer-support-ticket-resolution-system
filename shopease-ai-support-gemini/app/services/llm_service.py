"""Gemini LLM Service for structured output generation."""

import logging
from typing import Type, TypeVar, Optional
from pydantic import BaseModel

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

T = TypeVar("T", bound=BaseModel)


class LLMService:
    """Wrapper around Google Gemini GenAI SDK for structured schema-based outputs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name or settings.gemini_model
        self._client = None

        if self.is_configured:
            try:
                from google import genai
                from google.genai import types
                # Keep the support UI responsive. A slow/unreachable model must
                # never block a customer ticket for minutes; the grounded local
                # fallback will answer/refuse safely after this timeout.
                self._client = genai.Client(
                    api_key=self.api_key,
                    http_options=types.HttpOptions(timeout=int(settings.gemini_timeout_seconds * 1000)),
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")
                self._client = None

    @property
    def is_configured(self) -> bool:
        """Check whether a valid Gemini API key is configured."""
        return bool(
            self.api_key
            and self.api_key.strip()
            and self.api_key != "your_gemini_api_key_here"
        )

    def is_available(self) -> bool:
        """Check if LLM client is ready for requests."""
        return self.is_configured and self._client is not None

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: Optional[str] = None
    ) -> Optional[T]:
        """Generate structured output validated against a Pydantic schema using Gemini.
        
        Returns None if client is unavailable or if generation fails, allowing graceful fallback.
        """
        if not self.is_available():
            logger.warning("Gemini is unavailable: configured=%s client=%s model=%s", self.is_configured, self._client is not None, self.model_name)
            return None

        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                
            )
            if system_instruction:
                config.system_instruction = system_instruction

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            # Check if SDK pre-parsed into the Pydantic model
            if hasattr(response, "parsed") and response.parsed is not None:
                if isinstance(response.parsed, response_schema):
                    return response.parsed

            # Otherwise, parse raw JSON text response
            if response.text:
                return response_schema.model_validate_json(response.text)

            logger.warning("Empty response received from Gemini API.")
            return None

        except Exception as e:
            logger.error("Gemini API generation error (model=%s): %s", self.model_name, e, exc_info=False)
            return None


# Global singleton instance
_llm_service_instance: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Provide singleton instance of LLMService."""
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
    return _llm_service_instance
