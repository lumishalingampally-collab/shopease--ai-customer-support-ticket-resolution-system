"""Grounded response generation for ShopEase support."""
import logging
import re
from typing import List, Dict, Optional
from app.schemas.chat import ClassificationResult, GeneratedResponse, PolicyReference
from app.schemas.kb import KBSearchResult
from app.services.llm_service import LLMService, get_llm_service

logger = logging.getLogger(__name__)

SYSTEM_RESPONSE_INSTRUCTION = """You are ShopEase's customer-support response writer.
Answer ONLY from the supplied AUTHORITATIVE SHOP EASE KB EVIDENCE.
Never invent policy, prices, timelines, eligibility, order status, or actions.
The latest customer question is primary; use ONLY the relevant conversation history supplied for a genuine follow-up.
The retrieved policy chunks below are the ONLY factual source you may use.
Be concise, friendly and interactive (normally 3-7 short lines). Start with the direct answer.
Use bullets for steps/details. Never paste an entire KB section.
For service-availability questions, include the directly relevant eligibility/location, cutoff, and cost details when those facts are present in the supplied evidence.
If the retrieved evidence does not explicitly answer the question, set sufficient_information=false and do not guess.
Do not turn a related policy into an answer. Include one supplied Policy reference when evidence supports the answer.
Match the detected customer language when possible."""


class ResponseService:
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or get_llm_service()

    @staticmethod
    def _is_greeting(query: str) -> bool:
        q = re.sub(r"[^a-zA-Z\s]", "", query.lower()).strip()
        return q in {"hi", "hello", "hey", "hey there", "hi there", "hello there", "good morning", "good afternoon", "good evening", "namaste"}

    @staticmethod
    def _policy_reference(chunk: KBSearchResult) -> PolicyReference:
        stem = chunk.source.rsplit(".", 1)[0]
        prefixes = {
            "shipping_and_delivery": "SHIP", "returns_and_refunds": "RETURN",
            "payment_and_billing": "PAY", "account_and_security": "ACCOUNT",
            "cancellations_and_modifications": "ORDER", "damaged_and_wrong_products": "PRODUCT",
            "promotions_and_coupons": "PROMO", "general_faq": "FAQ", "human_escalation_policy": "ESCALATE",
        }
        prefix = prefixes.get(stem, re.sub(r"[^A-Z0-9]", "", stem.upper())[:10] or "KB")
        m = re.search(r"_(?:SEC_)(\d+)$", chunk.chunk_id.upper())
        number = m.group(1) if m else "00"
        return PolicyReference(reference_id=f"KB-{prefix}-{int(number):02d}", source=chunk.source, section=chunk.topic, relevance_score=round(float(chunk.similarity_score), 4))

    def _build_grounding_prompt(self, query: str, classification: ClassificationResult, chunks: List[KBSearchResult], conversation_history=None) -> str:
        history = []
        for msg in (conversation_history or [])[-8:]:
            sender = msg.get("role") or msg.get("sender") or "USER"
            history.append(f"{sender.upper()}: {msg.get('content', '')}")
        evidence = []
        for chunk in chunks:
            ref = self._policy_reference(chunk)
            # Pass the actual retrieved policy chunk to Gemini. The chunk is the
            # semantic retrieval result; evidence selection is shown separately
            # for UI readability but never substitutes for the retrieved source.
            full_text = chunk.content.strip()
            if len(full_text) > 7000:
                full_text = full_text[:7000] + "\n[retrieved chunk truncated for prompt safety]"
            evidence.append(
                f"REFERENCE: {ref.reference_id}\nSOURCE: {ref.source}\nSECTION: {ref.section}\n"
                f"RELEVANCE: {ref.relevance_score:.3f}\nRETRIEVED POLICY CHUNK (AUTHORITATIVE):\n{full_text}"
            )
        return (
            "CUSTOMER QUESTION:\n" + query.strip() + "\n\n" +
            f"QUERY ANALYSIS:\nCategory={classification.category}\nPriority={classification.priority}\n"
            f"Intent={classification.intent}\nSentiment={classification.sentiment}\nLanguage={classification.language}\n"
            f"Human escalation={classification.requires_human}\n\n" +
            "CONVERSATION HISTORY:\n" + ("\n".join(history) if history else "No previous messages.") + "\n\n" +
            "AUTHORITATIVE RETRIEVED KNOWLEDGE:\n" + "\n\n".join(evidence) + "\n\n" +
            "Return ONLY the GeneratedResponse JSON schema. Answer the latest question using ONLY the retrieved policy chunks above. Do not use general knowledge. If the chunks do not support the answer, set sufficient_information=false."
        )

    @staticmethod
    def clarification_response(language: str) -> GeneratedResponse:
        """Ask for the missing issue detail instead of guessing a KB policy."""
        if language == "Hindi":
            msg = "ज़रूर, मैं मदद करूँगा। कृपया बताइए कि पैकेज के साथ क्या समस्या हुई—जैसे डिलीवरी में देरी, ट्रैकिंग, नुकसान, गलत सामान या कुछ और?"
        elif language == "Telugu":
            msg = "తప్పకుండా సహాయం చేస్తాను. మీ ప్యాకేజీతో ఏమి సమస్య జరిగిందో చెప్పండి—డెలివరీ ఆలస్యం, ట్రాకింగ్, డ్యామేజ్, తప్పు వస్తువు లేదా మరేదైనా?"
        else:
            msg = "Absolutely, I can help. Could you tell me what happened with the package—for example, is it delayed, not tracking, damaged, the wrong item, or something else?"
        return GeneratedResponse(
            response=msg,
            language=language or "English",
            used_knowledge_base=False,
            sufficient_information=True,
            needs_human_review=False,
        )

    @staticmethod
    def _unsupported_response(language: str, reason: str, refs=None) -> GeneratedResponse:
        if language == "Hindi":
            msg = "मुझे उपलब्ध ShopEase Knowledge Base में इस सवाल का पर्याप्त और स्पष्ट उत्तर नहीं मिला। मैं अनुमान नहीं लगाऊँगा, इसलिए इसे मानव सहायता के लिए भेज रहा हूँ।"
        elif language == "Telugu":
            msg = "అందుబాటులో ఉన్న ShopEase Knowledge Base లో ఈ ప్రశ్నకు సరిపడే స్పష్టమైన సమాచారం లేదు. నేను ఊహించి సమాధానం ఇవ్వను, కాబట్టి దీన్ని మానవ సహాయానికి పంపుతున్నాను."
        else:
            msg = "I couldn't find enough specific information in the ShopEase Knowledge Base to answer that accurately. I won't guess, so this needs human support review."
        return GeneratedResponse(response=msg, language=language or "English", used_knowledge_base=bool(refs), sufficient_information=False, needs_human_review=True, reason=reason, policy_references=refs or [], llm_used=False, generation_method="grounded_refusal", grounding_verified=True)

    @staticmethod
    def _clean_line(line: str) -> str:
        return re.sub(r"^[-*]\s*", "", line.strip()).strip()

    @classmethod
    def _select_evidence(cls, query: str, content: str, max_lines: int = 7) -> List[str]:
        """Extract a small, contiguous set of KB lines that answer the query.

        This is deliberately evidence selection, not answer generation.  It keeps
        related facts together (for example service availability + cutoff + cost)
        instead of returning a random collection of vaguely matching bullets.
        """
        lines = []
        for raw in content.splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith("#") or stripped.lower().startswith(("policy document:", "section:")):
                continue
            lines.append(cls._clean_line(raw))
        if not lines:
            return []

        q = query.lower()
        q_terms = set(re.findall(r"[a-z0-9]+", q))
        stop = {"the","a","an","is","are","i","my","me","to","for","of","and","or","in","on","it","this","that","what","how","when","can","do","does","did","will","would","could","should","please","order","get"}
        q_terms -= stop

        scored = []
        for idx, line in enumerate(lines):
            low = line.lower()
            terms = set(re.findall(r"[a-z0-9]+", low))
            score = len(q_terms & terms)
            if any(term in q for term in ("same-day", "same day", "today")) and "same-day" in low:
                score += 8
            if any(term in q for term in ("express",)) and "express" in low:
                score += 8
            if any(term in q for term in ("standard", "regular")) and "standard" in low:
                score += 8
            if any(term in q for term in ("cutoff", "cut-off", "deadline")) and any(x in low for x in ("cutoff", "11:00 am", "before")):
                score += 8
            if any(term in q for term in ("cost", "price", "fee", "surcharge", "how much")) and any(x in low for x in ("cost:", "surcharge", "charge", "₹")):
                score += 6
            if "hyderabad" in q and "hyderabad" in low:
                score += 7
            scored.append((score, idx, line))

        ranked = sorted(scored, key=lambda x: (-x[0], x[1]))
        # For delivery tiers, use the explicit tier heading as the anchor. This
        # prevents neighboring Standard/Express/Same-Day tiers from leaking into
        # one another merely because they share the word "delivery".
        tier_terms = []
        if "same-day" in q or "same day" in q or "today" in q:
            tier_terms = ["same-day priority delivery"]
        elif "express" in q:
            tier_terms = ["express shipping"]
        elif "standard" in q or "regular" in q:
            tier_terms = ["standard shipping"]
        tier_anchor = next((i for i, line in enumerate(lines) if any(t in line.lower() for t in tier_terms)), None)
        if tier_anchor is not None:
            anchors = [tier_anchor]
        else:
            anchors = [idx for score, idx, _ in ranked if score > 0][:3]
        if not anchors:
            return [line for _, _, line in ranked[:min(3, max_lines)]]

        chosen = set()
        # Preserve the service heading and the facts that follow it.  Do not pull
        # the previous service tier into an answer (for example Express should not
        # inherit Standard Shipping's price).
        strong_service_terms = ("same-day", "same day", "express", "standard shipping", "standard delivery")
        for idx in anchors:
            if any(term in lines[idx].lower() for term in strong_service_terms):
                for j in range(idx, min(len(lines), idx + 6)):
                    if j > idx and re.match(r"^\*\*(?:Standard|Express|Same-Day).*\*\*.*$", lines[j], re.I):
                        break
                    chosen.add(j)
            else:
                for j in range(max(0, idx - 1), min(len(lines), idx + 3)):
                    chosen.add(j)

        ordered = [lines[i] for i in range(len(lines)) if i in chosen]

        # For a concrete attribute question, keep only the facts needed for that
        # attribute plus the service heading. This prevents the local fallback
        # from dumping neighboring policy facts and gives Gemini a tighter
        # evidence contract.
        heading = next((line for line in ordered if "delivery" in line.lower() and line.startswith("**")), None)
        if not heading:
            heading = next((line for line in ordered if line.startswith("**")), None)
        if re.search(r"\bhow much\b|\bcost\b|\bprice\b|\bfee\b|\bsurcharge\b", q):
            relevant = [line for line in ordered if any(x in line.lower() for x in ("cost:", "surcharge", "charge", "fee", "₹", "price"))]
            if relevant:
                return ([heading] if heading and heading not in relevant else []) + relevant[:max_lines-1]
        if re.search(r"\bcut[- ]?off\b|\bdeadline\b", q):
            relevant = [line for line in ordered if any(x in line.lower() for x in ("cutoff", "before", "deadline")) or re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)\b", line.lower())]
            if relevant:
                return ([heading] if heading and heading not in relevant else []) + relevant[:max_lines-1]
        if re.search(r"\bhow long\b|\bhow many days\b|\bwhen will\b", q):
            relevant = [line for line in ordered if any(x in line.lower() for x in ("delivery window", "return window", "business days", "business hours", "days", "hours"))]
            if relevant:
                return ([heading] if heading and heading not in relevant else []) + relevant[:max_lines-1]
        return ordered[:max_lines]

    @classmethod
    def _fallback_generate(cls, query: str, classification: ClassificationResult, chunks: List[KBSearchResult]) -> GeneratedResponse:
        lang = classification.language or "English"
        if cls._is_greeting(query) or classification.intent == "greeting":
            greetings = {"Hindi": "नमस्ते! 👋 ShopEase Support में आपका स्वागत है। मैं ऑर्डर, डिलीवरी, भुगतान, रिटर्न और अकाउंट से जुड़ी जानकारी में मदद कर सकता हूँ।", "Telugu": "నమస్కారం! 👋 ShopEase Support‌కు స్వాగతం. ఆర్డర్, డెలివరీ, పేమెంట్, రిటర్న్ మరియు అకౌంట్ సమస్యల్లో నేను సహాయం చేయగలను."}
            return GeneratedResponse(response=greetings.get(lang, "Hi! 👋 I’m ShopEase AI Support. What can I help you with?"), language=lang, used_knowledge_base=False, sufficient_information=True)
        if not chunks:
            return cls._unsupported_response(lang, "No sufficiently relevant KB evidence was retrieved.")
        ref = cls._policy_reference(chunks[0])
        selected = cls._select_evidence(query, chunks[0].content)
        if not selected:
            return cls._unsupported_response(lang, "The retrieved section did not contain a usable answer.", [ref])

        # The supplied KB defines failed/dropped payments, but does NOT define a
        # separate workflow for a payment that remains Pending. Never invent one.
        if classification.intent == "payment_status_pending":
            if lang == "English":
                text = ("### Payment status\n\n"
                        "The ShopEase KB covers **failed/dropped payment** cases, but it does not define a separate workflow for a payment that simply remains **Pending**. I won’t guess.\n\n"
                        "If your case is a failed/dropped payment, the documented policy includes:\n" +
                        "\n".join(f"- {x}" for x in selected[:3]) +
                        f"\n\n**KB reference:** `{ref.reference_id}`")
            elif lang == "Hindi":
                text = f"### Payment status\n\nKB में failed/dropped payment की policy है, लेकिन सिर्फ **Pending** status के लिए अलग प्रक्रिया नहीं दी गई है। मैं अनुमान नहीं लगाऊँगा।\n\n**KB reference:** `{ref.reference_id}`"
            else:
                text = f"### Payment status\n\nKBలో failed/dropped payment గురించి సమాచారం ఉంది, కానీ **Pending** status కోసం ప్రత్యేక ప్రక్రియ లేదు. నేను ఊహించి సమాధానం ఇవ్వను.\n\n**KB reference:** `{ref.reference_id}`"
            return GeneratedResponse(response=text, language=lang, used_knowledge_base=True, sufficient_information=False, needs_human_review=True, reason="The supplied KB does not define a specific pending-payment workflow.", policy_references=[ref])

        prefix = {"Hindi": "### ShopEase नीति\n\nउपलब्ध नीति के अनुसार:\n", "Telugu": "### ShopEase పాలసీ\n\nఅందుబాటులో ఉన్న పాలసీ ప్రకారం:\n"}.get(lang, "### ShopEase policy\n\nHere’s the relevant policy:\n")
        msg = prefix + "\n".join(f"- {x}" for x in selected) + f"\n\n**KB reference:** `{ref.reference_id}`"
        return GeneratedResponse(response=msg, language=lang, used_knowledge_base=True, sufficient_information=True, needs_human_review=classification.requires_human, reason=classification.escalation_reason, policy_references=[ref], llm_used=False, generation_method="local_kb_fallback", grounding_verified=True)

    @staticmethod
    def _validate_answer_coverage(query: str, answer: str, evidence: str) -> bool:
        """Reject an otherwise-grounded answer that omits an explicit requested fact."""
        q = query.lower()
        a = answer.lower()
        e = evidence.lower()
        # If the customer explicitly asks for a concrete attribute, the answer
        # should contain a corresponding fact that exists in the evidence.
        checks = [
            (r"\bhow much\b|\bcost\b|\bprice\b|\bfee\b|\bsurcharge\b", r"₹|rs\.?\s*\d|\b\d+(?:\.\d+)?\s*(?:rupees|fee|surcharge|charge)", "cost"),
            (r"\bcut[- ]?off\b|\bdeadline\b", r"cutoff|before|\b\d{1,2}:\d{2}\s*(?:am|pm)?\b", "cutoff"),
            (r"\bhow long\b|\bhow many days\b|\bwhen will", r"\b\d+(?:\s*[-–]\s*\d+)?\s*(?:business\s+)?(?:days?|hours?|minutes?)\b", "timeline"),
        ]
        for qpat, apat, _ in checks:
            if re.search(qpat, q):
                if not re.search(apat, a):
                    return False
                if not re.search(apat, e):
                    return False
        # A location explicitly named by the customer should not disappear from
        # a location-specific answer when the evidence contains that location.
        locations = [w for w in re.findall(r"\b[A-Z][a-z]{3,}\b", query) if w.lower() not in {"How", "What", "Can", "My", "The"}]
        for location in locations:
            if location.lower() in e and location.lower() not in a:
                return False
        return True

    @staticmethod
    def _validate_grounding(output: GeneratedResponse, chunks: List[KBSearchResult], query: str = "") -> bool:
        if not output.response.strip() or not output.sufficient_information:
            return False
        # Reject bloated model output; the local fallback is deliberately concise.
        if len(output.response) > 1200:
            return False
        evidence = "\n".join(c.content.lower() for c in chunks)
        answer = re.sub(r"KB-[A-Z0-9_-]+", "", output.response, flags=re.I)
        if not set(re.findall(r"\d+(?:\.\d+)?", answer)).issubset(set(re.findall(r"\d+(?:\.\d+)?", evidence))):
            return False
        valid_refs = {ResponseService._policy_reference(c).reference_id for c in chunks}
        if output.policy_references and not all(r.reference_id in valid_refs for r in output.policy_references):
            return False
        return ResponseService._validate_answer_coverage(query, output.response, evidence)

    def generate_response(self, query: str, classification: ClassificationResult, retrieved_chunks: List[KBSearchResult], conversation_history=None) -> GeneratedResponse:
        if self._is_greeting(query):
            return self._fallback_generate(query, classification, [])
        if not retrieved_chunks:
            return self._fallback_generate(query, classification, [])
        # Local fallback should also understand short follow-ups. Gemini receives
        # the full history through its prompt; this query is only for selecting
        # the most relevant evidence lines from the already-retrieved section.
        fallback_query = query
        if conversation_history:
            topic_markers = (
                "payment", "paid", "card", "upi", "refund", "return", "cashback", "coupon",
                "delivery", "shipping", "ship", "tracking", "track", "courier", "package",
                "parcel", "same-day", "same day", "express", "order", "password", "login",
                "account", "otp", "damaged", "broken", "wrong item", "defective", "cancel",
                "address", "discount", "promo", "voucher"
            )
            topic_user = next((m.get("content") for m in reversed(conversation_history) if (m.get("sender") or m.get("role")) == "USER" and any(t in (m.get("content") or "").lower() for t in topic_markers)), None)
            previous_user = topic_user or next((m.get("content") for m in reversed(conversation_history) if (m.get("sender") or m.get("role")) == "USER"), None)
            if previous_user and len(query.split()) <= 12:
                fallback_query = f"{previous_user} {query}"
        if self.llm_service.is_available():
            try:
                result = self.llm_service.generate_structured(
                    prompt=self._build_grounding_prompt(query, classification, retrieved_chunks, conversation_history),
                    response_schema=GeneratedResponse,
                    system_instruction=SYSTEM_RESPONSE_INSTRUCTION,
                )
                if result and result.sufficient_information and result.response.strip() and self._validate_grounding(result, retrieved_chunks, query):
                    result.used_knowledge_base = True
                    result.llm_used = True
                    result.generation_method = "gemini"
                    result.generation_model = self.llm_service.model_name
                    result.grounding_verified = True
                    if not result.policy_references:
                        result.policy_references = [self._policy_reference(retrieved_chunks[0])]
                    if classification.requires_human:
                        result.needs_human_review = True
                        result.reason = result.reason or classification.escalation_reason
                    return result
                if result and not result.sufficient_information:
                    refusal = self._unsupported_response(
                        result.language or classification.language,
                        result.reason or "The evidence does not directly answer the question.",
                        [self._policy_reference(retrieved_chunks[0])],
                    )
                    refusal.llm_used = True
                    refusal.generation_method = "gemini_grounded_refusal"
                    refusal.generation_model = self.llm_service.model_name
                    refusal.grounding_verified = True
                    return refusal
            except Exception as exc:
                logger.warning("Grounded Gemini generation failed; using local KB fallback: %s", exc)
        return self._fallback_generate(fallback_query, classification, retrieved_chunks)


_response_service_instance: Optional[ResponseService] = None

def get_response_service() -> ResponseService:
    global _response_service_instance
    if _response_service_instance is None:
        _response_service_instance = ResponseService()
    return _response_service_instance
