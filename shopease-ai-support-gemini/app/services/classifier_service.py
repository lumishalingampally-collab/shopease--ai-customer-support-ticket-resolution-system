"""Query Classification and Deterministic Escalation Service for ShopEase."""

import re
import logging
from typing import Optional

from app.models.ticket import (
    TicketPriority,
    TicketCategory,
    SentimentType,
)
from app.schemas.chat import ClassificationResult
from app.services.llm_service import LLMService, get_llm_service

logger = logging.getLogger(__name__)

SYSTEM_CLASSIFICATION_INSTRUCTION = """You are the AI Support Classifier for ShopEase, an e-commerce platform.
Your ONLY role is to analyze incoming customer support queries and return a structured classification.
Do NOT generate customer-facing answers. Do NOT invent order IDs or personal data.

Available Categories:
- PAYMENT: Failed payments, duplicate debits, refund credit delays, card/UPI charges, invoices.
- RETURN_REFUND: Return requests, refund inquiries, product exchanges for size/color.
- SHIPPING: Delivery timelines, tracking updates, courier delays, missing/lost packages.
- ORDER: Order cancellations, shipping address updates before dispatch.
- ACCOUNT: Password resets, 2FA, OTP verification, profile details, login lockouts.
- PROMOTION: Discount coupon codes, bank instant discounts, cashback offers.
- PRODUCT: Damaged, defective, or incorrect products received.
- OTHER: General inquiries, company hours, warranty questions, or unclear requests.

Priorities:
- URGENT: Unauthorized card transactions, fraud, legal/court threats.
- HIGH: Duplicate debits, missing parcels past 4 days, serious payment issues.
- MEDIUM: Standard returns, shipping tracking, delayed refunds within SLA.
- LOW: General policy queries, promotions, store hours.

Sentiments:
- POSITIVE, NEUTRAL, NEGATIVE, FRUSTRATED.

Languages:
- English, Hindi, Telugu, Other. (Support transliterated Hinglish/Telugu script).

Intents:
Use concise canonical names (e.g., duplicate_charge, payment_failed, unauthorized_transaction,
return_request, refund_status, order_tracking, delayed_delivery, password_reset,
account_security, coupon_issue, damaged_product, wrong_product, talk_to_agent, general_question, unknown).

Escalation:
Identify if human intervention is needed according to serious financial/logistical risks.
"""


class ClassifierService:
    """Service that orchestrates LLM classification and enforces deterministic escalation rules."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or get_llm_service()

    def detect_language_heuristics(self, text: str) -> str:
        """Identify language using script Unicode ranges or transliteration markers."""
        # Devanagari script (Hindi)
        if re.search(r"[\u0900-\u097F]", text):
            return "Hindi"
        # Telugu script
        if re.search(r"[\u0C00-\u0C7F]", text):
            return "Telugu"
        # Transliterated Hindi indicators
        if re.search(r"\b(kya|kaise|paise|kat|gaye|mera|meri|karna|hai|nahi|aaya)\b", text, re.IGNORECASE):
            return "Hindi"
        # Transliterated Telugu indicators
        if re.search(r"\b(enti|ela|dabbu|raledu|kattindi|chesanu|undi|cheyandi)\b", text, re.IGNORECASE):
            return "Telugu"
        return "English"

    def apply_deterministic_issue_analysis(self, query: str, result: ClassificationResult) -> ClassificationResult:
        """Strengthen requirement-2 analysis with high-confidence business signals.

        Gemini remains the classifier for nuanced cases. These rules only override
        fields when the customer wording contains a strong, unambiguous signal,
        preventing a model classification from sending a return/payment/shipping
        issue to the wrong category or intent.
        """
        q = query.lower().strip()

        # A generic package/order complaint is not enough to choose a KB policy.
        # Do not let semantic similarity guess what actually happened.
        generic_ambiguous = re.fullmatch(
            r"(?:i\\s+)?(?:have|got)\\s+(?:a\\s+)?(?:problem|issue)\\s+with\\s+(?:my\\s+)?(?:package|parcel|order)", q
        ) or q in {
            "i have a problem with my package", "problem with my package",
            "issue with my package", "problem with my order", "issue with my order",
        }
        if generic_ambiguous:
            result.category = TicketCategory.OTHER
            result.priority = TicketPriority.LOW
            result.intent = "needs_clarification"
            result.requires_human = False
            result.escalation_reason = None
            return result

        # Language is deterministic from the customer's script when possible.
        detected = self.detect_language_heuristics(query)
        if detected in {"Hindi", "Telugu"}:
            result.language = detected

        # Category + intent: strongest specific intents first.
        # Payment-status language MUST win over generic "order status/tracking"
        # language (e.g. "my order says payment pending").
        if re.search(r"payment (?:is )?(?:pending|processing|not updated|not updating|not reflected)|payment status|paid .*payment (?:status|is) .*pending|payment successful .*pending|money (?:was )?(?:deducted|debited) .*pending|paid .*order .*payment pending", q):
            result.category = TicketCategory.PAYMENT
            result.intent = "payment_status_pending"
            if result.priority == TicketPriority.LOW:
                result.priority = TicketPriority.MEDIUM
        elif re.search(r"charged twice|debited twice|double charge|duplicate charge|duplicate payment|multiple debits|दो बार|రెండుసార్లు", q):
            result.category = TicketCategory.PAYMENT
            result.intent = "duplicate_charge"
            result.priority = TicketPriority.HIGH
        elif re.search(r"unauthorized|not my transaction|card fraud|fraudulent|stolen card|धोखाधड़ी|अनधिकृत|అనుమతి లేకుండా", q):
            result.category = TicketCategory.PAYMENT
            result.intent = "unauthorized_transaction"
            result.priority = TicketPriority.URGENT
        elif re.search(r"cash on delivery|\\bcod\\b|agent.*asking.*money|asking.*cash|paid.*delivery.*cash|already paid.*cod|paid.*but.*cod", q):
            result.category = TicketCategory.PAYMENT
            result.intent = "cod_payment_mismatch"
            result.priority = TicketPriority.HIGH
            result.requires_human = True
            result.escalation_reason = "Payment status mismatch with delivery agent requires specialist resolution."
        elif re.search(r"payment method|payment option|mode of payment|how (?:can|do) i pay|accepted payment|payment.*support|supported.*payment|payment.*accept|payment.*ways|what payment|ways to pay|pay with|भुगतान विधि|చెల్లింపు విధానం", q):
            result.category = TicketCategory.PAYMENT
            result.intent = "payment_methods"
            if result.priority == TicketPriority.LOW:
                result.priority = TicketPriority.LOW
        elif re.search(r"refund|money back|रिफंड|वापसी.*पैसे|రిఫండ్", q):
            result.category = TicketCategory.RETURN_REFUND
            result.intent = "refund_status"
            if result.priority == TicketPriority.LOW:
                result.priority = TicketPriority.MEDIUM
        elif re.search(r"damaged|broken|defective|cracked|smashed|destroyed|totally damaged|badly damaged|खराब|क्षतिग्रस्त|పాడైంది|డామేజ్", q) and re.search(r"return|send.*back|how.*return|put.*return|вapас|वापस|रिटर्न|తిరిగి", q):
            result.category = TicketCategory.RETURN_REFUND
            result.intent = "damaged_return_request"
            if result.priority == TicketPriority.LOW:
                result.priority = TicketPriority.MEDIUM
        elif re.search(r"damaged|broken|defective|cracked|smashed|destroyed|totally damaged|badly damaged|खराब|क्षतिग्रस्त|పాడైంది|డామేజ్", q):
            result.category = TicketCategory.PRODUCT
            result.intent = "damaged_product"
            if result.priority == TicketPriority.LOW:
                result.priority = TicketPriority.MEDIUM
        elif re.search(r"cancel|cancellation|no longer want|don't want|dont want|do not want|changed my mind|not want this|not want that|want to cancel|अब नहीं चाहिए|इसे नहीं चाहिए|ఇంకా వద్దు", q):
            result.category = TicketCategory.ORDER
            result.intent = "order_cancellation"
            if result.priority == TicketPriority.LOW:
                result.priority = TicketPriority.MEDIUM
        elif re.search(r"return|send.*back|return.*product|वापस|रिटर्न|తిరిగి.*ఇవ్వ|వాపసు", q):
            result.category = TicketCategory.RETURN_REFUND
            result.intent = "return_request"
            if result.priority == TicketPriority.LOW:
                result.priority = TicketPriority.MEDIUM
        elif re.search(r"cashback|cash back|कैशबैक|క్యాష్‌బ్యాక్", q):
            result.category = TicketCategory.PROMOTION
            result.intent = "cashback_offer"
        elif re.search(r"coupon|promo|discount|voucher|कूपन|डिस्काउंट|కూపన్", q):
            result.category = TicketCategory.PROMOTION
            result.intent = "coupon_issue"
            result.priority = TicketPriority.LOW
        elif re.search(r"replacement.*(not received|not arrived|still waiting|didn't receive|did not receive)|replacement.*till now|replacement.*pending", q):
            result.category = TicketCategory.PRODUCT
            result.intent = "replacement_status"
            result.priority = TicketPriority.MEDIUM
            result.requires_human = True
            result.escalation_reason = "The KB documents replacement eligibility and dispatch timing, but does not provide a replacement-status tracking workflow."
        elif re.search(r"delivered but.*(not received|didn't receive|did not receive)|tracking says delivered|marked delivered|डिलीवर.*नहीं मिला|డెలివర్.*రాలేదు", q):
            result.category = TicketCategory.SHIPPING
            result.intent = "delivered_not_received"
            result.priority = TicketPriority.HIGH
        elif re.search(r"four consecutive|4 consecutive|hasn't moved|has not moved|stalled|lost in transit|चार दिन|నాలుగు రోజులు", q):
            result.category = TicketCategory.SHIPPING
            result.intent = "delayed_delivery"
            result.priority = TicketPriority.HIGH
        elif re.search(r"same-day|same day|deliver today|delivery today|today delivery", q):
            result.category = TicketCategory.SHIPPING
            result.intent = "same_day_delivery"
            if result.priority == TicketPriority.LOW:
                result.priority = TicketPriority.MEDIUM
        elif re.search(r"express delivery|express shipping", q):
            result.category = TicketCategory.SHIPPING
            result.intent = "express_delivery"
            if result.priority == TicketPriority.HIGH:
                result.priority = TicketPriority.MEDIUM
        elif re.search(r"standard delivery|standard shipping|regular delivery|regular shipment", q):
            result.category = TicketCategory.SHIPPING
            result.intent = "standard_delivery"
            if result.priority == TicketPriority.HIGH:
                result.priority = TicketPriority.MEDIUM
        elif re.search(r"cutoff|cut-off|order cutoff|डिलीवरी|டெலிவரி|డెలివరీ", q) and re.search(r"delivery|shipping|same-day|same day|today|डिलीवरी|డెలివరీ", q):
            result.category = TicketCategory.SHIPPING
            result.intent = "delivery_cutoff"
        elif re.search(r"where is my order|where's my order|where my order is|order location|order whereabouts|delivery status|delivery tracking|delivery|shipping|ship|tracking|track|courier|order delivered|डिलीवर|डिलीवरी|ऑर्डर|ఆర్డర్|డెలివరీ", q):
            result.category = TicketCategory.SHIPPING
            result.intent = "order_tracking"
            if result.priority == TicketPriority.HIGH:
                result.priority = TicketPriority.MEDIUM

        # Sentiment signals required by requirement 2.
        if re.search(r"worst|terrible|ridiculous|disgusted|scam|cheating|waste my time|frustrated|angry|बहुत परेशान|गुस्सा|చాలా కోపం", q):
            result.sentiment = SentimentType.FRUSTRATED
        elif re.search(r"not working|problem|issue|missing|delay|late|didn't receive|did not receive|damaged|broken|defective|नहीं|समस्या|आलस्यम|ఆలస్యం|రాలేదు", q):
            result.sentiment = SentimentType.NEGATIVE
        elif re.search(r"thank|thanks|great|good|satisfied|helpful|धन्यवाद|धन्य|ధన్యవాదాలు", q):
            result.sentiment = SentimentType.POSITIVE

        return result

    def apply_deterministic_escalation(
        self,
        query: str,
        result: ClassificationResult
    ) -> ClassificationResult:
        """Apply strict deterministic escalation rules per ShopEase human_escalation_policy.md.
        
        These business rules override LLM recommendations to prevent safety and financial risks.
        """
        q_lower = query.lower()

        # 1. Unauthorized transactions / suspected card fraud (URGENT)
        fraud_patterns = [
            r"unauthorized", r"stolen", r"card fraud", r"fraudulent", r"hacked",
            r"did not authorize", r"not my transaction", r"धोखाधड़ी", r"अनधिकृत",
            r"హ్యాక్", r"అనుమతి లేకుండా"
        ]
        if any(re.search(p, q_lower) for p in fraud_patterns):
            result.category = TicketCategory.PAYMENT
            result.priority = TicketPriority.URGENT
            result.intent = "unauthorized_transaction"
            result.requires_human = True
            result.escalation_reason = "Suspected unauthorized transaction or card fraud requires immediate specialist review."
            return result

        # 2. Duplicate payment / multiple debit (HIGH/URGENT)
        duplicate_patterns = [
            r"charged twice", r"debited twice", r"double charge", r"double debit",
            r"two times", r"deducted twice", r"2 times", r"दो बार", r"రెండుసార్లు"
        ]
        if any(re.search(p, q_lower) for p in duplicate_patterns):
            result.category = TicketCategory.PAYMENT
            result.priority = TicketPriority.HIGH
            result.intent = "duplicate_charge"
            result.requires_human = True
            result.escalation_reason = "Duplicate payment requires finance team review and reversal."
            return result

        # 3. Stalled / potentially lost shipment: the supplied shipping policy
        # explicitly requires human escalation after four consecutive business
        # days with no hub movement/scan updates.
        stalled_patterns = [
            r"four consecutive business days", r"4 consecutive business days",
            r"four consecutive days", r"4 consecutive days", r"no hub movement",
            r"zero hub movement", r"stalled", r"potential lost in transit",
        ]
        if any(re.search(p, q_lower) for p in stalled_patterns):
            result.category = TicketCategory.SHIPPING
            result.priority = TicketPriority.HIGH
            result.intent = "delayed_delivery"
            result.requires_human = True
            result.escalation_reason = "Tracking shows a stalled or potentially lost shipment and requires Logistics Investigation by a human specialist."
            return result

        # 3. Legal notice, court, regulatory, consumer forum threats (URGENT)
        legal_patterns = [
            r"legal action", r"lawyer", r"consumer court", r"consumer forum",
            r"sue you", r"police complaint", r"court notice", r"कंज्यूमर कोर्ट", r"కోర్టు"
        ]
        if any(re.search(p, q_lower) for p in legal_patterns):
            result.priority = TicketPriority.URGENT
            result.requires_human = True
            result.escalation_reason = "Legal or regulatory dispute raised; immediate specialist escalation."
            return result

        # 4. Explicit customer request for human representative
        human_request_patterns = [
            r"talk to (a )?human", r"speak to (a )?human", r"connect (me )?to (an? )?agent",
            r"speak to (an? )?agent", r"transfer to (an? )?agent", r"human agent",
            r"customer care executive", r"speak to (a )?manager", r"call me",
            r"इंसान से बात", r"एजेंट से बात", r"మనిషితో మాట్లాడాలి", r"ఏజెంట్ తో మాట్లాడాలి"
        ]
        if any(re.search(p, q_lower) for p in human_request_patterns):
            result.intent = "talk_to_agent"
            result.requires_human = True
            result.escalation_reason = "Customer explicitly requested a human support specialist."
            return result

        # 5. Severe frustration or repeated failure
        frustration_patterns = [
            r"worst service", r"cheating", r"scam", r"stop wasting my time",
            r"disgusted", r"ridiculous", r"fraud company"
        ]
        if any(re.search(p, q_lower) for p in frustration_patterns):
            result.sentiment = SentimentType.FRUSTRATED
            result.requires_human = True
            result.escalation_reason = "Severe customer frustration detected; handoff to senior support."
            return result

        # 6. Completely unclear / gibberish query
        cleaned_words = [w for w in re.sub(r"[^\w\s]", "", q_lower).split() if len(w) > 1]
        if len(cleaned_words) == 0 or (len(cleaned_words) <= 2 and not any(w.isalpha() for w in cleaned_words)):
            result.category = TicketCategory.OTHER
            result.priority = TicketPriority.LOW
            result.intent = "unknown"
            result.requires_human = True
            result.escalation_reason = "Query unclear or cannot be confidently understood."
            return result

        return result

    def fallback_classify(self, query: str) -> ClassificationResult:
        """Heuristic rule-based classifier when LLM service is unavailable or errors out."""
        q_lower = query.lower()
        detected_lang = self.detect_language_heuristics(query)

        # Initial default result
        category = TicketCategory.OTHER
        priority = TicketPriority.MEDIUM
        intent = "general_question"
        sentiment = SentimentType.NEUTRAL
        requires_human = False
        escalation_reason = None

        # Category and intent heuristics
        if re.search(r"payment (?:is )?(?:pending|processing|not updating|not updated|not reflected)|payment status|paid .*payment .*pending|payment successful .*pending|money .*deducted .*pending", q_lower):
            category = TicketCategory.PAYMENT
            intent = "payment_status_pending"
        elif any(k in q_lower for k in ["twice", "deducted", "debited", "charge", "payment", "bank", "upi", "card", "money", "paise", "dabbu"]):
            category = TicketCategory.PAYMENT
            intent = "payment_inquiry"
            if "failed" in q_lower or "failure" in q_lower or "कट गए" in q_lower:
                intent = "payment_failed"
        elif any(k in q_lower for k in ["return", "refund", "exchange", "वापस", "వాపసు"]):
            category = TicketCategory.RETURN_REFUND
            if "exchange" in q_lower:
                intent = "exchange_request"
            elif "refund" in q_lower:
                intent = "refund_status"
            else:
                intent = "return_request"
        elif any(k in q_lower for k in ["ship", "delivery", "track", "courier", "late", "delay", "आया", "రాలేదు"]):
            category = TicketCategory.SHIPPING
            intent = "delayed_delivery" if any(k in q_lower for k in ["delay", "late", "రాలేదు", "नहीं आया"]) else "order_tracking"
        elif any(k in q_lower for k in ["cancel", "cancellation", "modify", "address"]):
            category = TicketCategory.ORDER
            intent = "order_cancellation" if "cancel" in q_lower else "order_modification"
        elif any(k in q_lower for k in ["password", "login", "otp", "2fa", "account", "lock"]):
            category = TicketCategory.ACCOUNT
            intent = "password_reset" if "password" in q_lower else "account_security"
        elif any(k in q_lower for k in ["coupon", "promo", "discount", "voucher", "cashback", "cash back", "bank offer"]):
            category = TicketCategory.PROMOTION
            intent = "coupon_issue"
        elif any(k in q_lower for k in ["damaged", "broken", "wrong item", "defective", "खराब"]):
            category = TicketCategory.PRODUCT
            intent = "damaged_product" if any(k in q_lower for k in ["damaged", "broken", "defective", "खराब"]) else "wrong_product"

        # Sentiment heuristics
        if any(k in q_lower for k in ["angry", "terrible", "worst", "cheat", "scam", "twice", "frustrated", "court"]):
            sentiment = SentimentType.FRUSTRATED
        elif any(k in q_lower for k in ["problem", "not working", "broken", "issue", "missing", "delay"]):
            sentiment = SentimentType.NEGATIVE
        elif any(k in q_lower for k in ["thank", "great", "good", "satisfied", "helpful"]):
            sentiment = SentimentType.POSITIVE

        base_result = ClassificationResult(
            category=category,
            priority=priority,
            intent=intent,
            sentiment=sentiment,
            language=detected_lang,
            requires_human=requires_human,
            escalation_reason=escalation_reason
        )

        # Requirement-2 deterministic analysis + safety escalation.
        base_result = self.apply_deterministic_issue_analysis(query, base_result)
        return self.apply_deterministic_escalation(query, base_result)

    @staticmethod
    def is_greeting(query: str) -> bool:
        q = re.sub(r"[^a-zA-Z\s]", "", query.lower()).strip()
        return q in {
            "hi", "hello", "hey", "hey there", "hi there", "hello there",
            "good morning", "good afternoon", "good evening", "namaste"
        }

    def classify_query(self, query: str) -> ClassificationResult:
        """Classify a customer query using Gemini with deterministic fallback and rule enforcement."""
        if query and self.is_greeting(query):
            return ClassificationResult(
                category=TicketCategory.OTHER,
                priority=TicketPriority.LOW,
                intent="greeting",
                sentiment=SentimentType.POSITIVE,
                language=self.detect_language_heuristics(query),
                requires_human=False,
                escalation_reason=None,
                analysis_source="deterministic_policy_rules",
            )

        if not query or not query.strip():
            return ClassificationResult(
                category=TicketCategory.OTHER,
                priority=TicketPriority.LOW,
                intent="unknown",
                sentiment=SentimentType.NEUTRAL,
                language="English",
                requires_human=True,
                escalation_reason="Empty query received.",
                analysis_source="deterministic_policy_rules",
            )

        # Fast path for clear support intents. These deterministic signals are
        # deliberately applied before Gemini so common requests do not wait on
        # an external LLM call. Gemini remains available for genuinely ambiguous
        # queries.
        baseline = ClassificationResult(
            category=TicketCategory.OTHER,
            priority=TicketPriority.MEDIUM,
            intent="general_question",
            sentiment=SentimentType.NEUTRAL,
            language=self.detect_language_heuristics(query),
            requires_human=False,
            escalation_reason=None,
        )
        fast_result = self.apply_deterministic_issue_analysis(query, baseline)
        fast_result = self.apply_deterministic_escalation(query, fast_result)
        if fast_result.category != TicketCategory.OTHER or fast_result.intent != "general_question" or fast_result.requires_human:
            fast_result.analysis_source = "deterministic_policy_rules"
            return fast_result

        # Attempt structured classification via Gemini only for ambiguous queries.
        prompt = f"Customer Query:\n\"\"\"{query.strip()}\"\"\"\n\nPlease analyze and classify this query."
        llm_result = None

        if self.llm_service.is_available():
            try:
                llm_result = self.llm_service.generate_structured(
                    prompt=prompt,
                    response_schema=ClassificationResult,
                    system_instruction=SYSTEM_CLASSIFICATION_INSTRUCTION
                )
            except Exception as e:
                logger.warning(f"LLM classification encountered an error: {e}. Falling back to rule-based analysis.")
                llm_result = None

        if isinstance(llm_result, ClassificationResult):
            # Ensure detected language is accurate if LLM left default
            if llm_result.language not in ["English", "Hindi", "Telugu", "Other"]:
                llm_result.language = self.detect_language_heuristics(query)
            # Requirement-2 analysis is strengthened deterministically, then safety escalation is applied.
            llm_result = self.apply_deterministic_issue_analysis(query, llm_result)
            llm_result = self.apply_deterministic_escalation(query, llm_result)
            llm_result.analysis_source = "gemini_plus_deterministic_policy_rules"
            return llm_result

        # Fallback to deterministic heuristic classifier
        fallback = self.fallback_classify(query)
        fallback.analysis_source = "deterministic_fallback_rules"
        return fallback


# Global singleton instance
_classifier_service_instance: Optional[ClassifierService] = None


def get_classifier_service() -> ClassifierService:
    """Provide singleton instance of ClassifierService."""
    global _classifier_service_instance
    if _classifier_service_instance is None:
        _classifier_service_instance = ClassifierService()
    return _classifier_service_instance
