"""Production-style local hybrid RAG for the ShopEase policy library.

The KB is the source of truth. ChromaDB provides semantic candidates and a small
lexical/answerability gate prevents vaguely related chunks from being treated as
answers. No external API is required for retrieval.
"""

import re
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions

from app.config import get_settings
from app.schemas.kb import KBSearchResult, KBSearchResponse

settings = get_settings()


class DocumentChunk:
    def __init__(self, chunk_id: str, content: str, source: str, category: str, topic: str, doc_title: str):
        self.chunk_id = chunk_id
        self.content = content
        self.source = source
        self.category = category
        self.topic = topic
        self.doc_title = doc_title

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "category": self.category,
            "topic": self.topic,
            "doc_title": self.doc_title,
            "chunk_id": self.chunk_id,
        }


class RAGService:
    COLLECTION_NAME = "shopease_knowledge_base"

    def __init__(self, kb_dir: Optional[str] = None, persist_dir: Optional[str] = None,
                 model_name: Optional[str] = None, threshold: Optional[float] = None):
        # Resolve relative paths from the project root, not the caller's current
        # working directory. This matters when pytest/uvicorn is launched from
        # the parent Downloads directory rather than the project directory.
        project_root = Path(__file__).resolve().parents[2]

        def resolve_project_path(value: str) -> Path:
            path = Path(value)
            return path if path.is_absolute() else (project_root / path).resolve()

        self.kb_dir = resolve_project_path(kb_dir or settings.kb_dir)
        self.persist_dir = resolve_project_path(persist_dir or settings.chroma_persist_dir)
        self.model_name = model_name or settings.embedding_model
        self.default_threshold = threshold if threshold is not None else settings.retrieval_threshold
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.model_name)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def parse_markdown_to_chunks(self, file_path: Path) -> List[DocumentChunk]:
        text = file_path.read_text(encoding="utf-8").strip()
        doc_title = file_path.stem.replace("_", " ").title()
        for line in text.splitlines():
            if line.startswith("# "):
                doc_title = line[2:].strip()
                break
        category = file_path.stem
        sections = list(re.finditer(r"^##\s+(.+)$", text, re.MULTILINE))
        chunks: List[DocumentChunk] = []
        if not sections:
            return [DocumentChunk(f"{file_path.stem}_full", text, file_path.name, category, doc_title, doc_title)]
        preamble = text[:sections[0].start()].strip()
        if len(preamble) > 50:
            chunks.append(DocumentChunk(f"{file_path.stem}_intro", f"Document: {doc_title}\nOverview:\n{preamble}", file_path.name, category, "Overview & Introduction", doc_title))
        for i, match in enumerate(sections):
            heading = match.group(1).strip()
            # Keep metadata user-friendly while preserving the original heading in content.
            heading = re.sub(r"^\s*\d+(?:\.\d+)?\s*[.)-]\s*", "", heading).strip()
            end = sections[i + 1].start() if i + 1 < len(sections) else len(text)
            section_content = text[match.start():end].strip()
            chunks.append(DocumentChunk(
                f"{file_path.stem}_sec_{i + 1}",
                f"Policy Document: {doc_title}\nSection: {heading}\n\n{section_content}",
                file_path.name, category, heading, doc_title,
            ))
        return chunks

    def load_all_chunks(self) -> Tuple[List[DocumentChunk], int]:
        if not self.kb_dir.exists():
            return [], 0
        files = sorted(self.kb_dir.glob("*.md"))
        chunks: List[DocumentChunk] = []
        for path in files:
            chunks.extend(self.parse_markdown_to_chunks(path))
        return chunks, len(files)

    def _kb_fingerprint(self) -> str:
        """Stable hash of authoritative Markdown content used to decide whether reindexing is needed."""
        digest = hashlib.sha256()
        for path in sorted(self.kb_dir.glob("*.md")):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    @property
    def _manifest_path(self) -> Path:
        return self.persist_dir / "kb_manifest.json"

    def _stored_fingerprint(self) -> Optional[str]:
        try:
            return json.loads(self._manifest_path.read_text(encoding="utf-8")).get("fingerprint")
        except Exception:
            return None

    def _write_manifest(self, fingerprint: str, total_documents: int, total_chunks: int) -> None:
        self._manifest_path.write_text(json.dumps({
            "fingerprint": fingerprint,
            "embedding_model": self.model_name,
            "total_documents": total_documents,
            "total_chunks": total_chunks,
        }, indent=2), encoding="utf-8")

    def index_documents(self, force_reindex: bool = False) -> Dict[str, Any]:
        chunks, total_docs = self.load_all_chunks()
        if not chunks:
            return {"status": "warning", "total_documents": 0, "total_chunks": 0, "message": f"No Markdown files found in {self.kb_dir}"}

        fingerprint = self._kb_fingerprint()
        existing_count = self.collection.count()
        if (not force_reindex and existing_count > 0 and
                self._stored_fingerprint() == fingerprint):
            return {"status": "success", "total_documents": total_docs, "total_chunks": existing_count,
                    "message": f"Knowledge base already indexed ({existing_count} chunks); no changes detected.",
                    "reindexed": False}

        try:
            self.client.delete_collection(name=self.COLLECTION_NAME)
        except Exception:
            pass
        self._collection = None

        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[c.to_metadata() for c in chunks],
        )
        indexed_count = self.collection.count()
        if indexed_count <= 0:
            return {"status": "error", "total_documents": total_docs, "total_chunks": 0,
                    "message": "Knowledge-base indexing completed without creating any ChromaDB records."}
        self._write_manifest(fingerprint, total_docs, indexed_count)
        return {"status": "success", "total_documents": total_docs, "total_chunks": indexed_count,
                "message": f"Successfully indexed {len(chunks)} chunks across {total_docs} policy documents.",
                "reindexed": True}

    @staticmethod
    def _tokens(text: str) -> set[str]:
        words = re.findall(r"[a-zA-Z0-9]+|[\u0900-\u097F]+|[\u0C00-\u0C7F]+", text.lower())
        stop = {
            "the", "a", "an", "is", "are", "am", "i", "me", "my", "to", "for", "of", "and", "or", "on", "in",
            "it", "this", "that", "what", "how", "when", "can", "do", "does", "did", "will", "would", "could",
            "should", "please", "tell", "want", "need", "about", "with", "from", "you", "your", "shop", "shopease",
            "customer", "support", "policy", "according", "regarding", "the", "within", "offer", "have", "has",
        }
        return {w for w in words if len(w) > 1 and w not in stop}

    @staticmethod
    def _is_greeting(query: str) -> bool:
        q = re.sub(r"[^a-zA-Z\s]", "", query.lower()).strip()
        return q in {"hi", "hello", "hey", "hey there", "hi there", "hello there", "good morning", "good afternoon", "good evening", "namaste"}

    def _query_for_retrieval(self, query: str, classification=None) -> str:
        """Build a semantic retrieval query from the customer wording plus analysis labels.

        This never selects a KB answer. It only gives the embedding model additional
        semantic context so paraphrases and multilingual follow-ups retrieve better.
        """
        q = query.strip()
        lower = q.lower()
        hints = []
        synonym_groups = [
            (("standard shipping", "standard delivery", "regular shipment", "regular delivery"), "standard shipping delivery"),
            (("express shipping", "express delivery"), "express shipping delivery"),
            (("same-day", "same day", "delivered today", "deliver today", "delivery today", "fastest delivery"), "same-day priority delivery"),
            (("refund",), "refund reimbursement"),
            (("return",), "return"),
            (("charged twice", "debited twice", "double charge", "two times", "double debit"), "duplicate charge payment"),
            (("delayed", "delay", "late", "has not arrived", "hasn't arrived", "stalled", "four consecutive", "hasn't moved"), "delayed delivery transit"),
            (("delivered but", "marked delivered", "not received"), "delivered non-receipt package"),
            (("coupon", "promo", "discount"), "coupon promotion discount"),
        ]
        for terms, expansion in synonym_groups:
            if any(term in lower for term in terms):
                hints.append(expansion)
        # Analysis labels are retrieval hints, not policy answers.
        if classification is not None:
            category_value = getattr(classification, "category", "")
            intent_value = getattr(classification, "intent", "")
            category = str(getattr(category_value, "value", category_value)).replace("_", " ")
            intent = str(getattr(intent_value, "value", intent_value)).replace("_", " ")
            if category and category.lower() != "other":
                hints.append(f"support category {category}")
            if intent and intent.lower() not in {"general question", "unknown", "greeting"}:
                hints.append(f"customer intent {intent}")
        # For non-English text, add broad English domain vocabulary as an embedding aid.
        if re.search(r"[\u0900-\u097F]", q):
            hints.append("ShopEase customer support policy")
        elif re.search(r"[\u0C00-\u0C7F]", q):
            hints.append("ShopEase customer support policy")
        return q + ("\n" + " ".join(dict.fromkeys(hints)) if hints else "")

    def _looks_like_shop_support_query(self, query: str) -> bool:
        q = query.lower()
        terms = [
            "shopease", "order", "payment", "paid", "pay", "card", "upi", "bank", "refund", "return", "exchange",
            "delivery", "shipping", "ship", "shipment", "courier", "tracking", "track", "package", "parcel", "cancel", "address",
            "account", "password", "login", "otp", "2fa", "coupon", "promo", "discount", "voucher", "cashback",
            "damaged", "broken", "wrong item", "defective", "invoice", "billing", "charge", "charged", "debited", "money",
            "delay", "late", "support", "policy", "status", "warranty", "wallet", "रिफंड", "भुगतान", "ऑर्डर", "डिलीवरी",
            "पार्सल", "रिटर्न", "రీఫండ్", "చెల్లింపు", "ఆర్డర్", "డెలివరీ", "పార్సెల్", "రాలేదు", "ఆలస్యం",
        ]
        return any(term in q for term in terms)

    @staticmethod
    def _lexical_score(query: str, content: str, topic: str) -> float:
        q = RAGService._tokens(query)
        if not q:
            return 0.0
        c = RAGService._tokens(content + " " + topic)
        overlap = len(q & c) / max(1, len(q))
        topic_overlap = len(q & RAGService._tokens(topic)) / max(1, len(q))
        phrase_bonus = 0.0
        ql, cl = query.lower(), content.lower()
        for phrase in ["standard shipping", "standard delivery", "express shipping", "express delivery", "same-day", "same day", "duplicate charge", "return window", "refund", "cashback", "delayed delivery"]:
            if phrase in ql and phrase in cl:
                phrase_bonus += 0.18
        return min(1.0, overlap * 0.55 + topic_overlap * 0.30 + phrase_bonus)

    @staticmethod
    def _focus_query(query: str) -> str:
        """Extract the focused query if the query contains conversation prompt headers."""
        if "Follow-up customer question:" in query:
            return query.split("Follow-up customer question:")[-1].strip()
        if "Follow-up:" in query:
            return query.split("Follow-up:")[-1].strip()
        return query

    @staticmethod
    def _topic_affinity(query: str, topic: str, content: str) -> float:
        """Deterministic intent-to-topic boost layered on semantic retrieval.

        This is routing logic, not an answer generator: it only tells the
        reranker which supplied KB section best matches a support intent.
        """
        focused = RAGService._focus_query(query)
        q = RAGService._normalize(focused)
        full_q = RAGService._normalize(query)
        t = RAGService._normalize(topic)
        c = RAGService._normalize(content)
        rules = [
            (("standard delivery", "standard shipping", "regular shipment", "regular delivery"), "shipping tiers and service timelines"),
            (("express delivery", "express shipping"), "shipping tiers and service timelines"),
            (("same-day", "same day", "delivered today", "deliver today", "delivery today", "fastest delivery"), "shipping tiers and service timelines"),
            (("delayed", "delay", "late", "hasn't arrived", "has not arrived", "stalled", "no hub movement", "hasn't moved", "has not moved", "no movement", "four consecutive", "4 consecutive"), "delayed deliveries and escalation protocol"),
            (("track", "tracking", "awb", "where is my order"), "order tracking and status definitions"),
            (("delivered but", "not received", "nothing reached"), "package marked delivered but not received"),
            (("return policy", "return window", "return after", "eligible for return"), "return window and eligibility overview"),
            (("return condition", "packaging", "tags attached", "unworn"), "product condition requirements for returns"),
            (("exchange", "different size", "different color"), "product exchanges"),
            (("how do i return", "return pickup", "request a return"), "return request and pickup procedure"),
            (("refund", "money back", "card refund", "upi refund"), "inspection and refund timelines"),
            (("return shipping", "return fee", "shipping fee"), "return shipping charges"),
            (("charged twice", "billed twice", "debited twice", "double charge", "duplicate", "two times", "double debit", "multiple debits"), "duplicate charges and double deductions"),
            (("payment pending", "payment status", "payment processing", "paid but", "money deducted but"), "failed payments and automated reversals"),
            (("payment failed", "failed payment", "gateway timeout"), "failed payments and automated reversals"),
            (("unauthorized", "fraud", "stolen card"), "unauthorized transactions and card fraud"),
            (("coupon", "promo code", "discount code", "voucher"), "applying discount promo codes"),
            (("cashback", "bank offer", "payment partner"), "bank and payment partner offers"),
            (("cancel",), "order cancellation window and eligibility"),
            (("cancellation refund",), "cancellation refund processing"),
            (("change address", "modify order"), "order modifications and address changes"),
            (("password", "login", "sign in"), "password reset and login assistance"),
            (("2fa", "otp", "verification code"), "two-factor authentication"),
            (("account hacked", "account compromised", "security issue"), "suspicious account activity and security lockdown"),
            (("wrong product", "wrong item"), "wrong product dispatched by shopease"),
            (("damaged", "broken", "defective"), "damaged or defective goods resolution"),
            (("support hours", "when is support open"), "customer support operating hours and languages"),
        ]
        score = 0.0
        for phrases, expected in rules:
            if any(p in q for p in phrases) and expected in t:
                score = max(score, 1.0)
        # If the focused follow-up didn't match a rule, check full_q so previous topic context is retained
        if score == 0.0 and q != full_q:
            for phrases, expected in rules:
                if any(p in full_q for p in phrases) and expected in t:
                    score = max(score, 1.0)
        # Secondary evidence overlap helps for paraphrases not covered above.
        if score == 0.0:
            q_tokens = RAGService._tokens(full_q)
            topic_tokens = RAGService._tokens(t)
            content_tokens = RAGService._tokens(c)
            score = 0.5 * len(q_tokens & topic_tokens) / max(1, len(q_tokens))
            score += 0.2 * len(q_tokens & content_tokens) / max(1, len(q_tokens))
        return min(1.0, score)

    @staticmethod
    def _specificity_gate(query: str, content: str) -> bool:
        """Reject evidence that cannot support a concrete claim in the question.

        Numeric identifiers such as order IDs are ignored. Customer-facing
        amounts/percentages/time values are treated as claims and must be
        supported by the retrieved evidence. Comparative questions such as
        "after 90 days" are allowed to retrieve the applicable policy window
        even though the comparison number itself is not a policy value.
        """
        q = query.lower()
        c = content.lower()

        # Ignore likely identifiers (order IDs, UTR/RRN/transaction IDs).
        numeric_matches = list(re.finditer(r"(?<![a-z])(?:₹\s*)?\d[\d,]*(?:\.\d+)?\s*(?:%|percent|days?|hours?|business\s+days?|business\s+hours?|minutes?|minutes|rs\.?|rupees?)?", q))
        requested_claim_numbers = set()
        for match in numeric_matches:
            raw = match.group(0)
            digits_match = re.search(r"\d+(?:\.\d+)?", raw)
            if not digits_match:
                continue
            value = digits_match.group(0).replace(",", "")
            context = raw.lower()
            # A bare 4+ digit number is overwhelmingly likely to be an ID in
            # support queries. Currency/percentage/time units remain claims.
            if re.search(r"[₹%]|percent|days?|hours?|minutes?|rs\.?|rupees?", context):
                requested_claim_numbers.add(value)
            elif float(value) < 1000:
                requested_claim_numbers.add(value)

        evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", c))
        comparative_return = bool(
            re.search(r"\b(after|beyond|over|more than|exceed|past|within)\s+\d+\b", q)
            and "return" in q
        )
        if requested_claim_numbers and not requested_claim_numbers.issubset(evidence_numbers):
            if not comparative_return:
                return False

        required_terms = {
            "maximum": ["maximum", "max", "limit", "highest"],
            "minimum": ["minimum", "min", "least"],
            "percentage": ["percentage", "%", "percent"],
            "percent": ["percentage", "%", "percent"],
            "price": ["price", "cost", "₹", "surcharge", "fee", "charge"],
            "cost": ["cost", "₹", "surcharge", "fee", "charge"],
            "surcharge": ["surcharge"],
            "cutoff": ["cutoff", "before"],
        }
        q_tokens = RAGService._tokens(query)
        for requested, evidence_terms in required_terms.items():
            if requested in q_tokens and not any(term in c for term in evidence_terms):
                return False

        # A percentage/rate requested with a concrete value must be explicitly
        # present; this blocks unsupported claims such as "5% cashback".
        if re.search(r"\b\d+(?:\.\d+)?\s*%", q) or re.search(r"\b\d+(?:\.\d+)?\s*percent\b", q):
            if not re.search(r"\b\d+(?:\.\d+)?\s*%", c) and not re.search(r"\b\d+(?:\.\d+)?\s*percent\b", c):
                return False

        return True

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().replace("‑", "-").replace("–", "-").replace("—", "-")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _has_unsupported_explicit_claim(query: str, evidence: str) -> bool:
        """Return True when a concrete numeric claim is not supported anywhere in the KB.

        Order IDs/transaction IDs are ignored. Comparative questions such as
        "after 20 days" can still retrieve the policy that defines the return
        window; the customer's comparison number does not need to be a policy value.
        """
        q = query.lower()
        evidence = evidence.lower()
        claims = re.findall(
            r"(?:₹\s*|rs\.?\s*|rupees\s*)?[0-9][0-9,]*(?:\.\d+)?\s*(?:%|percent|days?|hours?|business\s+days?|business\s+hours?|minutes?)",
            q,
        )
        for raw in claims:
            number = re.search(r"[0-9][0-9,]*(?:\.\d+)?", raw)
            if not number:
                continue
            value = number.group(0).replace(",", "")
            raw_lower = raw.lower()
            # Rates/percentages and currency amounts must be supported with the
            # same kind of claim, not merely by the same number appearing in an
            # unrelated policy (e.g. 5% must not match an unrelated "5 to 7 days").
            if "%" in raw_lower or "percent" in raw_lower:
                if re.search(r"\b" + re.escape(value) + r"\s*(?:%|percent)\b", evidence):
                    continue
            elif any(currency in raw_lower for currency in ("₹", "rs", "rupees")):
                if re.search(r"(?:₹|rs\.?|rupees)\s*" + re.escape(value) + r"\b", evidence):
                    continue
            else:
                if re.search(r"\b" + re.escape(value) + r"\s*(?:days?|hours?|business\s+days?|business\s+hours?|minutes?)\b", evidence):
                    continue
            if re.search(r"\b(after|beyond|over|more than|exceed|past|within)\s+" + re.escape(value) + r"\b", q) and "return" in q:
                continue
            return True
        return False

    def search(self, query: str, top_k: int = 3, threshold: Optional[float] = None, classification=None) -> KBSearchResponse:
        """Retrieve authoritative KB evidence using semantic search plus intent-aware reranking.

        ``classification`` is optional for backwards compatibility.  When supplied,
        its category/intent is used only to improve ranking/filtering; it never
        supplies an answer or policy fact.
        """
        if not query or not query.strip():
            return KBSearchResponse(query=query, results_count=0, has_sufficient_context=False, results=[])
        if self._is_greeting(query):
            return KBSearchResponse(query=query, results_count=0, has_sufficient_context=False, results=[])
        if not self._looks_like_shop_support_query(query):
            return KBSearchResponse(query=query, results_count=0, has_sufficient_context=False, results=[])

        # A query explicitly classified as needing clarification must not be
        # turned into an arbitrary KB answer by semantic similarity.
        if classification is not None and getattr(classification, "intent", "") in {"needs_clarification", "cod_payment_mismatch", "replacement_status"}:
            return KBSearchResponse(query=query, results_count=0, has_sufficient_context=False, results=[])

        # Concrete claims are checked against the authoritative Markdown corpus
        # before semantic retrieval can promote a vaguely related chunk.
        chunks, _ = self.load_all_chunks()
        all_evidence = "\n".join(c.content.lower() for c in chunks)
        if self._has_unsupported_explicit_claim(query, all_evidence):
            return KBSearchResponse(query=query, results_count=0, has_sufficient_context=False, results=[])

        retrieval_query = self._query_for_retrieval(query, classification=classification)

        # ChromaDB rejects n_results=0. More importantly, an empty collection
        # means the application is not ready to answer from the KB. Recover by
        # indexing the authoritative Markdown files once, then fail safely if
        # indexing still produced no records.
        collection_count = self.collection.count()
        if collection_count <= 0:
            self.index_documents(force_reindex=False)
            collection_count = self.collection.count()
        if collection_count <= 0:
            return KBSearchResponse(query=query, results_count=0, has_sufficient_context=False, results=[])

        candidate_k = min(max(int(top_k) * 8, 24), collection_count)
        candidate_k = max(1, candidate_k)
        result = self.collection.query(query_texts=[retrieval_query], n_results=candidate_k)
        candidates: List[KBSearchResult] = []
        if result.get("documents") and result["documents"][0]:
            docs = result["documents"][0]
            metas = result["metadatas"][0]
            distances = result.get("distances", [[0.0] * len(docs)])[0]
            ids = result["ids"][0]
            for doc, meta, dist, cid in zip(docs, metas, distances, ids):
                semantic = max(0.0, min(1.0, 1.0 - float(dist) / 2.0))
                lexical = self._lexical_score(query, doc, meta.get("topic", ""))
                topic_affinity = self._topic_affinity(query, meta.get("topic", ""), doc)
                intent_affinity = 0.0
                category_affinity = 0.0
                if classification is not None:
                    intent_value = getattr(classification, "intent", "") or ""
                    intent = str(getattr(intent_value, "value", intent_value)).replace("_", " ").lower()
                    intent_tokens = set(intent.split())
                    topic_tokens = RAGService._tokens(meta.get("topic", ""))
                    content_tokens = RAGService._tokens(doc)
                    intent_affinity = min(1.0, len(intent_tokens & (topic_tokens | content_tokens)) / max(1, len(intent_tokens)))
                    category_value = getattr(classification, "category", "") or ""
                    category = str(getattr(category_value, "value", category_value)).replace("_", " ").lower()
                    source = str(meta.get("source", "")).lower()
                    category_aliases = {
                        "payment": ("payment", "billing"),
                        "return refund": ("return", "refund"),
                        "shipping": ("shipping", "delivery"),
                        "order": ("cancellation", "modification"),
                        "account": ("account", "security"),
                        "promotion": ("promotion", "coupon"),
                        "product": ("product", "damaged"),
                    }
                    aliases = category_aliases.get(category, ())
                    category_affinity = 1.0 if any(alias in source or alias in str(meta.get("category", "")).lower() for alias in aliases) else 0.0
                hybrid = semantic * 0.55 + lexical * 0.15 + topic_affinity * 0.15 + intent_affinity * 0.10 + category_affinity * 0.05
                candidates.append(KBSearchResult(
                    content=doc,
                    source=meta.get("source", "unknown"),
                    category=meta.get("category", "general"),
                    topic=meta.get("topic", "general"),
                    chunk_id=cid,
                    distance=round(float(dist), 4),
                    similarity_score=round(hybrid, 4),
                    metadata={**meta, "semantic_similarity": round(semantic, 4), "lexical_score": round(lexical, 4), "topic_affinity": round(topic_affinity, 4), "retrieval_method": "chroma_semantic_hybrid"},
                ))

        valid = []
        active_threshold = threshold if threshold is not None else self.default_threshold
        semantic_floor = max(0.62, 1.0 - (active_threshold / 2.0))
        for candidate in candidates:
            if not self._specificity_gate(query, candidate.content):
                continue
            semantic = float(candidate.metadata.get("semantic_similarity", 0.0))
            lexical = float(candidate.metadata.get("lexical_score", 0.0))
            affinity = float(candidate.metadata.get("topic_affinity", 0.0))
            if semantic >= semantic_floor or (semantic >= 0.56 and (lexical >= 0.12 or affinity >= 0.5)):
                valid.append(candidate)

        valid.sort(key=lambda x: x.similarity_score, reverse=True)
        if not valid:
            return KBSearchResponse(query=query, results_count=0, has_sufficient_context=False, results=[])
        return KBSearchResponse(query=query, results_count=min(top_k, len(valid)), has_sufficient_context=True, results=valid[:top_k])

    def get_status(self) -> Dict[str, Any]:
        count = self.collection.count()
        return {"is_indexed": count > 0, "total_chunks": count, "collection_name": self.COLLECTION_NAME,
                "persist_directory": str(self.persist_dir), "embedding_model": self.model_name,
                "threshold": self.default_threshold, "kb_fingerprint": self._stored_fingerprint()}


_rag_service_instance: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance
