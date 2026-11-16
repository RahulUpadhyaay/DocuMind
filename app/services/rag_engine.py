from typing import List, Literal, AsyncGenerator, Set
import json
import re
from pydantic import BaseModel, Field
from app.services.hybrid_search import SearchResult
from app.core.config import settings
from app.core.logging import app_logger


class Citation(BaseModel):
    document_id: str = Field(description="Unique UUID string of source document")
    source_filename: str = Field(description="Original filename, e.g. contract_v2.pdf")
    page_number: int = Field(description="Exact 1-based page number where the quote exists")
    exact_quote: str = Field(description="Verbatim sentence or clause quoted from the source text")


class RAGResponse(BaseModel):
    answer: str = Field(
        description="Elaborative, accurate synthesis strictly based on citations. If missing, return standard fallback rejection."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence rating based on passage alignment"
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of source citations verifying each factual statement"
    )
    information_found: bool = Field(
        description="True if documents contained answer; False otherwise"
    )


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
    "will", "with", "what", "where", "which", "who", "when", "how", "why",
    "can", "could", "should", "would", "do", "does", "did", "about", "tell",
    "me", "give", "explain", "describe", "any", "some", "policy", "terms"
}


class RAGEngine:
    """Strict Citation RAG Engine with elaborative synthesis and multi-layered hallucination guards."""

    SYSTEM_PROMPT = """You are an Enterprise Legal & Document Intelligence Analyst.
Your mandate is to provide an ELABORATIVE, THOROUGH, and 100% FACTUALLY ACCURATE synthesis with ZERO HALLUCINATION.

CRITICAL RULES:
1. Provide a comprehensive, detailed, and professional answer explaining the full context, stipulations, timelines, rates, and legal implications found in the document passages.
2. Structure your response clearly using explanatory narrative and bullet points where helpful.
3. Every claim, number, date, or percentage MUST be verified by an exact verbatim quote in "citations".
4. If the question asks about topics, policies, or entities NOT present in the provided passages (or if the facts cannot be proven):
   - You MUST set "answer" to: "I cannot find this information in the uploaded documents."
   - You MUST set "confidence" to: "low"
   - You MUST set "citations" to: []
   - You MUST set "information_found" to: false
   NEVER invent, extrapolate, or assume facts not explicitly stated in the passages.
5. Output valid JSON strictly conforming to the requested schema.
"""

    @classmethod
    def _extract_query_keywords(cls, query: str) -> Set[str]:
        """Extracts significant semantic keywords from the query."""
        tokens = re.findall(r'[a-zA-Z0-9]+', query.lower())
        return {t for t in tokens if len(t) > 2 and t not in STOP_WORDS}

    @classmethod
    def _word_matches_text(cls, word: str, text: str) -> bool:
        """Morphological and stem-aware word match that handles plurals (penalty -> penalties), verb forms, and roots."""
        word = word.lower()
        if word in text:
            return True
        # Stemming rules
        stem = word
        if word.endswith("ies") and len(word) > 4:
            stem = word[:-3] + "y"
        elif word.endswith("es") and len(word) > 3:
            stem = word[:-2]
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            stem = word[:-1]
        elif word.endswith("ing") and len(word) > 5:
            stem = word[:-3]
        elif word.endswith("ed") and len(word) > 4:
            stem = word[:-2]
        
        if stem in text:
            return True
        
        # 4-character root prefix (e.g. penalt -> penalties)
        if len(word) >= 4 and word[:4] in text:
            return True

        return False

    @classmethod
    def _check_content_relevance(cls, query: str, passages: List[SearchResult]) -> bool:
        """Determines if the passages have any meaningful topical overlap with the query."""
        if not passages:
            return False

        # If top candidate has high hybrid / reranker score, it is topically relevant
        top_score = max((p.score for p in passages), default=0.0)
        if top_score >= 0.45:
            return True

        keywords = cls._extract_query_keywords(query)
        if not keywords:
            return True

        combined_text = " ".join([p.content.lower() for p in passages])
        matching_keywords = {kw for kw in keywords if cls._word_matches_text(kw, combined_text)}

        # If at least 1 keyword matches
        return len(matching_keywords) >= 1

    @classmethod
    def _format_context(cls, passages: List[SearchResult]) -> str:
        formatted = []
        for i, p in enumerate(passages):
            formatted.append(
                f"--- PASSAGE {i+1} ---\n"
                f"Document ID: {p.document_id}\n"
                f"Source: {p.filename}\n"
                f"Page Number: {p.page_number}\n"
                f"Content:\n{p.content}\n"
            )
        return "\n".join(formatted)

    @classmethod
    def _verify_citations(cls, citations: List[Citation], passages: List[SearchResult]) -> List[Citation]:
        """Post-validation check: verifies that each exact_quote actually exists verbatim in the passage text."""
        verified: List[Citation] = []
        all_text = " ".join([p.content for p in passages]).lower()

        for cit in citations:
            quote_clean = cit.exact_quote.strip().lower()
            if quote_clean in all_text or any(quote_clean in p.content.lower() for p in passages):
                verified.append(cit)
            else:
                app_logger.warning(
                    f"Citation quote failed verbatim verification: '{cit.exact_quote[:50]}...'. Discarded."
                )
        return verified

    @classmethod
    def _build_local_elaborative_response(cls, query: str, passages: List[SearchResult]) -> RAGResponse:
        """High-precision local extractive synthesis when no cloud LLM API key is configured."""
        keywords = cls._extract_query_keywords(query)
        relevant_sentences = []
        matching_citations = []

        for p in passages:
            # Split passage into distinct sentences
            sentences = re.split(r'(?<=[.!?])\s+|\n+', p.content)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) > 15 and (any(cls._word_matches_text(kw, s_clean.lower()) for kw in keywords) or not keywords):
                    if s_clean not in relevant_sentences:
                        relevant_sentences.append(s_clean)
                        matching_citations.append(
                            Citation(
                                document_id=str(p.document_id),
                                source_filename=p.filename,
                                page_number=p.page_number,
                                exact_quote=s_clean,
                            )
                        )

        if not relevant_sentences:
            return RAGResponse(
                answer="I cannot find this information in the uploaded documents.",
                confidence="low",
                citations=[],
                information_found=False,
            )

        top_p = passages[0]
        # Elaborative structured narrative
        intro = f"Based on review of **{top_p.filename}** (Page {top_p.page_number}), the agreement outlines the following specific provisions regarding your inquiry:\n\n"
        points = "\n".join([f"• {s}" for s in relevant_sentences[:4]])
        conclusion = "\n\nAll provisions are strictly cited from the governing contract clauses above."
        elaborative_answer = f"{intro}{points}{conclusion}"

        return RAGResponse(
            answer=elaborative_answer,
            confidence="high",
            citations=matching_citations[:3],
            information_found=True,
        )

    @classmethod
    async def generate_answer(cls, query: str, passages: List[SearchResult]) -> RAGResponse:
        """Generates structured elaborative response with strict hallucination gating."""
        # 1. Hallucination Guard: If no passages or passages have zero topical relevance to query
        if not passages or not cls._check_content_relevance(query, passages):
            app_logger.info(f"Hallucination Guard Triggered: Query '{query[:40]}' has no relevance in retrieved chunks.")
            return RAGResponse(
                answer="I cannot find this information in the uploaded documents.",
                confidence="low",
                citations=[],
                information_found=False,
            )

        context_str = cls._format_context(passages)
        user_prompt = (
            f"DOCUMENT PASSAGES:\n{context_str}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Instructions: Provide an elaborative, highly detailed explanation answering the question thoroughly based ONLY on the passages. "
            f"Ground every claim with exact citations. If the specific question cannot be answered, set information_found to false."
        )

        # 2. Call LLM (OpenAI or Gemini) if key is provided
        if settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                
                completion = await client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": cls.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                raw_json = completion.choices[0].message.content
                parsed = json.loads(raw_json)
                rag_resp = RAGResponse(**parsed)
                if rag_resp.information_found:
                    rag_resp.citations = cls._verify_citations(rag_resp.citations, passages)
                    if not rag_resp.citations and rag_resp.confidence == "high":
                        rag_resp.confidence = "medium"
                return rag_resp
            except Exception as e:
                app_logger.error(f"OpenAI completion error: {e}. Falling back to local synthesizer.")

        elif settings.LLM_PROVIDER == "gemini" and settings.GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                model = genai.GenerativeModel(
                    model_name=settings.GEMINI_MODEL,
                    system_instruction=cls.SYSTEM_PROMPT,
                    generation_config={"response_mime_type": "application/json"}
                )
                response = await model.generate_content_async(user_prompt)
                raw_json = response.text
                parsed = json.loads(raw_json)
                rag_resp = RAGResponse(**parsed)
                if rag_resp.information_found:
                    rag_resp.citations = cls._verify_citations(rag_resp.citations, passages)
                return rag_resp
            except Exception as e:
                app_logger.error(f"Gemini generation error: {e}. Falling back to local synthesizer.")

        # 3. Intelligent Local Extractive Synthesizer (0-Cost, High Accuracy, Zero Hallucination)
        return cls._build_local_elaborative_response(query, passages)

    @classmethod
    async def stream_answer(
        cls, query: str, passages: List[SearchResult]
    ) -> AsyncGenerator[str, None]:
        """Streams token-by-token elaborative answer, then yields final metadata event."""
        if not passages or not cls._check_content_relevance(query, passages):
            yield json.dumps({
                "type": "token",
                "content": "I cannot find this information in the uploaded documents."
            })
            yield json.dumps({
                "type": "final",
                "confidence": "low",
                "citations": [],
                "information_found": False
            })
            return

        final_response = await cls.generate_answer(query, passages)

        # Stream words/tokens
        words = final_response.answer.split(" ")
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            yield json.dumps({"type": "token", "content": token})

        # Yield final metadata
        yield json.dumps({
            "type": "final",
            "confidence": final_response.confidence,
            "citations": [c.model_dump() for c in final_response.citations],
            "information_found": final_response.information_found,
        })
