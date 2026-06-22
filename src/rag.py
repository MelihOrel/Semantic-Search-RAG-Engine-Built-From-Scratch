"""RAG capstone — the generation layer, built by hand (no agent framework).

``answer_question`` retrieves top-k chunks with the chosen retriever, assembles
a grounded prompt, calls an LLM, and returns an answer with inline citations
back to specific ``complaint_id`` values. A confidence guardrail makes the
system refuse to answer when retrieval is weak, rather than hallucinate.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .retrievers import BaseRetriever, SearchResult
from .utils import get_logger

SYSTEM_PROMPT = (
    "You are a careful analyst of U.S. consumer-finance complaints. Answer the "
    "user's question USING ONLY the provided complaint excerpts. Cite every "
    "claim inline with the complaint id in square brackets, e.g. [complaint_id: "
    "123456]. If the excerpts do not support an answer, say you cannot answer "
    "from the available complaints. Do not invent facts or ids."
)


@dataclass
class RAGAnswer:
    query: str
    answer: str
    citations: List[str]
    confidence: float
    refused: bool = False
    contexts: List[SearchResult] = field(default_factory=list)


def _build_context(results: List[SearchResult], max_chars: int) -> str:
    blocks, used = [], 0
    for r in results:
        block = f"[complaint_id: {r.complaint_id}] (score={r.score:.3f})\n{r.text}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks)


def _call_anthropic(model: str, system: str, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=model,
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _call_openai(model: str, system: str, user: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def answer_question(
    query: str,
    retriever: BaseRetriever,
    config: Dict[str, Any],
) -> RAGAnswer:
    """Retrieve, guard, and generate a grounded, cited answer."""
    log = get_logger("rag", config.get("logging", {}).get("level", "INFO"))
    rcfg = config["rag"]
    results = retriever.search(query, rcfg["top_k"])

    # Confidence guardrail. Only cosine-scored retrievers have a meaningful
    # absolute threshold; for unbounded scores (BM25, RRF) we can only check
    # that something was retrieved at all (the corpus is the sole knowledge
    # source, so an empty/zero result is the honest "cannot answer" signal).
    score_kind = getattr(retriever, "score_kind", "unbounded")
    top_score = results[0].score if results else 0.0
    if score_kind == "cosine":
        confident = bool(results) and top_score >= rcfg["min_confidence"]
    else:
        confident = bool(results) and top_score > 0.0
    confidence = top_score

    if not confident:
        log.info("Low retrieval confidence (%.3f, %s) — refusing to answer", confidence, score_kind)
        return RAGAnswer(
            query=query,
            answer=(
                "I cannot answer this question from the available complaint "
                f"corpus (top retrieval confidence {confidence:.2f} below "
                f"threshold {rcfg['min_confidence']:.2f})."
            ),
            citations=[],
            confidence=confidence,
            refused=True,
            contexts=results,
        )

    context = _build_context(results, rcfg["max_context_chars"])
    user_prompt = f"Question: {query}\n\nComplaint excerpts:\n{context}\n\nAnswer:"

    provider = rcfg["provider"]
    have_key = (provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY")) or (
        provider == "openai" and os.environ.get("OPENAI_API_KEY")
    )
    if not have_key:
        # Deterministic offline fallback so the pipeline runs without a key:
        # surface the grounded contexts and cite them, but make clear no LLM ran.
        cited = sorted({r.complaint_id for r in results})
        answer = (
            "[LLM key not set — returning retrieved evidence without generation.] "
            "The most relevant complaints to this question are: "
            + ", ".join(f"[complaint_id: {c}]" for c in cited)
            + ". Set ANTHROPIC_API_KEY or OPENAI_API_KEY to generate a synthesised answer."
        )
        return RAGAnswer(query, answer, cited, confidence, False, results)

    if provider == "anthropic":
        text = _call_anthropic(rcfg["anthropic_model"], SYSTEM_PROMPT, user_prompt)
    else:
        text = _call_openai(rcfg["openai_model"], SYSTEM_PROMPT, user_prompt)

    cited = [r.complaint_id for r in results if f"complaint_id: {r.complaint_id}" in text]
    return RAGAnswer(query, text, cited, confidence, False, results)


# Worked example questions for the capstone demo.
EXAMPLE_QUESTIONS = [
    "What debt-collection tactics do consumers most commonly report?",
    "What problems do consumers describe with incorrect information on their credit reports?",
    "How do companies typically respond when consumers dispute a debt they say they do not owe?",
]
