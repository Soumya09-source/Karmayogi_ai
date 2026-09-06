"""
MCQ generation pipeline.

Per chunk:
  1. Extract candidate concepts (Ollama, JSON mode).
  2. Match each against concept_taxonomy via embedding similarity.
       - confident match  -> use that canonical_concept_id
       - no confident match -> log to concept_review_queue, skip generation
         for that concept (never auto-create a new canonical concept)
  3. For each matched concept, generate a difficulty-varied batch of MCQs
     sized by the LLM's own "breadth" rating of the concept.
  4. Run a self-consistency check per MCQ (independent re-derivation of the
     answer from the same chunk) and store the resulting confidence_score.
  5. Insert MCQs with status="live" — no pre-publish gate, per the
     reactive-flagging design already used elsewhere in this project.

No table is ever wiped or overwritten here — everything is a fresh INSERT
with a new UUID, so this is safe to run repeatedly and safe to run
alongside any pre-existing seeded/manual rows in `mcqs`.
"""

from __future__ import annotations  # keeps `str | None` style hints safe on
                                     # Python <3.10 too, since annotations
                                     # are then never evaluated at import time

import logging
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models.concept import ConceptTaxonomy
from app.models.concept_review_queue import ConceptReviewQueue
from app.models.document_chunk import DocumentChunk
from app.models.mcq import MCQ
from app.models.mcq_generation_log import McqGenerationLog
from app.services import ollama_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Same embedding model used in embed_and_load.py — MUST match, since we're
# comparing against vectors already stored via that model. Using a
# different model here would make similarity scores meaningless.
def _coerce_to_list(result, context: str) -> list:
    """
    Ollama's format="json" guarantees valid JSON, but not the requested
    SHAPE — local models frequently wrap a requested array in an object
    (e.g. {"concepts": [...]}) or, when there's only one item, return that
    one item as a bare object instead of a single-element array. This
    normalizes the common real-world variations instead of failing on them.
    """
    if isinstance(result, list):
        return result

    if isinstance(result, dict):
        # a single-item result returned as a bare object, e.g.
        # {"name": "...", "suggested_domain": "...", "breadth": "..."}
        # or {"question": "...", "options": [...], ...}
        if any(k in result for k in ("name", "question")):
            return [result]

        # wrapped under a key, e.g. {"concepts": [...]} or {"mcqs": [...]}
        # or {"questions": [...]} — take the first list-valued key found
        for value in result.values():
            if isinstance(value, list):
                return value

    raise ValueError(
        f"Could not coerce Ollama's {context} output into a list. "
        f"Got: {type(result)} -> {str(result)[:300]}"
    )


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model = None  # lazy-loaded singleton, avoid reloading per call


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


# Confidence threshold for concept matching against concept_taxonomy.
# Cosine similarity, not distance (higher = more similar). Calibrated
# empirically against real extraction output (not a guess): with the
# domain-enriched query text above, genuine matches (e.g. "GDP Base Year"
# -> GDP) should land meaningfully higher than true non-matches (e.g.
# "Publicity Activities", which scored ~0.22 with the bare-name query and
# has no real counterpart in the taxonomy). Re-check this value against a
# larger sample once more chunks have run — this is a starting point, not
# a final answer.
CONFIDENT_MATCH_THRESHOLD = 0.55

# Breadth rating (from the LLM's own extraction pass) -> how many MCQs to
# generate and how they should be split across difficulty levels. This is
# a heuristic, not a precise science — a concept the LLM judges "broad"
# (e.g. "National Accounts Statistics") reasonably warrants more coverage
# than a narrow one (e.g. "4th Economic Census").
MCQ_PLAN_BY_BREADTH = {
    "simple":   {"easy": 2, "medium": 2, "hard": 1},   # 5 total
    "moderate": {"easy": 3, "medium": 3, "hard": 2},   # 8 total
    "broad":    {"easy": 4, "medium": 5, "hard": 3},   # 12 total
}
DEFAULT_BREADTH = "moderate"  # fallback if the LLM omits/mis-formats this field


# ---------------------------------------------------------------------
# Step 1: concept extraction
# ---------------------------------------------------------------------

def extract_concepts_from_chunk(chunk_text: str) -> list[dict]:
    """
    Returns a list of dicts: [{"name": str, "suggested_domain": str,
    "breadth": "simple"|"moderate"|"broad"}, ...]

    Ollama is prompted to return ONLY JSON. Malformed output is caught by
    ollama_client.generate_json and re-raised as ValueError — callers
    should catch and log-and-skip rather than crash the whole batch run.
    """
    prompt = f"""You are analyzing a passage from an official Indian government
statistics training/methodology document. Identify the distinct statistical,
technical, or governance CONCEPTS this passage teaches or explains.

Return ONLY a JSON array, no other text, in this exact shape:
[
  {{"name": "<short concept name, 2-6 words>", "suggested_domain": "<one of: Statistical, Technical, Digital Governance, Behavioural & Managerial, Administrative/Governance>", "breadth": "<simple|moderate|broad>"}}
]

"breadth" means: "simple" = one narrow fact/definition, "moderate" = a
standard topic with a few sub-aspects, "broad" = a wide topic covering many
sub-topics. If the passage covers no clear teachable concept (e.g. it's a
title page, table of contents, or pure boilerplate), return an empty array [].

Passage:
\"\"\"{chunk_text}\"\"\"
"""
    result = ollama_client.generate_json(prompt)
    return _coerce_to_list(result, context="concept extraction")


# ---------------------------------------------------------------------
# Step 2: match extracted concept against concept_taxonomy
# ---------------------------------------------------------------------

def match_concept_to_taxonomy(
    db: Session, raw_concept_name: str, suggested_domain: str | None = None
) -> tuple[str | None, float]:
    """
    Returns (canonical_concept_id or None, best_similarity_score).

    IMPORTANT: concept_taxonomy.embedding_text is a full sentence
    ("<name>. Domain: <domain>. Competency area: <area>. Also known as:
    ...") — embedding a bare 2-6 word concept name against that
    systematically deflates cosine similarity (short-vs-long-text
    asymmetry). Building the query in the same "<name>. Domain: <domain>."
    shape closes most of that gap, using the domain the LLM already
    supplied during extraction.
    """
    model = get_embedding_model()
    query_text = raw_concept_name
    if suggested_domain:
        query_text = f"{raw_concept_name}. Domain: {suggested_domain}."
    query_vector = model.encode(query_text, normalize_embeddings=True).tolist()

    best = (
        db.query(
            ConceptTaxonomy.canonical_concept_id,
            ConceptTaxonomy.embedding.cosine_distance(query_vector).label("distance"),
        )
        .filter(ConceptTaxonomy.embedding.isnot(None))
        .order_by("distance")
        .first()
    )

    if best is None:
        return None, 0.0

    canonical_concept_id, distance = best
    similarity = 1 - distance  # cosine_distance -> similarity

    if similarity >= CONFIDENT_MATCH_THRESHOLD:
        return canonical_concept_id, similarity
    return None, similarity  # not confident enough — caller flags for review


def flag_for_review(
    db: Session,
    raw_concept_name: str,
    suggested_domain: str | None,
    source_chunk_id: str,
    best_match_concept_id: str | None,
    best_match_score: float,
) -> None:
    review_row = ConceptReviewQueue(
        raw_concept_name=raw_concept_name,
        suggested_domain=suggested_domain,
        source_chunk_id=source_chunk_id,
        best_match_concept_id=best_match_concept_id,
        best_match_score=best_match_score,
        status="pending",
    )
    db.add(review_row)


# ---------------------------------------------------------------------
# Step 3: generate MCQs for a matched concept
# ---------------------------------------------------------------------

def generate_single_mcq(chunk_text: str, concept_name: str, difficulty: str) -> dict:
    """
    Fallback path: generates exactly ONE MCQ at the given difficulty via
    a plain (non-schema) call. Only used if the schema-constrained batch
    call in generate_mcqs_for_concept() still comes up short — kept as a
    defensive backstop, not the primary path.
    """
    prompt = f"""You are writing ONE multiple-choice question for a training
platform for Indian government statistical officers, based STRICTLY on the
passage below. Do not introduce facts not present in or directly implied by
the passage.

Concept being tested: "{concept_name}"
Required difficulty: "{difficulty}"

Write exactly ONE question with exactly 4 options, one correct answer, and
a one-sentence explanation citing the reasoning from the passage.

Return ONLY a single JSON object (not an array) in this exact shape:
{{
  "question": "...",
  "options": [{{"id": "a", "text": "..."}}, {{"id": "b", "text": "..."}}, {{"id": "c", "text": "..."}}, {{"id": "d", "text": "..."}}],
  "correct_option_id": "a",
  "explanation": "...",
  "difficulty": "{difficulty}"
}}

Passage:
\"\"\"{chunk_text}\"\"\"
"""
    result = ollama_client.generate_json(prompt)
    if isinstance(result, list):
        result = result[0] if result else {}
    result["difficulty"] = difficulty
    return result


def generate_mcqs_for_concept(
    chunk_text: str,
    concept_name: str,
    breadth: str,
) -> list[dict]:
    """
    Primary path: ONE schema-constrained Ollama call per difficulty level
    (not per question), using a JSON Schema with minItems=maxItems=count
    to structurally force the array length — this is enforced by Ollama's
    grammar-constrained decoding, not just requested via prompt wording,
    and is expected to be far more reliable than either the original
    single-call-for-everything approach or the one-call-per-question
    fallback (which works but is slow: ~9-17 sequential calls per chunk).

    Falls back to generate_single_mcq() per missing question only if the
    schema call still comes up short for a given difficulty -- logged
    either way, never silently accepted as full success when it isn't.
    """
    plan = MCQ_PLAN_BY_BREADTH.get(breadth, MCQ_PLAN_BY_BREADTH[DEFAULT_BREADTH])
    all_mcqs: list[dict] = []

    for difficulty, count in plan.items():
        if count == 0:
            continue

        prompt = f"""You are writing multiple-choice questions for a training
platform for Indian government statistical officers, based STRICTLY on the
passage below. Do not introduce facts not present in or directly implied by
the passage.

Concept being tested: "{concept_name}"
Required difficulty for ALL questions in this batch: "{difficulty}"

Write {count} DISTINCT questions about this concept at "{difficulty}"
difficulty, each covering a different angle (definition, cause, comparison,
application) rather than repeating the same fact. Each question needs
exactly 4 options, one correct answer, and a one-sentence explanation
citing the reasoning from the passage.

Passage:
\"\"\"{chunk_text}\"\"\"
"""
        try:
            schema = ollama_client.mcq_array_schema(count)
            result = ollama_client.generate_json(prompt, schema=schema)
            batch = _coerce_to_list(result, context=f"MCQ generation ({difficulty})")
        except Exception as e:
            logger.warning(
                "Concept '%s' (%s): schema-constrained batch call failed: %s. "
                "Falling back to one-at-a-time generation for this difficulty.",
                concept_name, difficulty, e,
            )
            batch = []

        for mcq in batch:
            mcq["difficulty"] = difficulty  # force-set, don't trust echo
            all_mcqs.append(mcq)

        # defensive backstop: top up any shortfall one question at a time
        shortfall = count - len(batch)
        if shortfall > 0:
            logger.warning(
                "Concept '%s' (%s): schema call returned %d/%d, generating "
                "%d more individually.",
                concept_name, difficulty, len(batch), count, shortfall,
            )
            for _ in range(shortfall):
                try:
                    mcq = generate_single_mcq(chunk_text, concept_name, difficulty)
                    if mcq.get("question") and mcq.get("options"):
                        all_mcqs.append(mcq)
                except Exception as e:
                    logger.warning(
                        "Concept '%s' (%s): fallback single-question generation "
                        "failed: %s", concept_name, difficulty, e,
                    )

    return all_mcqs


# ---------------------------------------------------------------------
# Step 4: self-consistency confidence check
# ---------------------------------------------------------------------

def self_consistency_check(mcq: dict, chunk_text: str) -> float:
    """
    Independently re-derives an answer to the generated question from the
    same source passage, and compares it against the original
    correct_option_id. This is informational only — it does NOT gate
    publishing (rows still go live regardless), it just gives the trainer
    review queue a signal for prioritization later.

    Returns a float in [0.0, 1.0]:
      1.0 -> re-derived answer matches the original exactly
      0.0 -> re-derived answer disagrees, or couldn't be parsed
    """
    options_text = "\n".join(f"{opt['id']}) {opt['text']}" for opt in mcq["options"])
    prompt = f"""Based STRICTLY on the passage below, answer the following
question. Reply with ONLY the single letter of the correct option (a, b, c,
or d) and nothing else.

Passage:
\"\"\"{chunk_text}\"\"\"

Question: {mcq['question']}
{options_text}
"""
    try:
        raw_answer = ollama_client.generate_text(prompt).strip().lower()
        # take the first a/b/c/d character found, in case the model adds
        # stray punctuation or a word despite the instruction
        re_derived = next((ch for ch in raw_answer if ch in "abcd"), None)
        if re_derived is None:
            return 0.0
        return 1.0 if re_derived == mcq["correct_option_id"].lower() else 0.0
    except Exception as e:
        logger.warning("Self-consistency check failed for a question: %s", e)
        return 0.0


# ---------------------------------------------------------------------
# Orchestration: one chunk end-to-end
# ---------------------------------------------------------------------

def process_chunk(db: Session, chunk: DocumentChunk) -> McqGenerationLog:
    log_row = db.query(McqGenerationLog).filter_by(chunk_id=chunk.chunk_id).first()
    if log_row is None:
        log_row = McqGenerationLog(chunk_id=chunk.chunk_id, status="pending")
        db.add(log_row)

    try:
        concepts = extract_concepts_from_chunk(chunk.chunk_text)
    except Exception as e:
        logger.error("Concept extraction failed for chunk %s: %s", chunk.chunk_id, e)
        log_row.status = "error"
        log_row.error_message = str(e)
        log_row.processed_at = datetime.utcnow()
        db.commit()
        return log_row

    if not concepts:
        log_row.status = "skipped_no_concepts"
        log_row.concepts_found = 0
        log_row.processed_at = datetime.utcnow()
        db.commit()
        return log_row

    total_mcqs_generated = 0

    for concept in concepts:
        raw_name = concept.get("name", "").strip()
        suggested_domain = concept.get("suggested_domain")
        breadth = concept.get("breadth", DEFAULT_BREADTH)
        if not raw_name:
            continue

        canonical_id, score = match_concept_to_taxonomy(db, raw_name, suggested_domain)

        if canonical_id is None:
            flag_for_review(
                db, raw_name, suggested_domain, chunk.chunk_id,
                best_match_concept_id=None, best_match_score=score,
            )
            continue

        try:
            mcqs = generate_mcqs_for_concept(chunk.chunk_text, raw_name, breadth)
        except Exception as e:
            logger.warning(
                "MCQ generation failed for concept '%s' in chunk %s: %s",
                raw_name, chunk.chunk_id, e,
            )
            continue

        for mcq in mcqs:
            try:
                confidence = self_consistency_check(mcq, chunk.chunk_text)
                db.add(MCQ(
                    concept_id=canonical_id,
                    source_chunk_id=chunk.chunk_id,
                    options=mcq["options"],
                    correct_option_id=mcq["correct_option_id"],
                    explanation=mcq.get("explanation"),
                    difficulty=mcq["difficulty"],
                    status="live",
                    confidence_score=confidence,
                ))
                total_mcqs_generated += 1
            except Exception as e:
                logger.warning("Skipping malformed MCQ for concept '%s': %s", raw_name, e)
                continue

    log_row.status = "processed"
    log_row.concepts_found = len(concepts)
    log_row.mcqs_generated = total_mcqs_generated
    log_row.processed_at = datetime.utcnow()
    db.commit()
    return log_row


# ---------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------

def process_pending_chunks(limit: int = 20, doc_id: str | None = None) -> None:
    """
    Processes up to `limit` chunks that either have no log row yet, or
    previously errored (safe to retry). Never re-processes a chunk that
    already succeeded or was cleanly skipped — run this repeatedly to
    work through the full document_chunks table incrementally.

    If `doc_id` is given, only processes chunks from that specific
    document (parent_doc_id) -- lets you deliberately build deep,
    complete coverage on a chosen document rather than sampling
    scattered chunks across many documents.
    """
    db = SessionLocal()
    try:
        already_done = (
            db.query(McqGenerationLog.chunk_id)
            .filter(McqGenerationLog.status.in_(["processed", "skipped_no_concepts"]))
        )
        query = db.query(DocumentChunk).filter(~DocumentChunk.chunk_id.in_(already_done))
        if doc_id is not None:
            query = query.filter(DocumentChunk.parent_doc_id == doc_id)
        pending_chunks = query.limit(limit).all()

        logger.info(
            "Processing %d chunk(s)%s...",
            len(pending_chunks),
            f" from document '{doc_id}'" if doc_id else "",
        )
        for chunk in pending_chunks:
            logger.info("Processing chunk %s (doc: %s)", chunk.chunk_id, chunk.parent_doc_id)
            result = process_chunk(db, chunk)
            logger.info(
                "  -> status=%s concepts_found=%s mcqs_generated=%s",
                result.status, result.concepts_found, result.mcqs_generated,
            )
    finally:
        db.close()
