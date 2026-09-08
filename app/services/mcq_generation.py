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
import re
from difflib import SequenceMatcher
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
def normalize_mcq(mcq: dict) -> dict:
    """
    Normalize an LLM-generated MCQ into a consistent format.

    Requirements:
    - Exactly 4 options
    - Option IDs must be a, b, c, d
    - correct_option_id must be one of a, b, c, d
    - correct_option_id must match an existing option
    """

    if not isinstance(mcq, dict):
        raise ValueError("MCQ must be a dictionary")

    options = mcq.get("options")

    if not isinstance(options, list):
        raise ValueError("MCQ options must be a list")

    if len(options) != 4:
        raise ValueError(
            f"MCQ must have exactly 4 options, got {len(options)}"
        )

    # Map numeric IDs and letter IDs to the standard a/b/c/d format.
    id_map = {
        "1": "a",
        "2": "b",
        "3": "c",
        "4": "d",
        "a": "a",
        "b": "b",
        "c": "c",
        "d": "d",
    }

    normalized_options = []

    for index, option in enumerate(options):

        if not isinstance(option, dict):
            raise ValueError(
                f"Option {index + 1} is not a dictionary"
            )

        text = option.get("text")

        if not text or not str(text).strip():
            raise ValueError(
                f"Option {index + 1} has empty text"
            )

        raw_id = str(option.get("id", "")).strip().lower()

        # Remove accidental punctuation such as ":1", "1)", "a.", etc.
        cleaned_id = raw_id.strip(":.) ")

        # If the model produced a valid numeric/letter ID, map it.
        # Otherwise fall back to the option's position.
        if cleaned_id in id_map:
            normalized_id = id_map[cleaned_id]
        else:
            normalized_id = "abcd"[index]

        normalized_options.append(
            {
                "id": normalized_id,
                "text": str(text).strip(),
            }
        )

    # Normalize the correct answer.
    raw_correct_id = mcq.get("correct_option_id")

    if raw_correct_id is None:
        raise ValueError("Missing correct_option_id")

    correct_id = str(raw_correct_id).strip().lower()

    # Remove accidental punctuation.
    correct_id = correct_id.strip(":.) ")

    if correct_id not in id_map:
        raise ValueError(
            f"Invalid correct_option_id: {raw_correct_id}"
        )

    correct_id = id_map[correct_id]

    # Make sure all option IDs are unique.
    option_ids = [option["id"] for option in normalized_options]

    if len(set(option_ids)) != 4:
        raise ValueError(
            f"Duplicate option IDs after normalization: {option_ids}"
        )

    # Make sure the correct answer actually exists.
    if correct_id not in option_ids:
        raise ValueError(
            f"correct_option_id '{correct_id}' does not match "
            f"any option: {option_ids}"
        )

    # Basic question validation.
    question = mcq.get("question")

    if not question or not str(question).strip():
        raise ValueError("MCQ question is empty")

    mcq["question"] = str(question).strip()
    mcq["options"] = normalized_options
    mcq["correct_option_id"] = correct_id

    return mcq

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
    Match an extracted concept to the canonical taxonomy.

    Matching strategy:
    1. Exact canonical concept / alias match -> accept immediately.
    2. Otherwise use embedding similarity.
    3. Only accept a semantic match when similarity is very strong.
    4. Otherwise return None so the caller can send the concept
       to the concept review queue.

    This prevents broad or unrelated concepts from being silently
    mapped to the nearest available taxonomy concept.
    """
    normalized_name = raw_concept_name.strip().casefold()

    # ---------------------------------------------------------
    # 1. Exact canonical-name or alias-name match
    # ---------------------------------------------------------
    exact_match = (
        db.query(ConceptTaxonomy)
        .filter(
            (ConceptTaxonomy.canonical_concept_name.ilike(raw_concept_name.strip()))
            | (ConceptTaxonomy.alias_name.ilike(raw_concept_name.strip()))
        )
        .first()
    )

    if exact_match:
        return exact_match.canonical_concept_id, 1.0

    # ---------------------------------------------------------
    # 2. Semantic embedding match
    # ---------------------------------------------------------
    model = get_embedding_model()

    query_text = raw_concept_name.strip()

    if suggested_domain:
        query_text = f"{raw_concept_name.strip()}. Domain: {suggested_domain}."

    query_vector = model.encode(
        query_text,
        normalize_embeddings=True,
    ).tolist()

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
    similarity = 1 - distance

    # ---------------------------------------------------------
    # 3. Conservative semantic matching
    # ---------------------------------------------------------
    SEMANTIC_MATCH_THRESHOLD = 0.80

    if similarity >= SEMANTIC_MATCH_THRESHOLD:
        return canonical_concept_id, similarity

    # ---------------------------------------------------------
    # 4. Uncertain match -> caller sends to review queue
    # ---------------------------------------------------------
    return None, similarity

def flag_for_review(
    db: Session,
    raw_concept_name: str,
    suggested_domain: str | None,
    source_chunk_id: str,
    best_match_concept_id: str | None = None,
    best_match_score: float | None = None,
):
    review_item = ConceptReviewQueue(
        raw_concept_name=raw_concept_name,
        suggested_domain=suggested_domain,
        source_chunk_id=source_chunk_id,
        best_match_concept_id=best_match_concept_id,
        best_match_score=best_match_score,
        status="pending",
    )

    db.add(review_item)
    db.flush()

    return review_item

# ---------------------------------------------------------------------
# Step 3: generate MCQs for a matched concept
# ---------------------------------------------------------------------

def generate_single_mcq(
    chunk_text: str,
    concept_name: str,
    difficulty: str
) -> dict:
    """
    Fallback path: generates exactly ONE MCQ at the given difficulty.

    The generated MCQ is normalized before being returned so that
    option IDs are always a/b/c/d.
    """

    prompt = f"""You are writing ONE multiple-choice question for a training
platform for Indian government statistical officers, based STRICTLY on the
passage below.

Do not introduce facts not present in or directly implied by the passage.

Concept being tested: "{concept_name}"

Required difficulty: "{difficulty}"

Write exactly ONE question with:

- exactly 4 options
- exactly one correct answer
- one-sentence explanation
- option IDs MUST be exactly "a", "b", "c", "d"
- correct_option_id MUST be exactly one of "a", "b", "c", "d"

Do NOT use numeric option IDs such as 1, 2, 3, 4.
Do NOT add punctuation to option IDs.

Return ONLY a single JSON object in this exact shape:

{{
  "question": "...",
  "options": [
    {{"id": "a", "text": "..."}},
    {{"id": "b", "text": "..."}},
    {{"id": "c", "text": "..."}},
    {{"id": "d", "text": "..."}}
  ],
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

    if not isinstance(result, dict):
        raise ValueError(
            f"Expected MCQ object, got {type(result)}"
        )

    result["difficulty"] = difficulty

    # Normalize and validate the generated MCQ.
    result = normalize_mcq(result)

    return result


def generate_mcqs_for_concept(
    chunk_text: str,
    concept_name: str,
    breadth: str
) -> list[dict]:
    """
    Generate a difficulty-varied batch of MCQs for a concept.

    Primary path:
        One schema-constrained Ollama call per difficulty.

    Fallback:
        Generate missing questions one at a time.

    Every MCQ is normalized and validated before being returned.
    """

    plan = MCQ_PLAN_BY_BREADTH.get(
        breadth,
        MCQ_PLAN_BY_BREADTH[DEFAULT_BREADTH]
    )

    all_mcqs: list[dict] = []

    for difficulty, count in plan.items():

        if count == 0:
            continue

        prompt = f"""You are writing multiple-choice questions for a training
platform for Indian government statistical officers.

Base the questions STRICTLY on the passage below.

Do not introduce facts that are not present in or directly implied by
the passage.

Concept being tested:
"{concept_name}"

Required difficulty for ALL questions:
"{difficulty}"

Write exactly {count} DISTINCT questions.

Each question must:

1. Test the specified concept.
2. Cover a different angle where the passage allows it.
3. Have exactly 4 options.
4. Have exactly one correct answer.
5. Have a concise one-sentence explanation.
6. Be answerable using the supplied passage.

IMPORTANT OPTION FORMAT:

Every option ID MUST be exactly one of:

"a", "b", "c", "d"

Use each option ID exactly once.

The correct_option_id MUST be exactly one of:

"a", "b", "c", "d"

Do NOT use numeric IDs such as:
"1", "2", "3", "4"

Do NOT use punctuation in option IDs.

Return the questions in JSON format compatible with the supplied schema.

Passage:
\"\"\"{chunk_text}\"\"\"
"""

        try:

            schema = ollama_client.mcq_array_schema(count)

            result = ollama_client.generate_json(
                prompt,
                schema=schema
            )

            batch = _coerce_to_list(
                result,
                context=f"MCQ generation ({difficulty})"
            )

        except Exception as e:

            logger.warning(
                "Concept '%s' (%s): schema-constrained batch "
                "call failed: %s. Falling back to one-at-a-time "
                "generation for this difficulty.",
                concept_name,
                difficulty,
                e,
            )

            batch = []

        valid_batch_count = 0

        # Normalize and validate every generated MCQ.
        for mcq in batch:

            try:

                mcq = normalize_mcq(mcq)

                # Never trust the model's difficulty field.
                mcq["difficulty"] = difficulty

                all_mcqs.append(mcq)

                valid_batch_count += 1

            except Exception as e:

                logger.warning(
                    "Concept '%s' (%s): skipping malformed "
                    "generated MCQ: %s",
                    concept_name,
                    difficulty,
                    e,
                )

        # Generate missing questions individually.
        shortfall = count - valid_batch_count

        if shortfall > 0:

            logger.warning(
                "Concept '%s' (%s): valid generation returned "
                "%d/%d questions. Generating %d more individually.",
                concept_name,
                difficulty,
                valid_batch_count,
                count,
                shortfall,
            )

            for _ in range(shortfall):

                try:

                    mcq = generate_single_mcq(
                        chunk_text,
                        concept_name,
                        difficulty
                    )

                    all_mcqs.append(mcq)

                except Exception as e:

                    logger.warning(
                        "Concept '%s' (%s): fallback "
                        "single-question generation failed: %s",
                        concept_name,
                        difficulty,
                        e,
                    )

    return all_mcqs


# ---------------------------------------------------------------------
# Step 4: self-consistency confidence check
# ---------------------------------------------------------------------

def self_consistency_check(
    mcq: dict,
    chunk_text: str
) -> float:
    """
    Independently re-derives the answer from the source passage.

    Returns:
        1.0 -> re-derived answer matches the generated answer
        0.0 -> answer disagrees or could not be parsed
    """

    options_text = "\n".join(
        f"{opt['id']}) {opt['text']}"
        for opt in mcq["options"]
    )

    prompt = f"""Based STRICTLY on the passage below, answer the
following multiple-choice question.

Reply with ONLY the option ID.

Valid option IDs are:
a
b
c
d

Do not provide an explanation.
Do not provide any other text.

Passage:
\"\"\"{chunk_text}\"\"\"

Question:
{mcq['question']}

Options:
{options_text}
"""

    try:

        raw_answer = (
            ollama_client
            .generate_text(prompt)
            .strip()
            .lower()
        )

        # First handle the ideal response:
        # "a", "b", "c", or "d"
        if raw_answer in {"a", "b", "c", "d"}:
            re_derived = raw_answer

        else:
            # Handle common model responses:
            #
            # "answer: b"
            # "answer is b"
            # "b)"
            # "The correct answer is b."
            
            match = re.search(
                r"(?:answer\s*(?:is|:)?\s*)?([abcd])(?:\)|\.|\s|$)",
                raw_answer,
            )

            re_derived = (
                match.group(1)
                if match
                else None
            )

        if re_derived is None:

            logger.warning(
                "Could not parse self-consistency answer: %r",
                raw_answer,
            )

            return 0.0

        original_answer = (
            str(mcq["correct_option_id"])
            .strip()
            .lower()
        )

        return (
            1.0
            if re_derived == original_answer
            else 0.0
        )

    except Exception as e:

        logger.warning(
            "Self-consistency check failed for a question: %s",
            e,
        )

        return 0.0
def is_duplicate_question(
    question: str,
    existing_questions: list[str],
    threshold: float = 0.85,
) -> bool:
    """
    Check whether a question is an exact or near duplicate
    of any previously accepted sanity-check question.
    """

    normalized_question = " ".join(
        question.lower().split()
    )

    for existing in existing_questions:
        normalized_existing = " ".join(
            existing.lower().split()
        )

        similarity = SequenceMatcher(
            None,
            normalized_question,
            normalized_existing,
        ).ratio()

        if similarity >= threshold:
            return True

    return False

def generate_sanity_mcqs_for_document(
    doc_id: str,
    limit: int = 8,
) -> list[dict]:
    """
    Generate 7–8 temporary sanity-check MCQs for a document.

    These MCQs are generated only for trainer preview.
    They are NOT inserted into the mcqs table.

    Each MCQ is validated for:
    - valid question
    - exactly 4 options
    - valid correct option
    - explanation
    - valid difficulty
    - independent answer verification
    - duplicate / near-duplicate detection
    """

    if limit < 7:
        raise ValueError(
            "Sanity check must generate at least 7 MCQs"
        )

    if limit > 8:
        limit = 8

    db = SessionLocal()

    try:
        chunks = (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.parent_doc_id == doc_id
            )
            .order_by(DocumentChunk.chunk_order)
            .all()
        )

        if not chunks:
            raise ValueError(
                f"No chunks found for document '{doc_id}'"
            )

        samples: list[dict] = []
        accepted_questions: list[str] = []

        # Process chunks in document order.
        for chunk in chunks:

            if len(samples) >= limit:
                break

            try:
                concepts = extract_concepts_from_chunk(
                    chunk.chunk_text
                )

            except Exception as e:
                logger.warning(
                    "Sanity check: concept extraction failed "
                    "for chunk %s: %s",
                    chunk.chunk_id,
                    e,
                )
                continue

            for concept in concepts:

                if len(samples) >= limit:
                    break

                raw_name = str(
                    concept.get("name", "")
                ).strip()

                breadth = concept.get(
                    "breadth",
                    DEFAULT_BREADTH,
                )

                if not raw_name:
                    continue

                try:
                    mcqs = generate_mcqs_for_concept(
                        chunk.chunk_text,
                        raw_name,
                        breadth,
                    )

                except Exception as e:
                    logger.warning(
                        "Sanity check: MCQ generation failed "
                        "for concept '%s' in chunk %s: %s",
                        raw_name,
                        chunk.chunk_id,
                        e,
                    )
                    continue

                for mcq in mcqs:

                    if len(samples) >= limit:
                        break

                    try:
                        # -------------------------------------------------
                        # 1. Structural validation
                        # -------------------------------------------------
                        mcq = normalize_mcq(mcq)

                        question = mcq["question"]

                        # -------------------------------------------------
                        # 2. Explanation validation
                        # -------------------------------------------------
                        explanation = mcq.get("explanation")

                        if not explanation or not str(
                            explanation
                        ).strip():
                            logger.warning(
                                "Sanity check: MCQ rejected because "
                                "explanation is missing."
                            )
                            continue

                        # -------------------------------------------------
                        # 3. Difficulty validation
                        # -------------------------------------------------
                        valid_difficulties = {
                            "easy",
                            "medium",
                            "hard",
                        }

                        difficulty = str(
                            mcq.get("difficulty", "")
                        ).strip().lower()

                        if difficulty not in valid_difficulties:
                            logger.warning(
                                "Sanity check: invalid difficulty '%s'",
                                difficulty,
                            )
                            continue

                        # -------------------------------------------------
                        # 4. Duplicate / near-duplicate validation
                        # -------------------------------------------------
                        if is_duplicate_question(
                            question,
                            accepted_questions,
                        ):
                            logger.warning(
                                "Sanity check: duplicate or "
                                "near-duplicate question rejected: %s",
                                question,
                            )
                            continue

                        # -------------------------------------------------
                        # 5. Independent answer verification
                        # -------------------------------------------------
                        confidence = self_consistency_check(
                            mcq,
                            chunk.chunk_text,
                        )

                        # A confidence of 0 means the independent
                        # verification disagreed with the generated answer.
                        if confidence <= 0:
                            logger.warning(
                                "Sanity check: low-confidence MCQ "
                                "rejected: %s",
                                question,
                            )
                            continue

                        # -------------------------------------------------
                        # 6. Accept validated MCQ
                        # -------------------------------------------------
                        samples.append(
                            {
                                "question": question,
                                "options": mcq["options"],
                                "correct_option_id": (
                                    mcq["correct_option_id"]
                                ),
                                "explanation": str(
                                    explanation
                                ).strip(),
                                "difficulty": difficulty,
                                "concept_name": raw_name,
                                "source_chunk_id": chunk.chunk_id,
                                "confidence_score": confidence,
                            }
                        )

                        accepted_questions.append(
                            question
                        )

                    except Exception as e:
                        logger.warning(
                            "Sanity check: invalid MCQ skipped "
                            "for concept '%s': %s",
                            raw_name,
                            e,
                        )
                        continue

        # -------------------------------------------------------------
        # Final requirement: at least 7 valid questions
        # -------------------------------------------------------------
        if len(samples) < 7:
            raise ValueError(
                f"Sanity check generated only "
                f"{len(samples)} valid MCQs. "
                f"At least 7 are required for document "
                f"'{doc_id}'"
            )

        return samples[:limit]

    finally:
        db.close()
# ---------------------------------------------------------------------
# Orchestration: one chunk end-to-end
# ---------------------------------------------------------------------
def process_chunk(db: Session, chunk: DocumentChunk) -> McqGenerationLog:
    log_row = (
        db.query(McqGenerationLog)
        .filter_by(chunk_id=chunk.chunk_id)
        .first()
    )

    if log_row is None:
        log_row = McqGenerationLog(
            chunk_id=chunk.chunk_id,
            status="pending",
        )
        db.add(log_row)

    try:
        concepts = extract_concepts_from_chunk(chunk.chunk_text)

    except Exception as e:
        logger.error(
            "Concept extraction failed for chunk %s: %s",
            chunk.chunk_id,
            e,
        )

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

        canonical_id, score = match_concept_to_taxonomy(
            db,
            raw_name,
            suggested_domain,
        )

        if canonical_id is None:
            flag_for_review(
                db,
                raw_name,
                suggested_domain,
                chunk.chunk_id,
                best_match_concept_id=None,
                best_match_score=score,
            )
            continue

        try:
            mcqs = generate_mcqs_for_concept(
                chunk.chunk_text,
                raw_name,
                breadth,
            )

        except Exception as e:
            logger.warning(
                "MCQ generation failed for concept '%s' "
                "in chunk %s: %s",
                raw_name,
                chunk.chunk_id,
                e,
            )
            continue

        for mcq in mcqs:

            try:
                # Final validation before database insertion.
                mcq = normalize_mcq(mcq)

                confidence = self_consistency_check(
                    mcq,
                    chunk.chunk_text,
                )

                db.add(
                    MCQ(
                        question=mcq["question"],
                        concept_id=canonical_id,
                        source_chunk_id=chunk.chunk_id,
                        options=mcq["options"],
                        correct_option_id=mcq["correct_option_id"],
                        explanation=mcq.get("explanation"),
                        difficulty=mcq["difficulty"],
                        status="live",
                        confidence_score=confidence,
                    )
                )

                total_mcqs_generated += 1

            except Exception as e:
                logger.warning(
                    "Skipping malformed MCQ for concept '%s': %s",
                    raw_name,
                    e,
                )
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
