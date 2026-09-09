"""
MCQ generation pipeline.

Per chunk:
  1. Extract candidate concepts.
  2. Match each concept against concept_taxonomy.
  3. Generate difficulty-varied MCQs.
  4. Validate generated MCQs for quality.
  5. Run self-consistency checking.
  6. Insert valid MCQs into the database.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------
# Utility: normalize Ollama output
# ---------------------------------------------------------------------

def _coerce_to_list(result, context: str) -> list:
    """
    Convert common Ollama JSON response shapes into a list.
    """

    if isinstance(result, list):
        return result

    if isinstance(result, dict):

        # Single object
        if any(k in result for k in ("name", "question")):
            return [result]

        # Wrapped list
        for value in result.values():
            if isinstance(value, list):
                return value

    raise ValueError(
        f"Could not coerce Ollama's {context} output into a list. "
        f"Got: {type(result)} -> {str(result)[:300]}"
    )


# ---------------------------------------------------------------------
# MCQ normalization
# ---------------------------------------------------------------------

def normalize_mcq(mcq: dict) -> dict:
    """
    Normalize an LLM-generated MCQ.

    Requirements:
    - exactly 4 options
    - option IDs a/b/c/d
    - unique option IDs
    - unique option text
    - valid correct option
    - non-empty question
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

        raw_id = str(
            option.get("id", "")
        ).strip().lower()

        cleaned_id = raw_id.strip(":.) ")

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

    # ---------------------------------------------------------
    # Normalize correct answer
    # ---------------------------------------------------------

    raw_correct_id = mcq.get("correct_option_id")

    if raw_correct_id is None:
        raise ValueError("Missing correct_option_id")

    correct_id = str(
        raw_correct_id
    ).strip().lower()

    correct_id = correct_id.strip(":.) ")

    if correct_id not in id_map:
        raise ValueError(
            f"Invalid correct_option_id: {raw_correct_id}"
        )

    correct_id = id_map[correct_id]

    # ---------------------------------------------------------
    # Check option IDs
    # ---------------------------------------------------------

    option_ids = [
        option["id"]
        for option in normalized_options
    ]

    if len(set(option_ids)) != 4:
        raise ValueError(
            f"Duplicate option IDs after normalization: {option_ids}"
        )

    # ---------------------------------------------------------
    # Check duplicate option text
    # ---------------------------------------------------------

    option_texts = [
        option["text"].strip().casefold()
        for option in normalized_options
    ]

    if len(set(option_texts)) != 4:
        raise ValueError(
            "Duplicate option text detected"
        )

    # ---------------------------------------------------------
    # Check correct answer exists
    # ---------------------------------------------------------

    if correct_id not in option_ids:
        raise ValueError(
            f"correct_option_id '{correct_id}' does not match "
            f"any option: {option_ids}"
        )

    # ---------------------------------------------------------
    # Check question
    # ---------------------------------------------------------

    question = mcq.get("question")

    if not question or not str(question).strip():
        raise ValueError("MCQ question is empty")

    mcq["question"] = str(question).strip()
    mcq["options"] = normalized_options
    mcq["correct_option_id"] = correct_id

    return mcq


# ---------------------------------------------------------------------
# Quality validation
# ---------------------------------------------------------------------

def has_answer_length_bias(
    mcq: dict,
    ratio: float = 1.8
) -> bool:
    """
    Detect whether the correct answer is suspiciously longer
    or shorter than the distractors.
    """

    options = mcq["options"]
    correct_id = mcq["correct_option_id"]

    correct_text = next(
        option["text"]
        for option in options
        if option["id"] == correct_id
    )

    distractors = [
        option["text"]
        for option in options
        if option["id"] != correct_id
    ]

    correct_length = len(
        correct_text.strip()
    )

    distractor_lengths = [
        len(text.strip())
        for text in distractors
    ]

    if not distractor_lengths:
        return True

    average_distractor_length = (
        sum(distractor_lengths)
        / len(distractor_lengths)
    )

    if average_distractor_length == 0:
        return True

    return (
        correct_length >= ratio * average_distractor_length
        or
        correct_length <= average_distractor_length / ratio
    )


def validate_mcq_quality(
    mcq: dict
) -> tuple[bool, str]:
    """
    Run automated quality checks.

    Returns:
        (True, "") when valid.
        (False, reason) when invalid.
    """

    options = mcq.get("options", [])

    # Must have exactly 4 options
    if len(options) != 4:
        return False, "MCQ must have exactly 4 options"

    # Extract and normalize option text
    option_texts = [
        str(option.get("text", "")).strip().casefold()
        for option in options
    ]

    # No option should be empty
    if any(not text for text in option_texts):
        return False, "MCQ contains an empty option"

    # Reject duplicate option text
    if len(set(option_texts)) != len(option_texts):
        return False, "duplicate option text"

    # Reject suspicious correct-answer length bias
    if has_answer_length_bias(mcq):
        return (
            False,
            "correct answer has suspicious length bias"
        )

    return True, ""

def validate_mcq_relevance(
    mcq: dict,
    chunk_text: str,
    concept_name: str,
    min_keyword_overlap: int = 1,
) -> tuple[bool, str]:
    """
    Check whether the MCQ question meaningfully relates to the
    source chunk and target concept using basic keyword overlap.

    Returns:
        (True, "") when relevant.
        (False, reason) when insufficient overlap is found.
    """

    question = str(mcq.get("question", "")).strip()

    if not question:
        return False, "question is empty"

    # Extract keywords separately.
    question_words = set(
        re.findall(
            r"\b[a-zA-Z][a-zA-Z0-9-]*\b",
            question.casefold()
        )
    )

    source_words = set(
        re.findall(
            r"\b[a-zA-Z][a-zA-Z0-9-]*\b",
            chunk_text.casefold()
        )
    )

    concept_words = set(
        re.findall(
            r"\b[a-zA-Z][a-zA-Z0-9-]*\b",
            concept_name.casefold()
        )
    )

    stop_words = {
        "a", "an", "the", "is", "are", "was", "were",
        "what", "which", "who", "when", "where", "why",
        "how", "does", "do", "did", "can", "could",
        "would", "should", "of", "to", "in", "on", "for",
        "from", "with", "and", "or", "as", "by", "be",
        "this", "that", "these", "those", "it", "its",
        "used", "use", "using", "method", "methods"
    }

    def meaningful_words(words: set[str]) -> set[str]:
        return {
            word
            for word in words
            if word not in stop_words and len(word) >= 4
        }

    question_keywords = meaningful_words(question_words)
    source_keywords = meaningful_words(source_words)
    concept_keywords = meaningful_words(concept_words)

    # The question must overlap with the source.
    source_overlap = question_keywords & source_keywords

    # It must also overlap with the concept when the concept
    # contains meaningful terms.
    concept_overlap = question_keywords & concept_keywords

    if len(source_overlap) < min_keyword_overlap:
        return (
            False,
            f"insufficient source overlap: {sorted(source_overlap)}"
        )

    # If the concept has meaningful words, require at least
    # one concept-related keyword OR multiple source keywords.
    if concept_keywords and not concept_overlap and len(source_overlap) < 2:
        return (
            False,
            f"insufficient concept relevance: {sorted(concept_overlap)}"
        )

    return True, ""
# ---------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model

    if _embedding_model is None:
        logger.info(
            "Loading embedding model: %s",
            EMBEDDING_MODEL_NAME
        )

        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    return _embedding_model


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CONFIDENT_MATCH_THRESHOLD = 0.55

MCQ_PLAN_BY_BREADTH = {
    "simple": {
        "easy": 2,
        "medium": 2,
        "hard": 1,
    },
    "moderate": {
        "easy": 3,
        "medium": 3,
        "hard": 2,
    },
    "broad": {
        "easy": 4,
        "medium": 5,
        "hard": 3,
    },
}

DEFAULT_BREADTH = "moderate"


# ---------------------------------------------------------------------
# Step 1: Concept extraction
# ---------------------------------------------------------------------

def extract_concepts_from_chunk(
    chunk_text: str
) -> list[dict]:
    """
    Extract concepts from a document chunk.
    """

    prompt = f"""
You are analyzing a passage from an official Indian government
statistics training/methodology document.

Identify the distinct statistical, technical, or governance
CONCEPTS this passage teaches or explains.

Return ONLY a JSON array.

Each item must have:

{{
    "name": "<short concept name, 2-6 words>",
    "suggested_domain": "<one of: Statistical, Technical, Digital Governance, Behavioural & Managerial, Administrative/Governance>",
    "breadth": "<simple|moderate|broad>"
}}

Breadth means:

simple = one narrow fact or definition
moderate = a standard topic with a few sub-aspects
broad = a wide topic covering many sub-topics

If there is no clear teachable concept, return [].

Passage:

\"\"\"{chunk_text}\"\"\"
"""

    result = ollama_client.generate_json(prompt)

    return _coerce_to_list(
        result,
        context="concept extraction"
    )


# ---------------------------------------------------------------------
# Step 2: Match concept to taxonomy
# ---------------------------------------------------------------------

def match_concept_to_taxonomy(
    db: Session,
    raw_concept_name: str,
    suggested_domain: str | None = None
) -> tuple[str | None, float]:
    """
    Match an extracted concept to the canonical taxonomy.
    """

    exact_match = (
        db.query(ConceptTaxonomy)
        .filter(
            (
                ConceptTaxonomy.canonical_concept_name.ilike(
                    raw_concept_name.strip()
                )
            )
            |
            (
                ConceptTaxonomy.alias_name.ilike(
                    raw_concept_name.strip()
                )
            )
        )
        .first()
    )

    if exact_match:
        return (
            exact_match.canonical_concept_id,
            1.0
        )

    model = get_embedding_model()

    query_text = raw_concept_name.strip()

    if suggested_domain:
        query_text = (
            f"{raw_concept_name.strip()}. "
            f"Domain: {suggested_domain}."
        )

    query_vector = model.encode(
        query_text,
        normalize_embeddings=True
    ).tolist()

    best = (
        db.query(
            ConceptTaxonomy.canonical_concept_id,
            ConceptTaxonomy.embedding.cosine_distance(
                query_vector
            ).label("distance"),
        )
        .filter(
            ConceptTaxonomy.embedding.isnot(None)
        )
        .order_by("distance")
        .first()
    )

    if best is None:
        return None, 0.0

    canonical_concept_id, distance = best

    similarity = 1 - distance

    semantic_threshold = 0.80

    if similarity >= semantic_threshold:
        return (
            canonical_concept_id,
            similarity
        )

    return None, similarity


# ---------------------------------------------------------------------
# Concept review queue
# ---------------------------------------------------------------------

def flag_for_review(
    db: Session,
    raw_concept_name: str,
    suggested_domain: str | None,
    source_chunk_id: str,
    best_match_concept_id: str | None = None,
    best_match_score: float | None = None,
):
    """
    Add an uncertain concept to the review queue.
    """

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
# Step 3: Generate one MCQ
# ---------------------------------------------------------------------

def generate_single_mcq(
    chunk_text: str,
    concept_name: str,
    difficulty: str
) -> dict:
    """
    Generate exactly one MCQ.
    """

    prompt = f"""
You are writing ONE multiple-choice question for a training
platform for Indian government statistical officers.

Base the question STRICTLY on the passage below.

Do not introduce facts not present in or directly implied by
the passage.

Concept being tested:
"{concept_name}"

Required difficulty:
"{difficulty}"

Write exactly ONE question with:

- exactly 4 options
- exactly one correct answer
- one-sentence explanation
- option IDs exactly "a", "b", "c", "d"
- correct_option_id exactly one of "a", "b", "c", "d"

Do NOT use numeric option IDs.

Return ONLY this JSON structure:

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

    result = ollama_client.generate_json(
        prompt
    )

    if isinstance(result, list):
        result = result[0] if result else {}

    if not isinstance(result, dict):
        raise ValueError(
            f"Expected MCQ object, got {type(result)}"
        )

    result["difficulty"] = difficulty

    result = normalize_mcq(result)

    return result


# ---------------------------------------------------------------------
# Duplicate question detection
# ---------------------------------------------------------------------

def is_duplicate_question(
    question: str,
    existing_questions: list[str],
    threshold: float = 0.85
) -> bool:
    """
    Detect exact or near-duplicate questions.
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
            normalized_existing
        ).ratio()

        if similarity >= threshold:
            return True

    return False


# ---------------------------------------------------------------------
# Generate MCQs for concept
# ---------------------------------------------------------------------

def generate_mcqs_for_concept(
    chunk_text: str,
    concept_name: str,
    breadth: str
) -> list[dict]:
    """
    Generate difficulty-varied MCQs for a concept.

    Every MCQ is:
    - normalized
    - checked for duplicate options
    - checked for answer-length bias
    - checked for duplicate questions
    """

    plan = MCQ_PLAN_BY_BREADTH.get(
        breadth,
        MCQ_PLAN_BY_BREADTH[DEFAULT_BREADTH]
    )

    all_mcqs: list[dict] = []

    for difficulty, count in plan.items():

        if count == 0:
            continue

        prompt = f"""
You are writing multiple-choice questions for a training
platform for Indian government statistical officers.

Base the questions STRICTLY on the passage below.

Do not introduce facts that are not present in or directly
implied by the passage.

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

IMPORTANT:

Every option ID MUST be exactly:

"a"
"b"
"c"
"d"

Use each option ID exactly once.

The correct_option_id must also be one of:

"a"
"b"
"c"
"d"

Do NOT use numeric IDs.

Do NOT use punctuation in option IDs.

Do NOT make the correct answer obviously longer than the
distractors.

Return ONLY valid JSON compatible with the supplied schema.

Passage:

\"\"\"{chunk_text}\"\"\"
"""

        # ---------------------------------------------------------
        # Generate batch
        # ---------------------------------------------------------

        try:

            schema = ollama_client.mcq_array_schema(
                count
            )

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
                "Concept '%s' (%s): schema-constrained "
                "batch call failed: %s. Falling back "
                "to one-at-a-time generation.",
                concept_name,
                difficulty,
                e,
            )

            batch = []

        # ---------------------------------------------------------
        # Validate batch MCQs
        # ---------------------------------------------------------

        valid_batch_count = 0

        for mcq in batch:

            try:

                mcq = normalize_mcq(mcq)

                # Never trust model difficulty
                mcq["difficulty"] = difficulty

                # Quality validation
                quality_ok, quality_reason = (
                    validate_mcq_quality(mcq)
                )

                if not quality_ok:

                    logger.warning(
                        "Concept '%s' (%s): rejecting "
                        "low-quality MCQ: %s",
                        concept_name,
                        difficulty,
                        quality_reason,
                    )

                    continue
                relevance_ok, relevance_reason = validate_mcq_relevance(
                    mcq,
                    chunk_text,
                    concept_name,
                )

                if not relevance_ok:
                    logger.warning(
                        "Concept '%s' (%s): rejecting "
                        "irrelevant MCQ: %s",
                        concept_name,
                        difficulty,
                        relevance_reason,
                    )
                    continue

                # Duplicate question check
                existing_questions = [
                    item["question"]
                    for item in all_mcqs
                ]

                if is_duplicate_question(
                    mcq["question"],
                    existing_questions
                ):

                    logger.warning(
                        "Concept '%s' (%s): rejecting "
                        "duplicate/near-duplicate MCQ: %s",
                        concept_name,
                        difficulty,
                        mcq["question"],
                    )

                    continue

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

        # ---------------------------------------------------------
        # Fallback generation
        # ---------------------------------------------------------

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

            attempts = 0
            max_attempts = shortfall * 3

            while (
                len([
                    m for m in all_mcqs
                    if m["difficulty"] == difficulty
                ]) < count
                and attempts < max_attempts
            ):

                attempts += 1

                try:

                    mcq = generate_single_mcq(
                        chunk_text,
                        concept_name,
                        difficulty
                    )

                    # Quality validation
                    quality_ok, quality_reason = (
                        validate_mcq_quality(mcq)
                    )

                    if not quality_ok:

                        logger.warning(
                            "Concept '%s' (%s): rejecting "
                            "low-quality fallback MCQ: %s",
                            concept_name,
                            difficulty,
                            quality_reason,
                        )

                        continue
                                            # Source/concept relevance check
                    relevance_ok, relevance_reason = validate_mcq_relevance(
                        mcq,
                        chunk_text,
                        concept_name,
                    )

                    if not relevance_ok:
                        logger.warning(
                            "Concept '%s' (%s): rejecting "
                            "irrelevant fallback MCQ: %s",
                            concept_name,
                            difficulty,
                            relevance_reason,
                        )
                        continue
                    # Duplicate question validation
                    existing_questions = [
                        item["question"]
                        for item in all_mcqs
                    ]

                    if is_duplicate_question(
                        mcq["question"],
                        existing_questions
                    ):

                        logger.warning(
                            "Concept '%s' (%s): rejecting "
                            "duplicate fallback MCQ: %s",
                            concept_name,
                            difficulty,
                            mcq["question"],
                        )

                        continue

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
# Step 4: Self-consistency check
# ---------------------------------------------------------------------

def self_consistency_check(
    mcq: dict,
    chunk_text: str
) -> float:
    """
    Independently verify the generated answer.

    Returns:
        1.0 -> answer matches
        0.0 -> answer disagrees or cannot be parsed
    """

    options_text = "\n".join(
        f"{opt['id']}) {opt['text']}"
        for opt in mcq["options"]
    )

    prompt = f"""
Based STRICTLY on the passage below, answer the following
multiple-choice question.

Reply with ONLY the option ID.

Valid option IDs are:

a
b
c
d

Do not provide an explanation.

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

        if raw_answer in {
            "a",
            "b",
            "c",
            "d"
        }:

            re_derived = raw_answer

        else:

            match = re.search(
                r"(?:answer\s*(?:is|:)?\s*)?"
                r"([abcd])(?:\)|\.|\s|$)",
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
            str(
                mcq["correct_option_id"]
            )
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
            "Self-consistency check failed: %s",
            e,
        )

        return 0.0


# ---------------------------------------------------------------------
# Sanity MCQ generation
# ---------------------------------------------------------------------

def generate_sanity_mcqs_for_document(
    doc_id: str,
    limit: int = 8
) -> list[dict]:
    """
    Generate temporary sanity-check MCQs.

    These are NOT inserted into the mcqs table.
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
            .order_by(
                DocumentChunk.chunk_order
            )
            .all()
        )

        if not chunks:
            raise ValueError(
                f"No chunks found for document '{doc_id}'"
            )

        samples: list[dict] = []
        accepted_questions: list[str] = []

        # ---------------------------------------------------------
        # Process chunks
        # ---------------------------------------------------------

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
                    DEFAULT_BREADTH
                )

                if not raw_name:
                    continue

                try:

                    mcqs = generate_mcqs_for_concept(
                        chunk.chunk_text,
                        raw_name,
                        breadth
                    )

                except Exception as e:

                    logger.warning(
                        "Sanity check: MCQ generation "
                        "failed for concept '%s': %s",
                        raw_name,
                        e,
                    )

                    continue

                for mcq in mcqs:

                    if len(samples) >= limit:
                        break

                    try:

                        # -------------------------------------------------
                        # Structural validation
                        # -------------------------------------------------

                        mcq = normalize_mcq(mcq)

                        question = mcq["question"]

                        # -------------------------------------------------
                        # Quality validation
                        # -------------------------------------------------

                        quality_ok, quality_reason = (
                            validate_mcq_quality(mcq)
                        )

                        if not quality_ok:

                            logger.warning(
                                "Sanity check: low-quality "
                                "MCQ rejected: %s",
                                quality_reason,
                            )

                            continue

                        # -------------------------------------------------
                        # Explanation validation
                        # -------------------------------------------------

                        explanation = mcq.get(
                            "explanation"
                        )

                        if not explanation or not str(
                            explanation
                        ).strip():

                            logger.warning(
                                "Sanity check: MCQ rejected "
                                "because explanation is missing."
                            )

                            continue

                        # -------------------------------------------------
                        # Difficulty validation
                        # -------------------------------------------------

                        valid_difficulties = {
                            "easy",
                            "medium",
                            "hard",
                        }

                        difficulty = str(
                            mcq.get(
                                "difficulty",
                                ""
                            )
                        ).strip().lower()

                        if difficulty not in valid_difficulties:

                            logger.warning(
                                "Sanity check: invalid difficulty '%s'",
                                difficulty,
                            )

                            continue

                        # -------------------------------------------------
                        # Duplicate validation
                        # -------------------------------------------------

                        if is_duplicate_question(
                            question,
                            accepted_questions
                        ):

                            logger.warning(
                                "Sanity check: duplicate or "
                                "near-duplicate question rejected: %s",
                                question,
                            )

                            continue

                        # -------------------------------------------------
                        # Self-consistency
                        # -------------------------------------------------

                        confidence = self_consistency_check(
                            mcq,
                            chunk.chunk_text
                        )

                        if confidence <= 0:

                            logger.warning(
                                "Sanity check: low-confidence "
                                "MCQ rejected: %s",
                                question,
                            )

                            continue

                        # -------------------------------------------------
                        # Accept
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
                                "source_chunk_id": (
                                    chunk.chunk_id
                                ),
                                "confidence_score": confidence,
                            }
                        )

                        accepted_questions.append(
                            question
                        )

                    except Exception as e:

                        logger.warning(
                            "Sanity check: invalid MCQ "
                            "skipped for concept '%s': %s",
                            raw_name,
                            e,
                        )

                        continue

        # ---------------------------------------------------------
        # Final sanity requirement
        # ---------------------------------------------------------

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
# Step 5: Process one chunk
# ---------------------------------------------------------------------

def process_chunk(
    db: Session,
    chunk: DocumentChunk
) -> McqGenerationLog:

    log_row = (
        db.query(McqGenerationLog)
        .filter_by(
            chunk_id=chunk.chunk_id
        )
        .first()
    )

    if log_row is None:

        log_row = McqGenerationLog(
            chunk_id=chunk.chunk_id,
            status="pending",
        )

        db.add(log_row)

    try:

        concepts = extract_concepts_from_chunk(
            chunk.chunk_text
        )

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

    # ---------------------------------------------------------
    # Process each concept
    # ---------------------------------------------------------

    for concept in concepts:

        raw_name = str(
            concept.get("name", "")
        ).strip()

        suggested_domain = concept.get(
            "suggested_domain"
        )

        breadth = concept.get(
            "breadth",
            DEFAULT_BREADTH
        )

        if not raw_name:
            continue

        # ---------------------------------------------------------
        # Concept matching
        # ---------------------------------------------------------

        canonical_id, score = (
            match_concept_to_taxonomy(
                db,
                raw_name,
                suggested_domain
            )
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

        # ---------------------------------------------------------
        # Generate MCQs
        # ---------------------------------------------------------

        try:

            mcqs = generate_mcqs_for_concept(
                chunk.chunk_text,
                raw_name,
                breadth
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

        # ---------------------------------------------------------
        # Final validation + database insertion
        # ---------------------------------------------------------

        inserted_questions: list[str] = []

        for mcq in mcqs:

            try:

                # Structural validation
                mcq = normalize_mcq(mcq)

                # Quality validation
                quality_ok, quality_reason = (
                    validate_mcq_quality(mcq)
                )

                if not quality_ok:

                    logger.warning(
                        "Concept '%s': rejecting low-quality "
                        "MCQ before DB insertion: %s",
                        raw_name,
                        quality_reason,
                    )

                    continue
                # Source/concept relevance validation
                relevance_ok, relevance_reason = validate_mcq_relevance(
                    mcq,
                    chunk.chunk_text,
                    raw_name,
                )

                if not relevance_ok:
                    logger.warning(
                        "Concept '%s': rejecting irrelevant "
                        "MCQ before DB insertion: %s",
                        raw_name,
                        relevance_reason,
                    )
                    continue

                # Duplicate question validation
                if is_duplicate_question(
                    mcq["question"],
                    inserted_questions
                ):

                    logger.warning(
                        "Concept '%s': rejecting duplicate "
                        "MCQ before DB insertion: %s",
                        raw_name,
                        mcq["question"],
                    )

                    continue

                # Self-consistency
                confidence = self_consistency_check(
                    mcq,
                    chunk.chunk_text
                )

                if confidence <= 0:

                    logger.warning(
                        "Concept '%s': rejecting MCQ because "
                        "self-consistency check failed: %s",
                        raw_name,
                        mcq["question"],
                    )

                    continue

                # -------------------------------------------------
                # Insert
                # -------------------------------------------------

                db.add(
                    MCQ(
                        question=mcq["question"],
                        concept_id=canonical_id,
                        source_chunk_id=chunk.chunk_id,
                        options=mcq["options"],
                        correct_option_id=(
                            mcq["correct_option_id"]
                        ),
                        explanation=mcq.get(
                            "explanation"
                        ),
                        difficulty=mcq["difficulty"],
                        status="live",
                        confidence_score=confidence,
                    )
                )

                inserted_questions.append(
                    mcq["question"]
                )

                total_mcqs_generated += 1

            except Exception as e:

                logger.warning(
                    "Skipping malformed MCQ for concept "
                    "'%s': %s",
                    raw_name,
                    e,
                )

                continue

    # ---------------------------------------------------------
    # Update generation log
    # ---------------------------------------------------------

    log_row.status = "processed"
    log_row.concepts_found = len(concepts)
    log_row.mcqs_generated = total_mcqs_generated
    log_row.processed_at = datetime.utcnow()

    db.commit()

    return log_row


# ---------------------------------------------------------------------
# Step 6: Batch processing
# ---------------------------------------------------------------------

def process_pending_chunks(
    limit: int = 20,
    doc_id: str | None = None
) -> None:
    """
    Process pending document chunks.
    """

    db = SessionLocal()

    try:

        already_done = (
            db.query(
                McqGenerationLog.chunk_id
            )
            .filter(
                McqGenerationLog.status.in_(
                    [
                        "processed",
                        "skipped_no_concepts",
                    ]
                )
            )
        )

        query = (
            db.query(DocumentChunk)
            .filter(
                ~DocumentChunk.chunk_id.in_(
                    already_done
                )
            )
        )

        if doc_id is not None:

            query = query.filter(
                DocumentChunk.parent_doc_id == doc_id
            )

        pending_chunks = (
            query
            .limit(limit)
            .all()
        )

        logger.info(
            "Processing %d chunk(s)%s...",
            len(pending_chunks),
            (
                f" from document '{doc_id}'"
                if doc_id
                else ""
            ),
        )

        for chunk in pending_chunks:

            logger.info(
                "Processing chunk %s (doc: %s)",
                chunk.chunk_id,
                chunk.parent_doc_id,
            )

            result = process_chunk(
                db,
                chunk
            )

            logger.info(
                "  -> status=%s "
                "concepts_found=%s "
                "mcqs_generated=%s",
                result.status,
                result.concepts_found,
                result.mcqs_generated,
            )

    finally:

        db.close()