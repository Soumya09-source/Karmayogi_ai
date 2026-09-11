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

def is_self_contained_question(question: str) -> bool:
    """
    Reject questions that refer to answer choices or other
    question context instead of being independently understandable.
    """
    if not isinstance(question, str):
        return False

    q = re.sub(r"\s+", " ", question.strip().lower())

    forbidden_patterns = [
        r"\b(?:option|choice)\s*(?:[1-4]|[a-d])\b",
        r"\b(?:option|choice)\s+(?:above|below)\b",
        r"\b(?:this|that)\s+(?:option|choice)\b",
    ]

    return not any(
        re.search(pattern, q)
        for pattern in forbidden_patterns
    )


# Common words that do not provide useful evidence that a question is grounded
# in the source passage.
_GROUNDING_STOPWORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "be", "being", "been", "do", "does", "did",
    "can", "could", "should", "would", "will", "may", "might", "the", "a",
    "an", "and", "or", "of", "to", "in", "on", "for", "from", "by", "with",
    "about", "as", "at", "into", "through", "during", "than", "that", "this",
    "these", "those", "it", "its", "their", "they", "them", "describe",
    "described", "according", "following", "main", "primary", "purpose",
    "best", "most", "least", "correct", "true", "false", "statement",
}


def _meaningful_tokens(text: str) -> set[str]:
    """Return normalized content-bearing tokens for lightweight grounding."""
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    return {
        token
        for token in tokens
        if len(token) > 2 and token not in _GROUNDING_STOPWORDS
    }


def _explicit_facts(text: str) -> set[str]:
    """Extract numbers/years so fabricated factual values can be rejected."""
    return set(
        re.findall(
            r"\b(?:19|20)\d{2}\b|\b\d+(?:\.\d+)?%?\b",
            str(text),
        )
    )


def is_mcq_grounded(mcq: dict, chunk_text: str) -> bool:
    """
    Lightweight deterministic source-grounding check.

    This is intentionally conservative for explicit numbers/dates and checks
    meaningful lexical overlap between the question/correct answer and the
    source. It is a first-line validator, while self-consistency remains the
    independent answer verification step.
    """
    source_tokens = _meaningful_tokens(chunk_text)
    if not source_tokens:
        return False

    question = str(mcq.get("question", ""))
    correct_id = str(mcq.get("correct_option_id", "")).strip().lower()

    correct_text = ""
    for option in mcq.get("options", []):
        if str(option.get("id", "")).strip().lower() == correct_id:
            correct_text = str(option.get("text", ""))
            break

    # Any explicit number/year in the question or correct answer must occur
    # in the source. This directly catches fabricated facts such as "2024"
    # when the passage never mentions 2024.
    source_facts = _explicit_facts(chunk_text)
    candidate_facts = _explicit_facts(f"{question} {correct_text}")
    if not candidate_facts.issubset(source_facts):
        return False

    question_tokens = _meaningful_tokens(question)
    correct_tokens = _meaningful_tokens(correct_text)

    if not question_tokens or not correct_tokens:
        return False

    question_overlap = question_tokens & source_tokens
    correct_overlap = correct_tokens & source_tokens

    # Very short questions can be grounded by a single strong anchor.
    if len(question_tokens) <= 3:
        question_grounded = bool(question_overlap)
    else:
        question_grounded = (
            len(question_overlap) >= 2
            and len(question_overlap) / len(question_tokens) >= 0.20
        )

    # The correct answer must also contain at least one source-supported
    # content term. This rejects generic/unsupported answers such as an
    # invented entity or date.
    answer_grounded = bool(correct_overlap)

    return question_grounded and answer_grounded


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

# Safety bounds for fallback generation. These limits apply to the current
# generation run only and prevent repeated Ollama calls from becoming costly.
MAX_FALLBACK_ATTEMPTS_PER_QUESTION = 3
MAX_TOTAL_GENERATION_ATTEMPTS = 15



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
    difficulty: str,
    excluded_questions: list[str] | None = None,
) -> dict:
    """
    Fallback path: generates exactly ONE MCQ at the given difficulty.

    Previously attempted questions are supplied so fallback generation
    does not repeat or closely paraphrase questions that were already
    accepted or rejected during the current run.
    """
    excluded_questions = excluded_questions or []

    excluded_block = ""
    if excluded_questions:
        excluded_block = """
QUESTIONS ALREADY ATTEMPTED:
Do NOT generate a question that repeats, lightly rewords, or closely
paraphrases any of these questions:

""" + "\n".join(
            f"- {question}" for question in excluded_questions[-20:]
        )

    prompt = f"""You are writing ONE multiple-choice question for a training
platform for Indian government statistical officers, based STRICTLY on the
passage below.

Do not introduce facts not present in or directly implied by the passage.

Concept being tested: "{concept_name}"

Required difficulty: "{difficulty}"

{excluded_block}

Write exactly ONE question with:

- exactly 4 options
- exactly one correct answer
- one-sentence explanation
- option IDs MUST be exactly "a", "b", "c", "d"
- correct_option_id MUST be exactly one of "a", "b", "c", "d"
- the question must be completely self-contained
- do NOT refer to option numbers or option letters
- do NOT use phrases such as "option 1", "option A", "choice B",
  "the option above", or "the choice below"
- the question and correct answer must be answerable from the supplied passage
- do not introduce dates, numbers, entities, or facts that are absent from
  the supplied passage
- keep all four options reasonably similar in length and grammatical form
- do not make the correct answer noticeably longer or more detailed merely
  because it is the correct answer
- do NOT repeat or closely paraphrase an attempted question

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
    return normalize_mcq(result)

def _validate_generated_mcq_for_generation(
    mcq: dict,
    chunk_text: str,
    attempted_questions: list[str],
    generation_state: dict,
    concept_name: str,
    difficulty: str,
) -> tuple[bool, str | None]:
    """
    Validate a generated MCQ before it is counted toward the requested
    difficulty quota.

    This is used by the sanity-generation path so invalid candidates do not
    artificially satisfy the batch quota. `attempted_questions` is updated by
    the caller after this function returns, regardless of acceptance.
    """
    question = mcq["question"]

    def reject(reason: str, detail: str | None = None) -> tuple[bool, str]:
        counts = generation_state["rejection_counts"]
        counts[reason] = counts.get(reason, 0) + 1
        suffix = f" ({detail})" if detail else ""
        logger.warning(
            "Concept '%s' (%s): %s%s: %s",
            concept_name,
            difficulty,
            reason,
            suffix,
            question,
        )
        return False, reason

    if not is_self_contained_question(question):
        return reject("REJECTED_SELF_REFERENCE")

    # Hard application-level duplicate guard. Never rely only on the LLM
    # following the "do not repeat" prompt instruction.
    # Check this before expensive grounding/self-consistency calls.
    if is_duplicate_question(question, attempted_questions):
        return reject("REJECTED_DUPLICATE")

    if not is_mcq_grounded(mcq, chunk_text):
        return reject("REJECTED_NOT_GROUNDED")

    confidence = self_consistency_check(mcq, chunk_text)
    if confidence <= 0:
        return reject("REJECTED_SELF_CONSISTENCY")

    mcq["confidence_score"] = confidence
    return True, None


def generate_mcqs_for_concept(
    chunk_text: str,
    concept_name: str,
    breadth: str,
    attempted_questions: list[str] | None = None,
    generation_state: dict | None = None,
    quality_validate: bool = False,
) -> list[dict]:
    """
    Generate a difficulty-varied batch of MCQs for a concept.

    Primary path:
        One schema-constrained Ollama call per difficulty.

    Fallback:
        Generate missing questions one at a time, with bounded retries.

    `attempted_questions` is intentionally in-memory and is owned by the
    current sanity-generation run. Every structurally usable generated
    question is recorded, including questions that are rejected by quality
    validation.
    """
    attempted_questions = attempted_questions if attempted_questions is not None else []
    generation_state = generation_state if generation_state is not None else {
        "total_generation_attempts": 0,
        "rejection_counts": {},
    }

    plan = MCQ_PLAN_BY_BREADTH.get(
        breadth,
        MCQ_PLAN_BY_BREADTH[DEFAULT_BREADTH]
    )

    all_mcqs: list[dict] = []

    for difficulty, count in plan.items():
        if count == 0:
            continue

        excluded_block = ""
        if attempted_questions:
            excluded_block = """
QUESTIONS ALREADY ATTEMPTED:
Do NOT repeat, lightly reword, or closely paraphrase any of these questions:

""" + "\n".join(
                f"- {question}" for question in attempted_questions[-20:]
            )

        prompt = f"""You are writing multiple-choice questions for a training
platform for Indian government statistical officers.

Base the questions STRICTLY on the passage below.

Do not introduce facts that are not present in or directly implied by
the passage.

Concept being tested:
"{concept_name}"

Required difficulty for ALL questions:
"{difficulty}"

{excluded_block}

Write exactly {count} DISTINCT questions.

Each question must:

1. Test the specified concept.
2. Cover a different angle where the passage allows it.
3. Have exactly 4 options.
4. Have exactly one correct answer.
5. Have a concise one-sentence explanation.
6. Be answerable using the supplied passage.
7. Be completely self-contained.
8. Never refer to an option number, option letter, or another answer choice.
9. Never introduce dates, numbers, entities, or facts absent from the passage.
10. Not repeat or closely paraphrase any attempted question.

IMPORTANT OPTION QUALITY:
- Keep all four options reasonably similar in length and grammatical form.
- Do not make the correct answer noticeably longer or more detailed merely
  because it is correct.
- Make distractors plausible but clearly incorrect based on the passage.

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

        batch: list = []

        # One schema-constrained generation call counts as one generation
        # attempt against the overall run cap.
        if generation_state["total_generation_attempts"] >= MAX_TOTAL_GENERATION_ATTEMPTS:
            logger.warning(
                "Generation attempt cap reached (%d). Stopping further "
                "generation for concept '%s'.",
                MAX_TOTAL_GENERATION_ATTEMPTS,
                concept_name,
            )
            break

        generation_state["total_generation_attempts"] += 1

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

        # A candidate is only counted toward the requested quota after the
        # enabled quality checks have passed. Rejected candidates are still
        # recorded in attempted_questions so later Ollama calls do not repeat
        # them.
        for mcq in batch:
            try:
                mcq = normalize_mcq(mcq)
                question = mcq["question"]
                previous_attempts = list(attempted_questions)

                if quality_validate:
                    accepted, _ = _validate_generated_mcq_for_generation(
                        mcq,
                        chunk_text,
                        previous_attempts,
                        generation_state,
                        concept_name,
                        difficulty,
                    )
                    if not accepted:
                        attempted_questions.append(question)
                        continue
                else:
                    # Preserve the existing production behavior: only the
                    # structural normalization and self-reference check happen
                    # here; final production validation remains in process_chunk.
                    if not is_self_contained_question(question):
                        generation_state["rejection_counts"][
                            "REJECTED_SELF_REFERENCE"
                        ] = generation_state["rejection_counts"].get(
                            "REJECTED_SELF_REFERENCE", 0
                        ) + 1
                        logger.warning(
                            "Concept '%s' (%s): rejected self-referencing question: %s",
                            concept_name,
                            difficulty,
                            question,
                        )
                        attempted_questions.append(question)
                        continue

                mcq["difficulty"] = difficulty
                attempted_questions.append(question)
                all_mcqs.append(mcq)
                valid_batch_count += 1

            except Exception as e:
                generation_state["rejection_counts"]["REJECTED_INVALID_OPTIONS"] = (
                    generation_state["rejection_counts"].get(
                        "REJECTED_INVALID_OPTIONS", 0
                    ) + 1
                )
                logger.warning(
                    "Concept '%s' (%s): skipping malformed generated MCQ: %s",
                    concept_name,
                    difficulty,
                    e,
                )

        # Generate missing questions individually, with both a per-question
        # retry limit and an overall generation-attempt cap.
        shortfall = count - valid_batch_count

        if shortfall > 0:
            logger.warning(
                "Concept '%s' (%s): valid generation returned "
                "%d/%d questions. Generating up to %d fallback questions.",
                concept_name,
                difficulty,
                valid_batch_count,
                count,
                shortfall,
            )

            for _ in range(shortfall):
                fallback_success = False
                fallback_attempts = 0

                while (
                    not fallback_success
                    and fallback_attempts < MAX_FALLBACK_ATTEMPTS_PER_QUESTION
                    and generation_state["total_generation_attempts"]
                    < MAX_TOTAL_GENERATION_ATTEMPTS
                ):
                    fallback_attempts += 1
                    generation_state["total_generation_attempts"] += 1

                    try:
                        mcq = generate_single_mcq(
                            chunk_text,
                            concept_name,
                            difficulty,
                            excluded_questions=attempted_questions,
                        )

                        question = mcq["question"]
                        previous_attempts = list(attempted_questions)

                        if quality_validate:
                            # The validator performs a hard duplicate check
                            # against every previously attempted question,
                            # including rejected candidates.
                            accepted, _ = _validate_generated_mcq_for_generation(
                                mcq,
                                chunk_text,
                                previous_attempts,
                                generation_state,
                                concept_name,
                                difficulty,
                            )
                            if not accepted:
                                attempted_questions.append(question)
                                continue
                        else:
                            if not is_self_contained_question(question):
                                generation_state["rejection_counts"][
                                    "REJECTED_SELF_REFERENCE"
                                ] = generation_state["rejection_counts"].get(
                                    "REJECTED_SELF_REFERENCE", 0
                                ) + 1
                                logger.warning(
                                    "Concept '%s' (%s): fallback attempt %d/%d "
                                    "rejected self-referencing question: %s",
                                    concept_name,
                                    difficulty,
                                    fallback_attempts,
                                    MAX_FALLBACK_ATTEMPTS_PER_QUESTION,
                                    question,
                                )
                                attempted_questions.append(question)
                                continue

                        attempted_questions.append(question)
                        all_mcqs.append(mcq)
                        fallback_success = True

                    except Exception as e:
                        generation_state["rejection_counts"][
                            "REJECTED_INVALID_OPTIONS"
                        ] = generation_state["rejection_counts"].get(
                            "REJECTED_INVALID_OPTIONS", 0
                        ) + 1
                        logger.warning(
                            "Concept '%s' (%s): fallback attempt %d/%d "
                            "failed: %s",
                            concept_name,
                            difficulty,
                            fallback_attempts,
                            MAX_FALLBACK_ATTEMPTS_PER_QUESTION,
                            e,
                        )

                if not fallback_success:
                    logger.warning(
                        "Concept '%s' (%s): could not fill one missing "
                        "question after %d fallback attempts or due to "
                        "the overall generation cap.",
                        concept_name,
                        difficulty,
                        fallback_attempts,
                    )

    return all_mcqs

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
def _normalize_question_for_comparison(question: str) -> str:
    """Normalize punctuation, whitespace, and case for obvious duplicates."""
    return re.sub(
        r"[^a-z0-9\s]",
        " ",
        str(question).lower(),
    ).strip()


def is_duplicate_question(
    question: str,
    existing_questions: list[str],
    threshold: float = 0.85,
) -> bool:
    """
    Check for an exact normalized duplicate first, then use the existing
    SequenceMatcher threshold for near-duplicate detection.

    The 0.85 threshold is intentionally unchanged from the previous system.
    """
    normalized_question = " ".join(
        _normalize_question_for_comparison(question).split()
    )

    for existing in existing_questions:
        normalized_existing = " ".join(
            _normalize_question_for_comparison(existing).split()
        )

        # Fast obvious-duplicate check.
        if normalized_question == normalized_existing:
            return True

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

    Validation includes:
    - structural validity
    - explanation
    - difficulty
    - self-contained question
    - source grounding
    - exact/near-duplicate detection
    - independent self-consistency
    - explicit rejection-reason tracking

    `attempted_questions` exists only for this in-memory generation run.
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

        # This list is intentionally in-memory and scoped to this single
        # sanity-generation run. It contains both accepted and rejected
        # questions so later Ollama calls avoid repeating them.
        attempted_questions: list[str] = []

        generation_state = {
            "total_generation_attempts": 0,
            "rejection_counts": {},
        }

        def reject(reason: str, question: str, detail: str | None = None) -> None:
            counts = generation_state["rejection_counts"]
            counts[reason] = counts.get(reason, 0) + 1

            suffix = f" ({detail})" if detail else ""
            logger.warning(
                "Sanity check: %s%s: %s",
                reason,
                suffix,
                question,
            )

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
                        attempted_questions=attempted_questions,
                        generation_state=generation_state,
                        quality_validate=True,
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

                        # MCQs returned by generation should already have
                        # been recorded, but append defensively if a caller
                        # supplies an externally generated candidate.
                        if question not in attempted_questions:
                            attempted_questions.append(question)

                        # -------------------------------------------------
                        # 2. Explanation validation
                        # -------------------------------------------------
                        explanation = mcq.get("explanation")
                        if not explanation or not str(explanation).strip():
                            reject(
                                "REJECTED_INVALID_OPTIONS",
                                question,
                                "missing explanation",
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
                            reject(
                                "REJECTED_INVALID_OPTIONS",
                                question,
                                f"invalid difficulty '{difficulty}'",
                            )
                            continue

                        # -------------------------------------------------
                        # 4. Self-contained question validation
                        # -------------------------------------------------
                        if not is_self_contained_question(question):
                            reject(
                                "REJECTED_SELF_REFERENCE",
                                question,
                            )
                            continue

                        # -------------------------------------------------
                        # 5. Source-grounding validation
                        # -------------------------------------------------
                        if not is_mcq_grounded(
                            mcq,
                            chunk.chunk_text,
                        ):
                            reject(
                                "REJECTED_NOT_GROUNDED",
                                question,
                            )
                            continue

                        # -------------------------------------------------
                        # 6. Duplicate / near-duplicate validation
                        # -------------------------------------------------
                        accepted_questions = [
                            sample["question"] for sample in samples
                        ]
                        if is_duplicate_question(
                            question,
                            accepted_questions,
                        ):
                            reject(
                                "REJECTED_DUPLICATE",
                                question,
                            )
                            continue

                        # -------------------------------------------------
                        # 7. Independent answer verification
                        # -------------------------------------------------
                        confidence = self_consistency_check(
                            mcq,
                            chunk.chunk_text,
                        )

                        if confidence <= 0:
                            reject(
                                "REJECTED_SELF_CONSISTENCY",
                                question,
                            )
                            continue

                        # -------------------------------------------------
                        # 8. Accept validated MCQ
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

                    except Exception as e:
                        question_for_log = (
                            str(mcq.get("question", "<unknown>"))
                            if isinstance(mcq, dict)
                            else "<unknown>"
                        )
                        reject(
                            "REJECTED_INVALID_OPTIONS",
                            question_for_log,
                            str(e),
                        )
                        continue

        # -------------------------------------------------------------
        # Final requirement: at least 7 valid questions
        # -------------------------------------------------------------
        rejection_counts = generation_state["rejection_counts"]
        logger.info(
            "Sanity check summary for document '%s': "
            "valid=%d, generation_attempts=%d, rejections=%s",
            doc_id,
            len(samples),
            generation_state["total_generation_attempts"],
            rejection_counts,
        )

        if len(samples) < 7:
            raise ValueError(
                f"Sanity check generated only "
                f"{len(samples)} valid MCQs. "
                f"At least 7 are required for document "
                f"'{doc_id}'. "
                f"Generation attempts: "
                f"{generation_state['total_generation_attempts']}. "
                f"Rejections: {rejection_counts}"
            )

        return samples[:limit]

    finally:
        db.close()

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