"""
Bayesian Knowledge Tracing (BKT) — session engine.

Implements the 4-parameter BKT model (P_L0, P_T, P_G, P_S) against the
existing schema: reads from `concept_taxonomy`, `employee_profile`,
`competency_framework_matrix`, `mcqs`; writes to `concept_mastery` and
`assessment_history`.

No LLM calls anywhere in this module. Pure Python, session-time math only.

IMPORTANT — the single most common BKT bug:
    P_T (the learning-transition probability) is applied ONLY at the start
    of a new session (see `begin_session`), never mid-session. Applying it
    after every question causes mastery to swing on statistical noise
    within one sitting. `bayesian_update` / `apply_answer` never touch P_T.

Deviation from the brief's pseudocode signatures: every function that
needs to read or write the database takes an explicit `db: Session` as
its first argument, matching the dependency-injection style already used
in `app/routers/auth.py` (`Depends(get_db)`). The brief's pure-math
functions (`bayesian_update`, `apply_session_transition`,
`select_next_concept`) are kept dependency-free so they can be unit
tested without a database at all.

Architecture note on `run_quiz_session`:
    A real HTTP API answers one question per request, not in one blocking
    loop. `run_quiz_session` is provided as a synchronous orchestrator for
    offline testing / batch simulation (it matches the brief's requested
    signature and pseudocode exactly). A router would instead call the
    building blocks directly:
      - "next question" endpoint  -> get_eligible_concepts +
        select_next_concept_for_employee + select_question
      - "submit answer" endpoint  -> log_assessment_history + apply_answer
"""

from datetime import datetime
from typing import Callable, Dict, List, Optional

from sqlalchemy.orm import Session

from app.models.assessment_history import AssessmentHistory
from app.models.competency_matrix import CompetencyFrameworkMatrix
from app.models.concept import ConceptTaxonomy
from app.models.concept_mastery import ConceptMastery
from app.models.employee_profile import EmployeeProfile
from app.models.mcq import MCQ


# ---------------------------------------------------------------------------
# Config defaults (static for MVP; not yet fit from data via EM)
# ---------------------------------------------------------------------------

DEFAULT_P_L0 = 0.3      # used only if no competency-matrix row matches
DEFAULT_P_T = 0.15      # cross-session learning probability
DEFAULT_P_G = 0.25      # informational only — real G is always 1/len(options)
DEFAULT_P_S = 0.1       # slip probability, until personalized via EM

MASTERED_THRESHOLD = 0.85
GAP_CONFIRMED_THRESHOLD = 0.15
MIN_LIVE_MCQS_PER_CONCEPT = 5
MAX_QUESTIONS_PER_SESSION_DEFAULT = 15

_EPS = 1e-6


def _clamp(p: float) -> float:
    """Keep a probability strictly inside (0, 1) so it stays updatable."""
    return min(max(p, _EPS), 1.0 - _EPS)


# ---------------------------------------------------------------------------
# 1. Seeding P_L0 from the competency framework matrix
# ---------------------------------------------------------------------------

def seed_initial_mastery(db: Session, employee_id: str, concept_id: str) -> float:
    """
    Derive the initial mastery prior for (employee, concept) from
    `competency_framework_matrix`, per the employee's designation and the
    concept's competency domain. Falls back to DEFAULT_P_L0 if no
    employee, concept, or matching matrix row is found.

    Mapping: expected_proficiency_level (1-5) -> P_L0 in [0.1, 0.9],
    linear: level 1 -> 0.1, level 5 -> 0.9.
    """
    employee = (
        db.query(EmployeeProfile)
        .filter(EmployeeProfile.employee_id == employee_id)
        .first()
    )
    concept = (
        db.query(ConceptTaxonomy)
        .filter(ConceptTaxonomy.canonical_concept_id == concept_id)
        .first()
    )

    if employee is None or concept is None or not concept.competency_area:
        return DEFAULT_P_L0

    matrix_row = (
        db.query(CompetencyFrameworkMatrix)
        .filter(
            CompetencyFrameworkMatrix.designation == employee.designation,
            CompetencyFrameworkMatrix.competency_domain == concept.competency_area,
        )
        .first()
    )

    if matrix_row is None:
        return DEFAULT_P_L0

    expected_level = matrix_row.expected_proficiency_level  # 1-5
    return 0.1 + (expected_level - 1) * 0.2


# ---------------------------------------------------------------------------
# concept_mastery row access (get-or-create, seeding on first touch)
# ---------------------------------------------------------------------------

def get_or_create_mastery_row(
    db: Session, employee_id: str, concept_id: str
) -> ConceptMastery:
    row = (
        db.query(ConceptMastery)
        .filter(
            ConceptMastery.employee_id == employee_id,
            ConceptMastery.concept_id == concept_id,
        )
        .first()
    )
    if row is not None:
        return row

    seeded = seed_initial_mastery(db, employee_id, concept_id)
    row = ConceptMastery(
        employee_id=employee_id,
        concept_id=concept_id,
        p_l0=seeded,
        p_t=DEFAULT_P_T,
        p_g=DEFAULT_P_G,  # informational only, see module docstring
        p_s=DEFAULT_P_S,
        p_mastery_current=seeded,
        attempt_count=0,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_mastery(db: Session, employee_id: str, concept_id: str) -> float:
    return get_or_create_mastery_row(db, employee_id, concept_id).p_mastery_current


# ---------------------------------------------------------------------------
# 6. Bank health check (gates eligibility; not part of the BKT math itself)
# ---------------------------------------------------------------------------

def concept_bank_healthy(db: Session, concept_id: str) -> bool:
    """
    A concept's question bank is usable only if it has at least
    MIN_LIVE_MCQS_PER_CONCEPT live MCQs spanning at least 2 of the 3
    difficulty levels. Otherwise adaptive difficulty-matching degrades to
    "whatever's available".
    """
    live_mcqs = (
        db.query(MCQ)
        .filter(MCQ.concept_id == concept_id, MCQ.status == "live")
        .all()
    )
    if len(live_mcqs) < MIN_LIVE_MCQS_PER_CONCEPT:
        return False

    difficulties = {m.difficulty for m in live_mcqs}
    return len(difficulties) >= 2


# ---------------------------------------------------------------------------
# get_eligible_concepts
# ---------------------------------------------------------------------------

def get_eligible_concepts(db: Session, employee_id: str) -> List[str]:
    """
    A concept is eligible only if BOTH:
      - its question bank is healthy (see concept_bank_healthy), AND
      - the employee's current mastery for it is not already resolved
        (i.e. not > MASTERED_THRESHOLD or < GAP_CONFIRMED_THRESHOLD from
        a prior session)
    """
    all_concept_ids = [
        row.canonical_concept_id
        for row in db.query(ConceptTaxonomy.canonical_concept_id).all()
    ]

    eligible: List[str] = []
    for concept_id in all_concept_ids:
        if not concept_bank_healthy(db, concept_id):
            continue

        mastery = get_mastery(db, employee_id, concept_id)
        if mastery > MASTERED_THRESHOLD or mastery < GAP_CONFIRMED_THRESHOLD:
            continue

        eligible.append(concept_id)

    return eligible


# ---------------------------------------------------------------------------
# 3. select_next_concept — closest to 0.5 (max uncertainty / info gain)
# ---------------------------------------------------------------------------

def select_next_concept(mastery_by_concept: Dict[str, float]) -> str:
    """Pure function: pick the concept whose mastery is closest to 0.5."""
    if not mastery_by_concept:
        raise ValueError("mastery_by_concept must be non-empty")
    return min(mastery_by_concept, key=lambda c: abs(mastery_by_concept[c] - 0.5))


def select_next_concept_for_employee(
    db: Session, employee_id: str, eligible_concepts: List[str]
) -> str:
    """DB-aware wrapper: fetches current mastery for each candidate, then
    delegates to the pure `select_next_concept`."""
    mastery_by_concept = {
        concept_id: get_mastery(db, employee_id, concept_id)
        for concept_id in eligible_concepts
    }
    return select_next_concept(mastery_by_concept)


# ---------------------------------------------------------------------------
# select_question — difficulty-matched, live-only, not-already-asked
# ---------------------------------------------------------------------------

def _difficulty_preference_order(mastery: float) -> List[str]:
    """Preferred difficulty first, then fallbacks, per the brief's
    mastery -> difficulty bands."""
    if mastery < 0.4:
        return ["easy", "medium", "hard"]
    if mastery <= 0.7:
        return ["medium", "easy", "hard"]
    return ["hard", "medium", "easy"]


def _mcq_to_dict(mcq: MCQ) -> dict:
    return {
        "id": mcq.id,
        "concept_id": mcq.concept_id,
        "options": mcq.options,
        "correct_option_id": mcq.correct_option_id,
        "difficulty": mcq.difficulty,
    }


def select_question(
    db: Session,
    concept_id: str,
    employee_id: str,
    asked_this_session: set,
) -> Optional[dict]:
    """
    Picks a status='live' MCQ tagged to concept_id, not already asked this
    session, with difficulty closest to the employee's current mastery.
    Falls back through the other difficulty levels before giving up.
    Returns None if the bank is exhausted for this concept this session
    (caller must remove the concept from eligible_concepts and continue).
    """
    mastery = get_mastery(db, employee_id, concept_id)

    for difficulty in _difficulty_preference_order(mastery):
        query = db.query(MCQ).filter(
            MCQ.concept_id == concept_id,
            MCQ.status == "live",
            MCQ.difficulty == difficulty,
        )
        if asked_this_session:
            query = query.filter(~MCQ.id.in_(asked_this_session))

        mcq = query.order_by(MCQ.times_served.asc()).first()
        if mcq is not None:
            return _mcq_to_dict(mcq)

    return None


# ---------------------------------------------------------------------------
# 4. Bayesian update — standard BKT equations, no P_T here (ever)
# ---------------------------------------------------------------------------

def bayesian_update(L: float, correct: bool, num_options: int, S: float = DEFAULT_P_S) -> float:
    """
    Pure math: standard BKT posterior update given one observed answer.
    G is always derived as 1/num_options — never hardcoded. P_T is
    NEVER applied here — see apply_session_transition / begin_session.
    """
    L = _clamp(L)
    G = 1.0 / num_options

    if correct:
        numerator = L * (1 - S)
        denominator = numerator + (1 - L) * G
    else:
        numerator = L * S
        denominator = numerator + (1 - L) * (1 - G)

    if denominator <= 0:
        return L
    return _clamp(numerator / denominator)


def apply_answer(
    db: Session,
    employee_id: str,
    concept_id: str,
    mcq: dict,
    correct: bool,
) -> float:
    """
    DB-aware wrapper: fetches current mastery + personalized slip rate,
    computes the new mastery via bayesian_update, and persists it to
    concept_mastery (does NOT apply P_T — session-boundary only).
    """
    row = get_or_create_mastery_row(db, employee_id, concept_id)
    S = row.p_s if row.p_s is not None else DEFAULT_P_S
    num_options = len(mcq["options"])

    new_mastery = bayesian_update(row.p_mastery_current, correct, num_options, S=S)

    row.p_mastery_current = new_mastery
    row.attempt_count += 1
    row.last_updated = datetime.utcnow()
    db.add(row)
    db.commit()

    return new_mastery


# ---------------------------------------------------------------------------
# 5. Cross-session transition — applied ONLY at session start
# ---------------------------------------------------------------------------

def apply_session_transition(prior_mastery: float, P_T: float = DEFAULT_P_T) -> float:
    """Pure math: reflects learning that may have happened since the
    employee's last session (e.g. completed a recommended course)."""
    return _clamp(prior_mastery + (1 - prior_mastery) * P_T)


def begin_session(db: Session, employee_id: str) -> None:
    """
    Call this ONCE, at the very start of a new quiz session, before
    selecting eligible concepts. NEVER call it mid-session — that is the
    single most common BKT bug (see module docstring).

    Applies P_T only to concepts the employee has previously attempted
    (attempt_count > 0); concepts never touched yet still hold their
    freshly-seeded P_L0 and don't need a transition applied.
    """
    rows = (
        db.query(ConceptMastery)
        .filter(
            ConceptMastery.employee_id == employee_id,
            ConceptMastery.attempt_count > 0,
        )
        .all()
    )
    for row in rows:
        p_t = row.p_t if row.p_t is not None else DEFAULT_P_T
        row.p_mastery_current = apply_session_transition(row.p_mastery_current, p_t)
        db.add(row)
    db.commit()


# ---------------------------------------------------------------------------
# assessment_history logging (+ bank usage stats on the served MCQ)
# ---------------------------------------------------------------------------

def log_assessment_history(
    db: Session,
    session_id: str,
    employee_id: str,
    mcq_id: str,
    concept_id: str,
    correct: bool,
) -> None:
    entry = AssessmentHistory(
        session_id=session_id,
        employee_id=employee_id,
        mcq_id=mcq_id,
        concept_id=concept_id,
        correct=correct,
    )
    db.add(entry)

    mcq = db.query(MCQ).filter(MCQ.id == mcq_id).first()
    if mcq is not None:
        mcq.times_served += 1
        if correct:
            mcq.times_correct += 1
        db.add(mcq)

    db.commit()


# ---------------------------------------------------------------------------
# 3 / 7. Orchestration — see "Architecture note" in the module docstring
# ---------------------------------------------------------------------------

def run_quiz_session(
    db: Session,
    employee_id: str,
    session_id: str,
    answer_provider: Callable[[dict], bool],
    max_questions: int = MAX_QUESTIONS_PER_SESSION_DEFAULT,
) -> dict:
    """
    Offline/test-harness orchestrator matching the brief's pseudocode.
    `answer_provider(mcq) -> bool` is the "present question and get
    answer" boundary — explicitly not BKT's job (brief §3) — swap in a
    real UI/API call, or a scripted answer sequence for testing.

    Stops when any of: all eligible concepts resolved, max_questions
    reached, or no eligible concepts remain.
    """
    begin_session(db, employee_id)

    eligible_concepts = get_eligible_concepts(db, employee_id)
    asked_this_session: set = set()
    outcomes = []

    while eligible_concepts and len(outcomes) < max_questions:
        concept_id = select_next_concept_for_employee(db, employee_id, eligible_concepts)
        mcq = select_question(db, concept_id, employee_id, asked_this_session)

        if mcq is None:
            # Bank exhausted for this concept this session.
            eligible_concepts.remove(concept_id)
            continue

        correct = answer_provider(mcq)
        log_assessment_history(db, session_id, employee_id, mcq["id"], concept_id, correct)
        new_mastery = apply_answer(db, employee_id, concept_id, mcq, correct)

        asked_this_session.add(mcq["id"])
        outcomes.append(
            {
                "mcq_id": mcq["id"],
                "concept_id": concept_id,
                "correct": correct,
                "mastery_after": new_mastery,
            }
        )

        if new_mastery > MASTERED_THRESHOLD or new_mastery < GAP_CONFIRMED_THRESHOLD:
            eligible_concepts.remove(concept_id)

    return {
        "session_id": session_id,
        "employee_id": employee_id,
        "questions_asked": len(outcomes),
        "outcomes": outcomes,
    }
