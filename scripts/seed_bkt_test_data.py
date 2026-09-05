"""
Seed test data for the BKT pipeline.

Populates (in dependency order):
  1. concept_taxonomy        — only inserts fallback concepts if empty;
                               otherwise reuses whatever real concepts exist.
  2. employee_profile        — a handful of test employees.
  3. competency_framework_matrix — designation x competency_area ->
                               expected_proficiency_level. Drives BKT's
                               initial mastery seeding (see app/services/bkt.py
                               :: seed_initial_mastery). The `competency_domain`
                               column here MUST match concept_taxonomy's
                               `competency_area` values exactly, or seeding
                               silently falls back to the 0.3 default.
  4. mcqs                    — 5 real questions per concept, spread across
                               easy/medium/hard, status='live'. BKT's bank
                               health check (app/services/bkt.py ::
                               concept_bank_healthy) requires >=5 live MCQs
                               and >=2 distinct difficulty levels per concept
                               or that concept gets silently excluded.

Safe to re-run: every insert is guarded by an existence check first, so
running this twice will not create duplicates or raise PK conflicts.

Run from the repo root, with your real DATABASE_URL configured (.env or
environment), inside your venv:

    python scripts/seed_bkt_test_data.py
"""

from app.db import SessionLocal
from app.models.concept import ConceptTaxonomy
from app.models.employee_profile import EmployeeProfile
from app.models.competency_matrix import CompetencyFrameworkMatrix
from app.models.mcq import MCQ


# ---------------------------------------------------------------------------
# 1. Fallback concepts (only used if concept_taxonomy is empty)
# ---------------------------------------------------------------------------

FALLBACK_CONCEPTS = [
    {
        "canonical_concept_id": "concept_python_basics",
        "canonical_concept_name": "Python Basics",
        "competency_area": "engineering",
    },
    {
        "canonical_concept_id": "concept_sql_joins",
        "canonical_concept_name": "SQL Joins",
        "competency_area": "engineering",
    },
    {
        "canonical_concept_id": "concept_data_privacy_basics",
        "canonical_concept_name": "Data Privacy Basics",
        "competency_area": "compliance",
    },
]


# ---------------------------------------------------------------------------
# 2. Test employees (designations must line up with the matrix in step 3)
# ---------------------------------------------------------------------------

TEST_EMPLOYEES = [
    {
        "employee_id": "emp_test_001",
        "name": "Test Employee One",
        "designation": "Software Engineer",
        "years_of_service": 2,
        "department": "Engineering",
    },
    {
        "employee_id": "emp_test_002",
        "name": "Test Employee Two",
        "designation": "Compliance Officer",
        "years_of_service": 5,
        "department": "Legal & Compliance",
    },
]


# ---------------------------------------------------------------------------
# 3. Competency framework matrix rows
#    (designation, competency_domain) -> expected_proficiency_level (1-5)
# ---------------------------------------------------------------------------

COMPETENCY_MATRIX_ROWS = [
    {"designation": "Software Engineer", "competency_domain": "engineering", "expected_proficiency_level": 4},
    {"designation": "Software Engineer", "competency_domain": "compliance", "expected_proficiency_level": 2},
    {"designation": "Compliance Officer", "competency_domain": "compliance", "expected_proficiency_level": 4},
    {"designation": "Compliance Officer", "competency_domain": "engineering", "expected_proficiency_level": 1},
]


# ---------------------------------------------------------------------------
# 4. MCQs — 5 real questions per concept, mixed difficulty, status='live'
# ---------------------------------------------------------------------------

def _opts(*texts):
    letters = "abcd"
    return [{"id": letters[i], "text": t} for i, t in enumerate(texts)]


MCQS_BY_CONCEPT = {
    "concept_python_basics": [
        {
            "id": "mcq_py_1", "difficulty": "easy",
            "options": _opts("int", "float", "str", "bool"),
            "correct_option_id": "b",
            # "What data type does the literal 3.14 represent in Python?"
        },
        {
            "id": "mcq_py_2", "difficulty": "easy",
            "options": _opts("func", "def", "lambda", "method"),
            "correct_option_id": "b",
            # "Which keyword defines a function in Python?"
        },
        {
            "id": "mcq_py_3", "difficulty": "medium",
            "options": _opts("List", "Tuple", "Set", "Dict"),
            "correct_option_id": "b",
            # "Which of these Python collection types is immutable?"
        },
        {
            "id": "mcq_py_4", "difficulty": "medium",
            "options": _opts("try/except", "if/else", "for/while", "def/return"),
            "correct_option_id": "a",
            # "Which construct is used to handle runtime errors gracefully?"
        },
        {
            "id": "mcq_py_5", "difficulty": "hard",
            "options": _opts(
                "It copies the value", "It shares the same reference",
                "It creates a new empty object", "It raises an error",
            ),
            "correct_option_id": "b",
            # "When you pass a mutable object (like a list) into a function
            #  without copying it, what happens to the reference inside
            #  the function by default?"
        },
    ],
    "concept_sql_joins": [
        {
            "id": "mcq_sql_1", "difficulty": "easy",
            "options": _opts("INNER JOIN", "UNION", "GROUP BY", "ORDER BY"),
            "correct_option_id": "a",
            # "Which clause returns only rows with matches in both tables?"
        },
        {
            "id": "mcq_sql_2", "difficulty": "easy",
            "options": _opts("MERGE", "JOIN", "APPEND", "LINK"),
            "correct_option_id": "b",
            # "Which SQL keyword combines rows from two tables based on a
            #  related column?"
        },
        {
            "id": "mcq_sql_3", "difficulty": "medium",
            "options": _opts(
                "Nothing extra", "Unmatched left rows with NULLs on the right",
                "Only unmatched rows", "An error",
            ),
            "correct_option_id": "b",
            # "What does a LEFT JOIN return that an INNER JOIN does not?"
        },
        {
            "id": "mcq_sql_4", "difficulty": "medium",
            "options": _opts("Subqueries", "Table aliases", "Indexes", "Views"),
            "correct_option_id": "b",
            # "In a self-join, what must you use to distinguish the two
            #  references to the same table?"
        },
        {
            "id": "mcq_sql_5", "difficulty": "hard",
            "options": _opts(
                "A CROSS JOIN only", "A junction/bridge table",
                "A UNION of both tables", "It's not possible in SQL",
            ),
            "correct_option_id": "b",
            # "When two tables don't share a direct key, what technique
            #  lets you connect them through a third linking table?"
        },
    ],
    "concept_data_privacy_basics": [
        {
            "id": "mcq_priv_1", "difficulty": "easy",
            "options": _opts(
                "Personally Identifiable Information",
                "Personal Internet Identity",
                "Private Information Index",
                "Public Identity Info",
            ),
            "correct_option_id": "a",
            # "What does 'PII' stand for?"
        },
        {
            "id": "mcq_priv_2", "difficulty": "easy",
            "options": _opts("Data minimization", "Data maximization", "Data replication", "Data encryption"),
            "correct_option_id": "a",
            # "What principle refers to collecting only the data necessary
            #  for a stated purpose?"
        },
        {
            "id": "mcq_priv_3", "difficulty": "medium",
            "options": _opts("Encryption", "Anonymization", "Compression", "Indexing"),
            "correct_option_id": "b",
            # "What is the term for removing identifying details from a
            #  dataset so individuals can't be re-identified?"
        },
        {
            "id": "mcq_priv_4", "difficulty": "medium",
            "options": _opts("Consent", "A backup", "A firewall rule", "A checksum"),
            "correct_option_id": "a",
            # "What must typically be obtained from a person before
            #  processing their personal data for a new purpose?"
        },
        {
            "id": "mcq_priv_5", "difficulty": "hard",
            "options": _opts(
                "Encryption failure", "Re-identification risk",
                "Data duplication", "Schema drift",
            ),
            "correct_option_id": "b",
            # "What is the risk called when 'anonymized' data can be
            #  re-identified by combining it with other available data?"
        },
    ],
}


def seed_concepts(db) -> list:
    """Returns the list of concept_ids to seed employees/matrix/mcqs against.
    Reuses existing concept_taxonomy rows if any exist; otherwise inserts
    the fallback concepts above."""
    existing = db.query(ConceptTaxonomy).all()
    if existing:
        concept_ids = [c.canonical_concept_id for c in existing]
        print(f"[skip] concept_taxonomy already has {len(existing)} row(s) — reusing: {concept_ids}")
        print("        NOTE: MCQs below are only inserted for the fallback concept_ids.")
        print("        If your real concepts have different IDs, adjust MCQS_BY_CONCEPT accordingly.")
        return concept_ids

    print(f"[insert] concept_taxonomy is empty — inserting {len(FALLBACK_CONCEPTS)} fallback concepts")
    for c in FALLBACK_CONCEPTS:
        db.add(ConceptTaxonomy(
            canonical_concept_id=c["canonical_concept_id"],
            canonical_concept_name=c["canonical_concept_name"],
            competency_area=c["competency_area"],
        ))
    db.commit()
    return [c["canonical_concept_id"] for c in FALLBACK_CONCEPTS]


def seed_employees(db) -> None:
    for e in TEST_EMPLOYEES:
        if db.query(EmployeeProfile).filter_by(employee_id=e["employee_id"]).first():
            print(f"[skip] employee_profile {e['employee_id']} already exists")
            continue
        db.add(EmployeeProfile(**e))
        print(f"[insert] employee_profile {e['employee_id']} ({e['designation']})")
    db.commit()


def seed_competency_matrix(db) -> None:
    for row in COMPETENCY_MATRIX_ROWS:
        exists = (
            db.query(CompetencyFrameworkMatrix)
            .filter_by(
                designation=row["designation"],
                competency_domain=row["competency_domain"],
            )
            .first()
        )
        if exists:
            print(f"[skip] competency_framework_matrix ({row['designation']}, {row['competency_domain']}) already exists")
            continue
        db.add(CompetencyFrameworkMatrix(**row))
        print(f"[insert] competency_framework_matrix ({row['designation']}, {row['competency_domain']}) -> level {row['expected_proficiency_level']}")
    db.commit()


def seed_mcqs(db, concept_ids: list) -> None:
    for concept_id in concept_ids:
        questions = MCQS_BY_CONCEPT.get(concept_id)
        if not questions:
            print(f"[warn] no MCQs defined for concept_id={concept_id} — skipping. "
                  f"Add entries to MCQS_BY_CONCEPT for real concepts.")
            continue

        for q in questions:
            if db.query(MCQ).filter_by(id=q["id"]).first():
                print(f"[skip] mcq {q['id']} already exists")
                continue
            db.add(MCQ(
                id=q["id"],
                concept_id=concept_id,
                options=q["options"],
                correct_option_id=q["correct_option_id"],
                difficulty=q["difficulty"],
                status="live",
                times_served=0,
                times_correct=0,
            ))
            print(f"[insert] mcq {q['id']} (concept={concept_id}, difficulty={q['difficulty']})")
    db.commit()


def verify(db, concept_ids: list) -> None:
    print("\n--- Bank health check (must be >=5 total, >=2 difficulty levels per concept) ---")
    for concept_id in concept_ids:
        rows = (
            db.query(MCQ.difficulty)
            .filter(MCQ.concept_id == concept_id, MCQ.status == "live")
            .all()
        )
        difficulties = [r[0] for r in rows]
        distinct = set(difficulties)
        status = "OK" if len(difficulties) >= 5 and len(distinct) >= 2 else "FAILS health check"
        print(f"  {concept_id}: total={len(difficulties)} difficulties={sorted(distinct)}  [{status}]")


def main():
    db = SessionLocal()
    try:
        concept_ids = seed_concepts(db)
        seed_employees(db)
        seed_competency_matrix(db)
        seed_mcqs(db, concept_ids)
        verify(db, concept_ids)
        print("\nDone.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
