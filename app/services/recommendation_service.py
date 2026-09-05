import random
import numpy as np
from sqlalchemy.orm import Session

from app.models.concept_mastery import ConceptMastery
from app.models.concept import ConceptTaxonomy
from app.models.course import Course
from app.models.recommendation import Recommendation


DOMAIN_BOOST_FACTOR = 1.2
MASTERY_THRESHOLD = 0.7


def get_gap_concepts(
    db: Session,
    employee_id: str,
    threshold: float = MASTERY_THRESHOLD
):
    gaps = (
        db.query(ConceptMastery, ConceptTaxonomy)
        .join(
            ConceptTaxonomy,
            ConceptMastery.concept_id == ConceptTaxonomy.canonical_concept_id
        )
        .filter(
            ConceptMastery.employee_id == employee_id,
            ConceptMastery.p_mastery_current < threshold
        )
        .all()
    )

    return [
        {
            "concept_id": mastery.concept_id,
            "domain": concept.parent_domain,
            "embedding": concept.embedding,   # <-- added, needed by get_similarity_scores
        }
        for mastery, concept in gaps
    ]


def get_candidate_courses(db: Session):
    """
    Fetch available courses from the database.
    """

    return db.query(Course).all()



def get_similarity_scores(concept, candidate_courses):
    """
    Real embedding-based similarity: cosine similarity between the gap
    concept's embedding and each candidate course's embedding.
    """
    if concept.get("embedding") is None:
        return []  # can't score a concept with no embedding yet — skip it
                    # rather than crash; it just won't get any recommendation
                    # until the embedding pipeline catches up

    concept_vector = np.array(concept["embedding"], dtype=float)
    concept_norm = np.linalg.norm(concept_vector)

    scores = []
    for course in candidate_courses:
        if course.embedding is None:
            continue

        course_vector = np.array(course.embedding, dtype=float)
        course_norm = np.linalg.norm(course_vector)

        if concept_norm == 0 or course_norm == 0:
            similarity = 0.0
        else:
            similarity = float(
                np.dot(concept_vector, course_vector) / (concept_norm * course_norm)
            )

        scores.append({"course_id": course.course_id, "score": similarity})

    return scores


def apply_domain_boost(
    raw_score: float,
    employee_domain: str,
    course_domain: str
) -> float:
    """
    Apply a 1.2x boost for same-domain matches,
    capped at 1.0.
    """

    if employee_domain and course_domain:
        if employee_domain.lower() == course_domain.lower():
            raw_score *= DOMAIN_BOOST_FACTOR

    return min(raw_score, 1.0)


def rank_recommendations(
    raw_recommendations: list,
    top_n: int = 5
):
    """
    Deduplicate courses, keeping the highest score
    for each course.
    """

    best_courses = {}

    for recommendation in raw_recommendations:
        course_id = recommendation["course_id"]
        score = recommendation["score"]

        if (
            course_id not in best_courses
            or score > best_courses[course_id]["score"]
        ):
            best_courses[course_id] = recommendation

    ranked = sorted(
        best_courses.values(),
        key=lambda item: item["score"],
        reverse=True
    )

    return ranked[:top_n]


def save_recommendations(
    db: Session,
    recommendations: list
):
    """
    Save generated recommendations to the database.
    """

    saved_recommendations = []

    for item in recommendations:

        existing = (
            db.query(Recommendation)
            .filter(
                Recommendation.employee_id == item["employee_id"],
                Recommendation.gap_concept_id == item["gap_concept_id"],
                Recommendation.recommended_course_id == item["course_id"]
            )
            .first()
        )

        if existing:
            existing.similarity_score = item["score"]
            existing.status = "active"

            saved_recommendations.append(existing)

        else:
            recommendation = Recommendation(
                employee_id=item["employee_id"],
                gap_concept_id=item["gap_concept_id"],
                recommended_course_id=item["course_id"],
                similarity_score=item["score"],
                status="active"
            )

            db.add(recommendation)
            saved_recommendations.append(recommendation)

    db.commit()

    for recommendation in saved_recommendations:
        db.refresh(recommendation)

    return saved_recommendations


def generate_recommendations(
    db: Session,
    employee_id: str,
    top_n: int = 5
):
    """
    Generate and save course recommendations for an employee.

    Uses:
    - Real BKT concept mastery data
    - Real courses from the database
    - Temporary random similarity scores
    """

    gap_concepts = get_gap_concepts(
        db=db,
        employee_id=employee_id
    )

    candidate_courses = get_candidate_courses(db)

    raw_recommendations = []

    for gap in gap_concepts:

        similarity_scores = get_similarity_scores(
            gap,
            candidate_courses
        )

        for similarity in similarity_scores:

            course = next(
                (
                    course
                    for course in candidate_courses
                    if course.course_id == similarity["course_id"]
                ),
                None
            )

            if course is None:
                continue

            boosted_score = apply_domain_boost(
                raw_score=similarity["score"],
                employee_domain=gap["domain"],
                course_domain=course.internal_category
            )

            raw_recommendations.append(
                {
                    "employee_id": employee_id,
                    "gap_concept_id": gap["concept_id"],
                    "course_id": course.course_id,
                    "score": boosted_score
                }
            )

    ranked_recommendations = rank_recommendations(
        raw_recommendations,
        top_n
    )

    return save_recommendations(
        db,
        ranked_recommendations
    )