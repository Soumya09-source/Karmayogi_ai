import random


DOMAIN_BOOST_FACTOR = 1.2


def get_gap_concepts(employee_id: str):
    """
    Temporary stub for gap detection.

    Later this will query concept_mastery where:
    employee_id matches and p_mastery < threshold.
    """

    return [
        {
            "concept_id": "concept_1",
            "domain": "statistics"
        },
        {
            "concept_id": "concept_2",
            "domain": "data_analysis"
        }
    ]


def get_similarity_scores(concept, candidate_courses):
    """
    Temporary seam for the AI embedding pipeline.

    Later the AI/embedding person can replace this function
    with real vector similarity scores.
    """

    scores = []

    for course in candidate_courses:
        scores.append(
            {
                "course_id": course.course_id,
                "score": random.uniform(0.0, 1.0)
            }
        )

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

    if employee_domain == course_domain:
        boosted = raw_score * DOMAIN_BOOST_FACTOR
    else:
        boosted = raw_score

    return min(boosted, 1.0)

def rank_recommendations(
    raw_recommendations: list,
    top_n: int = 5
):
    """
    Deduplicate courses, keeping the highest score for each course,
    then rank and return the top N recommendations.
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

def generate_recommendations(
    employee_id: str,
    candidate_courses: list,
    top_n: int = 5
):
    """
    Generate recommendations for an employee.

    Uses temporary gap concepts and fake similarity scores
    until the real BKT and embedding pipelines are connected.
    """

    gap_concepts = get_gap_concepts(employee_id)

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

    return rank_recommendations(
        raw_recommendations,
        top_n
    )