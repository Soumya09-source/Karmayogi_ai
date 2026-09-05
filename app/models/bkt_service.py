"""
Bayesian Knowledge Tracing (BKT) — core update algorithm.

Pure functions only: no DB/ORM/session imports. This module is meant to be
unit-tested in isolation and then called by a higher-level
`assessment_service.py` that handles question selection, persistence, and
the request/response loop.

Reference: Corbett, A. T., & Anderson, J. R. (1995). Knowledge Tracing:
Modeling the Acquisition of Procedural Knowledge.
"""

from dataclasses import dataclass


# Numerical safety margin so probabilities never hit exact 0 or 1
# (which would break the Bayes-rule division and make the concept
# "stuck" forever in one state).
_EPS = 1e-6


@dataclass(frozen=True)
class BKTParams:
    """
    Per-concept BKT parameters.

    p_l0: prior probability the learner already knows the concept
          before any attempt.
    p_t:  learning probability — chance of transitioning from
          "doesn't know" to "knows" after one attempt/opportunity.
    p_g:  guess probability — chance of answering correctly despite
          not knowing the concept.
    p_s:  slip probability — chance of answering incorrectly despite
          knowing the concept.

    Defaults are reasonable starting points for a cold-start system
    with no historical attempt data. Once enough `attempts` rows exist
    per concept, these can be re-estimated via Expectation-Maximization
    (future work — not needed for the hackathon MVP).
    """

    p_l0: float = 0.3
    p_t: float = 0.2
    p_g: float = 0.25
    p_s: float = 0.1

    def __post_init__(self) -> None:
        for name in ("p_l0", "p_t", "p_g", "p_s"):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}")


def _clamp(p: float) -> float:
    """Keep a probability strictly inside (0, 1) to avoid divide-by-zero
    and to keep the state updatable in future steps."""
    return min(max(p, _EPS), 1.0 - _EPS)


def posterior_given_observation(p_l: float, params: BKTParams, is_correct: bool) -> float:
    """
    Step 1 of a BKT update: Bayesian posterior on P(L) given the
    observed answer (correct/incorrect), BEFORE applying the learning
    transition.

    p_l: current P(L) — probability the learner knows the concept,
         going into this attempt.
    """
    p_l = _clamp(p_l)
    p_g, p_s = params.p_g, params.p_s

    if is_correct:
        numerator = p_l * (1 - p_s)
        denominator = numerator + (1 - p_l) * p_g
    else:
        numerator = p_l * p_s
        denominator = numerator + (1 - p_l) * (1 - p_g)

    if denominator <= 0:
        # Degenerate case (shouldn't happen given _clamp, but guard anyway)
        return p_l

    return _clamp(numerator / denominator)


def apply_learning_transition(p_l_given_obs: float, params: BKTParams) -> float:
    """
    Step 2 of a BKT update: account for the chance the learner picked
    up the concept as a result of this attempt (regardless of whether
    they got it right).
    """
    p_l_given_obs = _clamp(p_l_given_obs)
    p_next = p_l_given_obs + (1 - p_l_given_obs) * params.p_t
    return _clamp(p_next)


def update_mastery(p_l: float, params: BKTParams, is_correct: bool) -> float:
    """
    Full BKT update in one call: Bayesian posterior + learning transition.

    This is what the assessment loop calls after each answered question:

        new_p_l = update_mastery(user_concept.p_l, concept.bkt_params, was_correct)

    Returns the updated P(L) to persist back to `user_concept_mastery`.
    """
    posterior = posterior_given_observation(p_l, params, is_correct)
    return apply_learning_transition(posterior, params)


def predict_correct_probability(p_l: float, params: BKTParams) -> float:
    """
    Given current mastery, predict the probability the learner answers
    correctly on the NEXT attempt. Useful for difficulty-matching when
    selecting the next question (e.g. pick a harder item if this is high,
    an easier one if it's low).
    """
    p_l = _clamp(p_l)
    return p_l * (1 - params.p_s) + (1 - p_l) * params.p_g


def is_mastered(p_l: float, threshold: float = 0.95) -> bool:
    """
    Whether a concept should be considered mastered and removed from
    the active question rotation for this learner.
    """
    return p_l >= threshold


def initial_mastery(params: BKTParams) -> float:
    """Starting P(L) for a learner who has never attempted this concept."""
    return params.p_l0
