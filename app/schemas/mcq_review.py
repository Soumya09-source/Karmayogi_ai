from typing import Literal, Optional

from pydantic import BaseModel


class MCQReviewCreate(BaseModel):
    decision: Literal["confirmed_bad", "confirmed_correct", "edited"]
    options: Optional[list[dict]] = None
    correct_option_id: Optional[str] = None
    difficulty: Optional[str] = None
    explanation: Optional[str] = None
