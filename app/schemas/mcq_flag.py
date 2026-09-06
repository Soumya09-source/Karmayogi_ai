from typing import Optional

from pydantic import BaseModel


class MCQFlagCreate(BaseModel):
    reason: str
    comment: Optional[str] = None
    session_id: Optional[str] = None
