from pydantic import BaseModel, validator
from typing import List, Optional
from enum import Enum

class EducationalStage(str, Enum):
    PRIMARY = "Primary School"
    MIDDLE = "Middle School"
    HIGH = "High School"
    UNIVERSITY_UG = "University_UG"
    UNIVERSITY_PG = "University_PG"
    PHD = "PhD"

class AgeGroup(str, Enum):
    YOUNG = "0-18"
    ADULT = "18-36"
    MATURE = "36+"

class StudyPlanRequest(BaseModel):
    educational_stage: EducationalStage
    age_group: AgeGroup
    adhd_severity: str
    study_duration: int
    break_duration: int

    @validator('adhd_severity')
    def validate_adhd_severity(cls, v):
        if v not in ["1", "2", "3", "4", "5", "6"]:
            raise ValueError("ADHD Severity must be between 1 and 6")
        return v

    @validator('study_duration')
    def validate_study_duration(cls, v):
        if v not in [60, 90, 120]:
            raise ValueError("Study duration must be 60, 90, or 120 minutes")
        return v

    @validator('break_duration')
    def validate_break_duration(cls, v):
        if v not in [10, 15, 20]:
            raise ValueError("Break duration must be 10, 15, or 20 minutes")
        return v

class StudyPlanResponse(BaseModel):
    participant_id: str
    data: dict