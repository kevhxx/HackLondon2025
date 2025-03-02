from typing import Optional, List
from sqlmodel import Field, SQLModel, create_engine, Session, select, Relationship
from datetime import datetime


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    name: str
    adhd_severity: int = Field(ge=1, le=10)
    education_stage: str
    age_group: str
    favorite_subjects: str
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    study_sessions: List["StudySession"] = Relationship(back_populates="user")


class StudySession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    start_time: datetime
    end_time: datetime
    duration: int  # in minutes
    break_duration: int  # in minutes
    completed: bool = Field(default=False)
    user: User = Relationship(back_populates="study_sessions")


# 数据库连接
DATABASE_URL = "sqlite:///./study_planner.db"
engine = create_engine(DATABASE_URL, echo=True)


def init_db():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
