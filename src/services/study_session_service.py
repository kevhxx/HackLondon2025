from sqlmodel import Session, select
from src.models.database import StudySession
from datetime import datetime

class StudySessionService:
    def __init__(self, session: Session):
        self.session = session

    def create_session(self, user_id: int, duration: int, break_duration: int) -> StudySession:
        study_session = StudySession(
            user_id=user_id,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            duration=duration,
            break_duration=break_duration
        )
        self.session.add(study_session)
        self.session.commit()
        self.session.refresh(study_session)
        return study_session

    def get_user_sessions(self, user_id: int) -> list[StudySession]:
        statement = select(StudySession).where(StudySession.user_id == user_id)
        return self.session.exec(statement).all()