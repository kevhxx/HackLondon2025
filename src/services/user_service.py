from sqlmodel import Session, select
from src.models.database import User
from fastapi import HTTPException
from datetime import datetime

class UserService:
    def __init__(self, session: Session):
        self.session = session

    def create_user(self, user_data: dict) -> User:
        user = User(**user_data)
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user

    def get_user(self, user_id: int) -> User:
        user = self.session.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user

    def update_user(self, user_id: int, user_data: dict) -> User:
        user = self.get_user(user_id)
        for key, value in user_data.items():
            setattr(user, key, value)
        self.session.commit()
        self.session.refresh(user)
        return user