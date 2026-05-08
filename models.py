from pydantic import BaseModel, EmailStr
from typing import Optional


class SignUpRequest(BaseModel):
    email: EmailStr
    username: str
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class User(BaseModel):
    id: Optional[str] = None
    email: Optional[str] = None
    username: Optional[str] = None


class AuthResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    user: Optional[User] = None
    token: Optional[str] = None

