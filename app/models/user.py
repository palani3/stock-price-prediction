from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None
    picture: Optional[str] = None

class UserInDB(BaseModel):
    email: EmailStr
    hashed_password: Optional[str] = None
    created_at: datetime
    auth_provider: Optional[str] = "local"
    name: Optional[str] = None
    picture: Optional[str] = None

class UserLogin(BaseModel):
    username: EmailStr
    password: str

class UserResponse(BaseModel):
    email: str
    name: Optional[str] = None
    picture: Optional[str] = None
    created_at: datetime  # Default to "local" for regular users, "google" for Google auth users

class GoogleUser(BaseModel):
    email: str
    name: str
    picture: str
    created_at: datetime

class GoogleAuthResponse(BaseModel):
    access_token: str
    token_type: str
    user: GoogleUser

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordReset(BaseModel):
    token: str
    new_password: str

class GoogleAuthRequest(BaseModel):
    credential: str

