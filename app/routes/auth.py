from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import RedirectResponse
from datetime import datetime, timedelta
import httpx
import secrets
from ..models.user import (
    UserCreate,
    UserResponse,
    GoogleUser,
    GoogleAuthResponse,
    PasswordResetRequest,
    PasswordReset,
    UserLogin,
    GoogleAuthRequest

)
from ..utils.auth import get_password_hash, verify_password, create_access_token, get_current_user
from ..database import get_database
from ..config import settings

from fastapi import APIRouter

router = APIRouter()


# google sign library
from google.oauth2 import id_token
from google.auth.transport import requests

router = APIRouter()

@router.post("/register", response_model=UserResponse)
async def register_user(user: UserCreate):
    db = await get_database()
    # Check if user already exists
    if await db["users"].find_one({"email": user.email}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    # Create new user with name and picture
    user_dict = {
        "email": user.email,
        "hashed_password": get_password_hash(user.password),
        "created_at": datetime.utcnow(),
        "name": user.name,
        "picture": user.picture,
        "auth_provider": "local"
    }
    await db["users"].insert_one(user_dict)
    return UserResponse(
        email=user.email,
        name=user.name,
        picture=user.picture,
        created_at=user_dict["created_at"]
    )

@router.post("/login")
async def login(user_credentials: OAuth2PasswordRequestForm = Depends()):
    print(user_credentials) 
    db = await get_database()
    user = await db["users"].find_one({"email": user_credentials.username})
    
    if not user or not verify_password(user_credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user["email"]},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    # Return token and user information
    return {
        "access_token": access_token,
        "token_type": "bearer",
        # "user": {
        #     "email": user["email"],
        #     "name": user.get("name"),
        #     "picture": user.get("picture"),
        #     "created_at": user["created_at"]
        # }
    }


@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user = Depends(get_current_user)):
    return UserResponse(
        email=current_user["email"],
        name=current_user.get("name"),
        picture=current_user.get("picture"),
        created_at=current_user["created_at"]
    )

@router.post("/auth/google")
async def google_auth(auth_request: GoogleAuthRequest):
    credential = auth_request.credential
    try:
        # Verify the Google ID token
        idinfo = id_token.verify_oauth2_token(
            credential, 
            requests.Request(), 
            settings.GOOGLE_CLIENT_ID
        )

        email = idinfo['email']
        
        db = await get_database()
        user = await db["users"].find_one({"email": email})

        if not user:
            # Create new user if not exists
            user = {
                "email": email,
                "name": idinfo.get("name", ""),
                "picture": idinfo.get("picture", ""),
                "hashed_password": None,
                "auth_provider": "google",
                "created_at": datetime.utcnow(),
            }
            await db["users"].insert_one(user)
        else:
            # Update existing user's profile
            await db["users"].update_one(
                {"email": email},
                {"$set": {
                    "name": idinfo.get("name", user.get("name", "")),
                    "picture": idinfo.get("picture", user.get("picture", "")),
                }}
            )
            user = await db["users"].find_one({"email": email})

        # Generate access token
        app_access_token = create_access_token(data={"sub": email})

        return {
            "access_token": app_access_token,
            "token_type": "bearer",
            "user": {
                "email": email,
                "name": idinfo.get("name", ""),
                "picture": idinfo.get("picture", ""),
                "created_at": user["created_at"],
            },
        }

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Google token"
        )

@router.post("/forgot-password")
async def forgot_password(reset_request: PasswordResetRequest):
    """
    Request a password reset. Generates a reset token and stores it in the database.
    """
    db = await get_database()
    user = await db["users"].find_one({"email": reset_request.email})
    if not user:
        # Explicitly throw an error if the email is not registered
        raise HTTPException(
            status_code=404,
            detail="Email is not registered"
        )
    # Generate a secure random token
    reset_token = secrets.token_urlsafe(32)
    reset_token_expiry = datetime.utcnow() + timedelta(hours=1)
    # Store the reset token and its expiry in the database
    await db["users"].update_one(
        {"email": reset_request.email},
        {
            "$set": {
                "reset_token": reset_token,
                "reset_token_expiry": reset_token_expiry
            }
        }
    )
    # In a real application, send an email with the reset link
    reset_link = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"
    return {
        "message": "Password reset link has been sent to your email",
        "reset_link": reset_link  # Remove this in production
    }


@router.post("/reset-password")
async def reset_password(reset_data: PasswordReset):
    """
    Reset the password using the token received in email
    """
    db = await get_database()
    # Find user with the reset token
    user = await db["users"].find_one({
        "reset_token": reset_data.token,
        "reset_token_expiry": {"$gt": datetime.utcnow()}  # Token hasn't expired
    })
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    # Update password and remove reset token
    hashed_password = get_password_hash(reset_data.new_password)
    await db["users"].update_one(
        {"_id": user["_id"]},
        {
            "$set": {"hashed_password": hashed_password},
            "$unset": {
                "reset_token": "",
                "reset_token_expiry": ""
            }
        }
    )
    return {"message": "Password has been reset successfully"}