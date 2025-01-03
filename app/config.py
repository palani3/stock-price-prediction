from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

# Load environment variables from the .env file before initializing Settings
load_dotenv()  # Ensure .env file is loaded

class Settings(BaseSettings):
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "stock_prediction"
    SECRET_KEY: str = "your-secret-key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/auth/google/callback"
    FRONTEND_URL: str = "http://localhost:3000"

    class Config:
        env_file = ".env"  # Specifies that the values should be read from the .env file

# Initialize the settings
settings = Settings()

# Optional: Print out values for debugging purposes
print("GOOGLE_CLIENT_ID:", settings.GOOGLE_CLIENT_ID)
print("GOOGLE_CLIENT_SECRET:", settings.GOOGLE_CLIENT_SECRET)
