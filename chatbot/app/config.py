import os
from typing import List, Dict, Any
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    REDIS_HOST: str = "redis-vector"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    REDIS_URL: str = f"redis://{REDIS_HOST}:{REDIS_PORT}"
    INDEX_NAME: str = "finance_docs"
    

    OPENAI_MODEL_NAME:str = os.environ["OPENAI_MODEL_NAME"]
    OPENAI_BASE_URL:str = os.environ["OPENAI_BASE_URL"]
    OPENAI_API_KEY:str = os.environ["OPENAI_API_KEY"]

    HUGGINGFACE_MODEL_NAME:str = "all-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"

settings = Settings()