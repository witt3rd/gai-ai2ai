from dataclasses import dataclass
from functools import lru_cache
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    OPENAI_API_KEY: str
    DATETIME_FORMAT: str
    SESSION_DIR: str
    TRANSCRIPT_DIR: str
    PROMPTS_FILE: str


@lru_cache()
def get_config() -> Settings:
    return Settings(
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
        DATETIME_FORMAT=os.getenv("DATETIME_FORMAT", "%Y-%m-%dT%H:%M:%S.%fZ"),
        SESSION_DIR=os.getenv("SESSION_DIR", "./sessions"),
        TRANSCRIPT_DIR=os.getenv("TRANSCRIPT_DIR", "./transcripts"),
        PROMPTS_FILE=os.getenv("PROMPTS_FILE", "prompts.json"),
    )
