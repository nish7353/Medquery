from typing import Any

from pydantic import BaseModel, field_validator

from app.config import get_settings
from multilingual.languages import LANGUAGES, PIPELINE_LANGUAGE
from query_processing.preprocess import clean


def _known_language(value: str) -> str:
    if value not in LANGUAGES:
        raise ValueError(f"Unsupported language '{value}'. Supported: {', '.join(LANGUAGES)}.")
    return value


class QueryRequest(BaseModel):
    question: str
    # Language the question is asked in and the answer should come back in.
    language: str = PIPELINE_LANGUAGE

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        return _known_language(value)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        settings = get_settings()
        text = clean(value)
        if len(text) < settings.min_question_chars:
            raise ValueError(f"Question must be at least {settings.min_question_chars} characters long.")
        if len(text) > settings.max_question_chars:
            raise ValueError(f"Question must be at most {settings.max_question_chars} characters long.")
        if not any(ch.isalpha() for ch in text):
            raise ValueError("Question must contain words.")
        return text


class SpeechRequest(BaseModel):
    text: str
    language: str = PIPELINE_LANGUAGE

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        return _known_language(value)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        text = clean(value)
        limit = get_settings().max_speech_chars
        if not text:
            raise ValueError("There is no text to read aloud.")
        if len(text) > limit:
            raise ValueError(f"Text to read aloud must be at most {limit} characters.")
        return text


def ok(data: Any) -> dict:
    return {"success": True, "data": data}


def error(code: str, message: str, details: Any = None) -> dict:
    body: dict[str, Any] = {"success": False, "error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return body
