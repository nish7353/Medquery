"""Application configuration, loaded from environment variables / a `.env` file.

Every setting can be overridden with an environment variable prefixed `MEDQUERY_`,
e.g. `MEDQUERY_QUERY_MODEL_NAME=dmis-lab/biobert-base-cased-v1.1`.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEDQUERY_",
        env_file=(PROJECT_DIR / ".env", BACKEND_DIR / ".env"),
        extra="ignore",
    )

    app_name: str = "MedQuery"
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]

    # --- Request limits -------------------------------------------------------
    max_question_chars: int = 500
    min_question_chars: int = 3
    max_request_bytes: int = 8_192

    # --- BERT query understanding ---------------------------------------------
    # Any BERT-family encoder on the Hugging Face Hub (or a local directory).
    # The default is a 6-layer BERT trained for sentence similarity, the same model that
    # builds the knowledge-base index, so its question embedding drives the search.
    query_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    query_max_tokens: int = 128
    # Load models at startup (in a background thread) instead of on first use.
    preload_models: bool = False
    # If true, never contact the Hugging Face Hub; use only locally cached weights.
    hf_offline: bool = False

    # --- Knowledge retrieval (YOUR knowledge base plugs in here) -------------
    # Import path "package.module:ClassName" of a class implementing
    # retrieval.base.Retriever. Empty -> NotConfiguredRetriever.
    kb_adapter: str = ""
    # Free-form connection string handed to your adapter (file path, DB URL, ...).
    kb_connection: str = ""
    kb_top_k: int = 5
    kb_timeout_seconds: float = 10.0
    # Passages scoring below this are discarded as not relevant enough.
    kb_min_relevance: float = 0.0

    # --- Answer processing ----------------------------------------------------
    # "extractive": BERT reading-comprehension model selects an answer span.
    # "evidence_only": no answer synthesis; the best passages are returned.
    answer_mode: Literal["extractive", "evidence_only"] = "extractive"
    # BioBERT fine-tuned on SQuAD 2.0: the best of three candidates on the COVID-QA dev split
    # (see evaluation/results/reader-dev-*.json).
    qa_model_name: str = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2"
    # Extractive answers with a lower confidence are flagged as low confidence.
    qa_min_confidence: float = 0.15
    qa_max_answer_tokens: int = 64

    # --- Multilingual voice and text (Sarvam AI) --------------------------------
    # Read from SARVAM_API_KEY (or MEDQUERY_SARVAM_API_KEY). Empty -> English only, and the
    # microphone falls back to the browser's own speech recognition.
    sarvam_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias=AliasChoices("SARVAM_API_KEY", "MEDQUERY_SARVAM_API_KEY"))
    sarvam_base_url: str = "https://api.sarvam.ai"
    sarvam_stt_model: str = "saaras:v4"
    sarvam_translate_model: str = "mayura:v1"
    sarvam_tts_model: str = "bulbul:v3"
    # Empty -> Sarvam's default voice for the model.
    sarvam_tts_speaker: str = ""
    sarvam_timeout_seconds: float = 20.0
    # Sarvam's real-time speech API accepts up to 30 s of audio; 30 s of 16 kHz WAV is ~1 MB.
    max_audio_bytes: int = 5 * 1024 * 1024
    max_speech_chars: int = 2_000

    # --- Persistence ----------------------------------------------------------
    database_path: str = str(BACKEND_DIR / "data" / "medquery.sqlite3")
    history_limit: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
