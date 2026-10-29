"""Builds the application's long-lived services from Settings."""

import logging
from dataclasses import dataclass

from answer_processing.generators import AnswerGenerator, EvidenceOnlyGenerator, ExtractiveQAGenerator
from answer_processing.pipeline import AnswerPipeline
from app.config import Settings
from app.storage import QueryStore
from multilingual.sarvam import SarvamClient
from multilingual.service import MultilingualService
from query_processing.bert_encoder import BertQueryEncoder
from query_processing.understanding import QueryUnderstanding
from retrieval.service import RetrievalService, load_retriever

logger = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    encoder: BertQueryEncoder
    retrieval: RetrievalService
    generator: AnswerGenerator
    pipeline: AnswerPipeline
    store: QueryStore
    multilingual: MultilingualService

    def close(self) -> None:
        self.retrieval.close()
        self.store.close()
        self.multilingual.close()


def build_services(settings: Settings) -> Services:
    encoder = BertQueryEncoder(settings.query_model_name, settings.query_max_tokens, settings.hf_offline)
    retrieval = RetrievalService(
        load_retriever(settings.kb_adapter, settings.kb_connection),
        timeout_seconds=settings.kb_timeout_seconds,
        min_relevance=settings.kb_min_relevance,
    )
    if settings.answer_mode == "extractive":
        generator: AnswerGenerator = ExtractiveQAGenerator(
            settings.qa_model_name, settings.qa_max_answer_tokens, settings.hf_offline
        )
    else:
        generator = EvidenceOnlyGenerator()

    pipeline = AnswerPipeline(
        QueryUnderstanding(encoder), retrieval, generator,
        top_k=settings.kb_top_k, min_confidence=settings.qa_min_confidence,
    )
    return Services(settings, encoder, retrieval, generator, pipeline, QueryStore(settings.database_path),
                    MultilingualService(sarvam_client(settings)))


def sarvam_client(settings: Settings) -> SarvamClient | None:
    parts = settings.sarvam_api_key.get_secret_value().split()
    if not parts:
        return None
    # Keys never contain spaces; anything after the first word is a stray comment or paste.
    key = parts[0]
    if len(parts) > 1:
        logger.warning("SARVAM_API_KEY has extra text after the key; only the first word is used. "
                       "Put comments on their own line in .env.")
    return SarvamClient(key, stt_model=settings.sarvam_stt_model, translate_model=settings.sarvam_translate_model,
                        tts_model=settings.sarvam_tts_model, tts_speaker=settings.sarvam_tts_speaker or None,
                        timeout=settings.sarvam_timeout_seconds, base_url=settings.sarvam_base_url)
