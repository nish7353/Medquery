"""Multilingual wrapper around the English MedQuery pipeline.

    question (user's language) -> translate to English -> existing pipeline (retrieval + BERT QA)
                               -> translate the answer, notices and status back -> response

Translation only ever touches text the pipeline already produced. It never adds an answer
when the evidence had none, and the English answer, evidence and sources are always
returned alongside the translation so the two can be compared.
"""

import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from answer_processing.models import AnswerStatus, Localization, QueryResult
from multilingual.languages import PIPELINE_LANGUAGE, get_language, needs_translation
from multilingual.sarvam import SarvamClient, SarvamError

logger = logging.getLogger(__name__)

# What the assistant says for each status, translated for non-English users.
STATUS_MESSAGES = {
    AnswerStatus.ANSWERED: "This answer was taken from the medical sources listed below.",
    AnswerStatus.LOW_CONFIDENCE: "This answer comes from a passage that only partly matches your question, "
                                 "so treat it with caution.",
    AnswerStatus.EVIDENCE_ONLY: "I found related passages but couldn't pick out a direct answer.",
    AnswerStatus.NO_ANSWER_IN_EVIDENCE: "The sources I found don't directly answer this. Try rephrasing, "
                                        "or ask your doctor.",
    AnswerStatus.NO_EVIDENCE: "I couldn't find reliable medical information on this, so I can't answer it.",
    AnswerStatus.KB_NOT_CONFIGURED: "Medical sources are unavailable right now, so I can't answer this.",
    AnswerStatus.RETRIEVAL_FAILED: "Something went wrong while searching the medical sources. Please try again.",
}

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_ASCII_ONLY = re.compile(r"^[\x00-\x7f]*$")


def numbers_preserved(source: str, translated: str) -> bool:
    """True if every number (doses, units, ages...) in `source` also appears in `translated`."""
    want = Counter(n.replace(",", "") for n in _NUMBER.findall(source))
    have = Counter(n.replace(",", "") for n in _NUMBER.findall(translated))
    return not (want - have)


class TranslationFailed(Exception):
    def __init__(self, message: str, code: str = "translation_failed"):
        super().__init__(message)
        self.code, self.message = code, message


class MultilingualService:
    def __init__(self, client: SarvamClient | None):
        self.client = client
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sarvam")

    @property
    def configured(self) -> bool:
        return self.client is not None

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        if self.client:
            self.client.close()

    # ------------------------------------------------------------ question
    def question_to_english(self, question: str, language: str) -> tuple[str, bool]:
        """Return (English question, whether it was translated)."""
        get_language(language, "translation")
        if not needs_translation(language):
            return question, False
        if not self.client:
            raise TranslationFailed("Questions in this language need Sarvam AI. Set SARVAM_API_KEY on the server.",
                                    "multilingual_not_configured")
        # Latin-script input may be English or a romanised Indian language: let Sarvam detect it.
        source = "auto" if _ASCII_ONLY.match(question) else language
        try:
            out = self.client.translate(question, source, PIPELINE_LANGUAGE)
        except SarvamError as exc:
            raise TranslationFailed(f"Your question could not be translated: {exc.message}", exc.code) from exc
        if out.source_language_code == PIPELINE_LANGUAGE:
            return question, False  # it was English already
        return out.text, True

    # ------------------------------------------------------------ answer
    def localize(self, result: QueryResult, language: str, original_question: str,
                 question_translated: bool) -> Localization:
        """Translate the pipeline's English output into `language`. Never raises for translation errors."""
        status_message = STATUS_MESSAGES.get(result.status, "")
        pieces = [status_message] + [n.message for n in result.notices]
        if result.answer:
            pieces.append(result.answer.text)

        if needs_translation(language) and self.client:
            futures = [self._pool.submit(self.client.translate, p, PIPELINE_LANGUAGE, language) for p in pieces]
            translated, failed = [], 0
            for piece, fut in zip(pieces, futures):
                try:
                    translated.append(fut.result().text)
                except SarvamError:
                    translated.append(piece)  # keep the English rather than show nothing
                    failed += 1
        else:
            translated, failed = list(pieces), 0

        warnings = []
        if failed:
            warnings.append("partial_translation")
        answer_text = translated[-1] if result.answer else None
        if result.answer and not numbers_preserved(result.answer.text, answer_text):
            warnings.append("numbers_changed")

        return Localization(
            language=language,
            original_question=original_question,
            english_question=result.question,
            question_translated=question_translated,
            answer_text=answer_text,
            status_message=translated[0],
            notices=translated[1:1 + len(result.notices)],
            complete=not failed,
            warnings=warnings,
        )
