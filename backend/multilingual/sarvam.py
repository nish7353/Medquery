"""Minimal client for the three Sarvam AI REST APIs MedQuery uses.

Endpoints and fields follow docs.sarvam.ai (checked October 2026):

    POST /speech-to-text   multipart: file, model, language_code       -> {transcript, language_code}
    POST /translate        json: input, source_language_code,
                                 target_language_code, model, mode      -> {translated_text, source_language_code}
    POST /text-to-speech   json: text, language_code, model, speaker    -> {audios: [base64 WAV]}

Authentication is the `api-subscription-key` header. The key never leaves the backend,
and question text, transcripts and audio are never logged.
"""

import base64
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.sarvam.ai"

# Sarvam error codes -> (our error code, message safe to show users).
_ERRORS = {
    "invalid_api_key_error": ("voice_auth_failed", "The Sarvam AI API key is invalid. Check SARVAM_API_KEY."),
    "authentication_error": ("voice_auth_failed", "The Sarvam AI API key is invalid. Check SARVAM_API_KEY."),
    "insufficient_quota_error": ("voice_quota", "Sarvam AI credits have run out."),
    "rate_limit_exceeded_error": ("voice_rate_limited", "Too many requests to Sarvam AI. Please wait a moment."),
}
_BY_STATUS = {
    401: _ERRORS["invalid_api_key_error"],
    402: _ERRORS["insufficient_quota_error"],
    403: _ERRORS["invalid_api_key_error"],
    429: _ERRORS["rate_limit_exceeded_error"],
    503: ("voice_unavailable", "Sarvam AI is overloaded right now. Please try again."),
}


class SarvamError(Exception):
    """A failed Sarvam call, with a code and a message that is safe to show users."""

    def __init__(self, code: str, message: str, status: int | None = None):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


@dataclass
class Transcription:
    text: str
    language_code: str | None


@dataclass
class Translation:
    text: str
    source_language_code: str | None


class SarvamClient:
    def __init__(self, api_key: str, *, stt_model: str = "saaras:v4", translate_model: str = "mayura:v1",
                 tts_model: str = "bulbul:v3", tts_speaker: str | None = None, timeout: float = 20.0,
                 base_url: str = DEFAULT_BASE_URL, transport: httpx.BaseTransport | None = None):
        self.stt_model, self.translate_model = stt_model, translate_model
        self.tts_model, self.tts_speaker = tts_model, tts_speaker
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport,
                                  headers={"api-subscription-key": api_key})

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------ calls
    def transcribe(self, audio: bytes, filename: str, content_type: str, language_code: str) -> Transcription:
        data = self._post("/speech-to-text", "speech-to-text",
                          files={"file": (filename, audio, content_type)},
                          data={"model": self.stt_model, "language_code": language_code})
        return Transcription(text=str(data.get("transcript") or "").strip(), language_code=data.get("language_code"))

    def translate(self, text: str, source: str, target: str) -> Translation:
        data = self._post("/translate", "translation", json={
            "input": text, "source_language_code": source, "target_language_code": target,
            "model": self.translate_model, "mode": "formal", "numerals_format": "international",
        })
        translated = str(data.get("translated_text") or "").strip()
        if not translated:
            raise SarvamError("translation_failed", "The translation came back empty.")
        return Translation(text=translated, source_language_code=data.get("source_language_code"))

    def speak(self, text: str, language_code: str) -> bytes:
        body = {"text": text, "language_code": language_code, "model": self.tts_model}
        if self.tts_speaker:
            body["speaker"] = self.tts_speaker
        data = self._post("/text-to-speech", "text-to-speech", json=body)
        audios = data.get("audios") or []
        if not audios:
            raise SarvamError("speech_failed", "No audio was generated.")
        try:
            return b"".join(base64.b64decode(a) for a in audios)
        except ValueError as exc:
            raise SarvamError("speech_failed", "The generated audio could not be decoded.") from exc

    # ------------------------------------------------------------ transport
    def _post(self, path: str, service: str, **kwargs) -> dict:
        try:
            resp = self._http.post(path, **kwargs)
        except httpx.TimeoutException as exc:
            logger.warning("Sarvam %s timed out", service)
            raise SarvamError("voice_timeout", f"Sarvam AI {service} took too long. Please try again.") from exc
        except httpx.HTTPError as exc:
            logger.warning("Sarvam %s unreachable: %s", service, type(exc).__name__)
            raise SarvamError("voice_unreachable", "Cannot reach Sarvam AI. Check the internet connection.") from exc

        if resp.status_code >= 400:
            try:
                err = resp.json().get("error") or {}
            except ValueError:
                err = {}
            code, message = _ERRORS.get(err.get("code"), _BY_STATUS.get(
                resp.status_code, ("voice_failed", f"Sarvam AI {service} failed (HTTP {resp.status_code}).")))
            # Log the status and Sarvam's error code only: their message can echo the input.
            logger.warning("Sarvam %s failed: HTTP %s %s", service, resp.status_code, err.get("code"))
            raise SarvamError(code, message, resp.status_code)
        try:
            return resp.json()
        except ValueError as exc:
            raise SarvamError("voice_failed", f"Sarvam AI {service} returned an unreadable response.") from exc
