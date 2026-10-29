import logging

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from app.container import Services
from app.schemas import QueryRequest, SpeechRequest, error, ok
from multilingual.languages import LANGUAGES, PIPELINE_LANGUAGE, UnsupportedLanguage, get_language, needs_translation
from multilingual.sarvam import SarvamError
from multilingual.service import TranslationFailed

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


def _services(request: Request) -> Services:
    return request.app.state.services


# Upstream (Sarvam) failures -> HTTP status for our own response.
_UPSTREAM_STATUS = {"voice_timeout": 504, "voice_rate_limited": 429, "multilingual_not_configured": 503}


def _fail(code: str, message: str, status: int | None = None) -> JSONResponse:
    return JSONResponse(status_code=status or _UPSTREAM_STATUS.get(code, 502), content=error(code, message))


def _not_configured() -> JSONResponse:
    return _fail("multilingual_not_configured",
                 "Voice and multilingual support need a Sarvam AI key. Set SARVAM_API_KEY on the server.")


@router.get("/health")
def health(request: Request):
    s = _services(request)
    return ok({
        "status": "ok",
        "app": s.settings.app_name,
        "components": {
            "query_encoder": s.encoder.model.status(),
            "knowledge_base": {
                "configured": s.retrieval.is_configured,
                "adapter": s.retrieval.retriever.name,
                "status": "CONFIGURED" if s.retrieval.is_configured else "NOT_CONFIGURED",
            },
            "answer_generator": s.generator.status(),
            "multilingual": {"configured": s.multilingual.configured, "provider": "sarvam"},
        },
        "limits": {
            "min_question_chars": s.settings.min_question_chars,
            "max_question_chars": s.settings.max_question_chars,
        },
    })


@router.get("/languages")
def languages(request: Request):
    configured = _services(request).multilingual.configured
    return ok({
        "default": PIPELINE_LANGUAGE,
        "multilingual_configured": configured,
        # Without a key only English works, and the browser's own speech recognition is used.
        "languages": [
            {**lang.to_dict(), "available": configured or not needs_translation(lang.code)}
            for lang in LANGUAGES.values()
        ],
    })


@router.post("/query")
def create_query(body: QueryRequest, request: Request):
    s = _services(request)
    try:
        english, translated = s.multilingual.question_to_english(body.question, body.language)
    except TranslationFailed as exc:
        return _fail(exc.code, exc.message)
    result = s.pipeline.run(english)
    if needs_translation(body.language):
        result.localization = s.multilingual.localize(result, body.language, body.question, translated)
    try:
        s.store.save(result)
    except Exception:  # noqa: BLE001 - history is best-effort; the answer still returns
        logger.exception("Failed to persist query %s", result.id)
    return ok(result.model_dump(mode="json"))


@router.get("/query/{query_id}")
def get_query(query_id: str, request: Request):
    if not query_id.isalnum() or len(query_id) > 64:
        return JSONResponse(status_code=400, content=error("invalid_id", "Malformed query id."))
    result = _services(request).store.get(query_id)
    if result is None:
        return JSONResponse(status_code=404, content=error("not_found", "No query with that id."))
    return ok(result.model_dump(mode="json"))


@router.get("/history")
def history(request: Request):
    s = _services(request)
    return ok(s.store.history(s.settings.history_limit))


@router.delete("/history")
def clear_history(request: Request):
    return ok({"deleted": _services(request).store.clear()})


@router.post("/transcribe")
def transcribe(request: Request, file: UploadFile = File(...), language: str = Form(PIPELINE_LANGUAGE)):
    s = _services(request)
    try:
        get_language(language, "speech_to_text")
    except UnsupportedLanguage as exc:
        return _fail("unsupported_language", str(exc), 422)
    if not s.multilingual.configured:
        return _not_configured()
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if not (content_type.startswith("audio/") or content_type in ("video/webm", "video/mp4", "video/ogg")):
        return _fail("unsupported_audio", "Please upload an audio recording.", 415)
    audio = file.file.read(s.settings.max_audio_bytes + 1)
    if len(audio) > s.settings.max_audio_bytes:
        return _fail("payload_too_large", "The recording is too long. Keep voice questions under 30 seconds.", 413)
    if len(audio) < 1024:
        return _fail("empty_recording", "The recording is empty. Hold the microphone button and speak.", 422)
    try:
        out = s.multilingual.client.transcribe(audio, file.filename or "question.webm", content_type, language)
    except SarvamError as exc:
        return _fail(exc.code, exc.message)
    if not out.text:
        return _fail("no_speech", "No speech was recognised. Please speak closer to the microphone and try again.", 422)
    return ok({"transcript": out.text, "language": language, "detected_language": out.language_code})


@router.post("/speech")
def speech(body: SpeechRequest, request: Request):
    s = _services(request)
    try:
        get_language(body.language, "text_to_speech")
    except UnsupportedLanguage as exc:
        return _fail("unsupported_language", str(exc), 422)
    if not s.multilingual.configured:
        return _not_configured()
    try:
        audio = s.multilingual.client.speak(body.text, body.language)
    except SarvamError as exc:
        return _fail(exc.code, exc.message)
    return Response(content=audio, media_type="audio/wav", headers={"Cache-Control": "no-store"})
