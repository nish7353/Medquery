"""Multilingual text and voice via Sarvam AI, against a fake Sarvam server (no network, no key).

Questions and answers here are neutral TEST-ONLY placeholders, not medical content.
"""

import base64
import json

import httpx
import pytest

from answer_processing.generators import GeneratedAnswer
from multilingual.languages import LANGUAGES, UnsupportedLanguage, get_language
from multilingual.sarvam import SarvamClient, SarvamError
from multilingual.service import numbers_preserved
from tests.doubles import FakeGenerator, make_client, ok_retriever

WAV = b"RIFF" + b"\0" * 2048  # placeholder bytes; the fake server never decodes them


class FakeSarvam:
    """Records requests and answers like the Sarvam REST API."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.fail: dict[str, tuple[int, str]] = {}   # path -> (HTTP status, sarvam error code)
        self.timeout: set[str] = set()
        self.transcript = "placeholder question in kannada"
        self.detected = None      # source_language_code to report for "auto"
        self.drop_numbers = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path in self.timeout:
            raise httpx.ReadTimeout("slow", request=request)
        if path in self.fail:
            status, code = self.fail[path]
            return httpx.Response(status, json={"error": {"code": code, "message": "echo of user text"}})
        if path == "/speech-to-text":
            return httpx.Response(200, json={"request_id": "r", "transcript": self.transcript, "language_code": "kn-IN"})
        if path == "/translate":
            body = json.loads(request.content)
            src, tgt, text = body["source_language_code"], body["target_language_code"], body["input"]
            if tgt == "en-IN":
                detected = self.detected or (src if src != "auto" else "hi-IN")
                english = text if detected == "en-IN" else "What is the placeholder thing?"
                return httpx.Response(200, json={"translated_text": english, "source_language_code": detected})
            out = f"<{tgt}> {text}"
            if self.drop_numbers:
                out = out.replace("5", "")
            return httpx.Response(200, json={"translated_text": out, "source_language_code": src})
        if path == "/text-to-speech":
            return httpx.Response(200, json={"audios": [base64.b64encode(WAV).decode()]})
        return httpx.Response(404)

    def client(self) -> SarvamClient:
        return SarvamClient("test-key", transport=httpx.MockTransport(self.handler))

    def calls(self, path: str) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.url.path == path]


@pytest.fixture
def sarvam():
    return FakeSarvam()


def answering_client(sarvam, answer="ALPHA 5 mg"):
    gen = FakeGenerator(GeneratedAnswer(text=answer, confidence=0.9, passage_index=0, start=0, end=len(answer)))
    return make_client(retriever=ok_retriever(), generator=gen, sarvam=sarvam.client() if sarvam else None)


# ---------------------------------------------------------------- languages
def test_language_registry_validates_each_service():
    assert {"en-IN", "hi-IN", "kn-IN", "ta-IN", "te-IN"} <= set(LANGUAGES)
    assert get_language("kn-IN", "text_to_speech").name == "Kannada"
    with pytest.raises(UnsupportedLanguage):
        get_language("fr-FR")


def test_numbers_check():
    assert numbers_preserved("take 2.5 mg twice, 10 days", "<x> 10 ... 2.5 ... 2")
    assert not numbers_preserved("take 25 mg", "<x> take 2 mg")


# ---------------------------------------------------------------- client
def test_client_sends_documented_fields_and_auth_header(sarvam):
    c = sarvam.client()
    assert c.translate("hello", "en-IN", "kn-IN").text == "<kn-IN> hello"
    assert c.speak("hello", "kn-IN") == WAV
    assert c.transcribe(WAV, "q.webm", "audio/webm", "kn-IN").text == sarvam.transcript
    assert all(r.headers["api-subscription-key"] == "test-key" for r in sarvam.requests)
    t = sarvam.calls("/translate")[0]
    assert t["model"] == "mayura:v1" and t["mode"] == "formal" and t["numerals_format"] == "international"
    tts = sarvam.calls("/text-to-speech")[0]
    assert tts == {"text": "hello", "language_code": "kn-IN", "model": "bulbul:v3"}
    stt = sarvam.requests[-1]
    assert b'name="model"' in stt.content and b"saaras:v4" in stt.content and b"kn-IN" in stt.content


@pytest.mark.parametrize("status,code,ours", [
    (403, "invalid_api_key_error", "voice_auth_failed"),
    (429, "rate_limit_exceeded_error", "voice_rate_limited"),
    (402, None, "voice_quota"),
    (500, "internal_server_error", "voice_failed"),
])
def test_client_maps_errors_without_leaking_upstream_message(sarvam, status, code, ours):
    sarvam.fail["/translate"] = (status, code)
    with pytest.raises(SarvamError) as exc:
        sarvam.client().translate("hello", "en-IN", "kn-IN")
    assert exc.value.code == ours
    assert "echo of user text" not in exc.value.message


def test_client_timeout(sarvam):
    sarvam.timeout.add("/text-to-speech")
    with pytest.raises(SarvamError) as exc:
        sarvam.client().speak("hello", "hi-IN")
    assert exc.value.code == "voice_timeout"


# ---------------------------------------------------------------- /api/query
def test_english_query_is_unchanged_and_never_calls_sarvam(sarvam):
    with answering_client(sarvam) as c:
        r = c.post("/api/query", json={"question": "What is the placeholder thing?"}).json()["data"]
    assert r["status"] == "ANSWERED" and r["localization"] is None
    assert sarvam.requests == []


def test_kannada_query_runs_english_pipeline_and_translates_back(sarvam):
    with answering_client(sarvam) as c:
        r = c.post("/api/query", json={"question": "ಪ್ಲೇಸ್‌ಹೋಲ್ಡರ್ ಪ್ರಶ್ನೆ", "language": "kn-IN"}).json()["data"]
        history = c.get("/api/history").json()["data"]
    assert r["question"] == "What is the placeholder thing?", "the pipeline searches in English"
    loc = r["localization"]
    assert loc["original_question"] == "ಪ್ಲೇಸ್‌ಹೋಲ್ಡರ್ ಪ್ರಶ್ನೆ" and loc["question_translated"] is True
    assert loc["answer_text"] == "<kn-IN> ALPHA 5 mg"
    assert r["answer"]["text"] == "ALPHA 5 mg", "the English answer is kept"
    assert loc["status_message"].startswith("<kn-IN> ") and loc["complete"] and loc["warnings"] == []
    assert r["evidence"] and r["sources"], "evidence and sources are preserved"
    to_en = [b for b in sarvam.calls("/translate") if b["target_language_code"] == "en-IN"]
    assert to_en[0]["source_language_code"] == "kn-IN"
    assert history[0]["question"] == "ಪ್ಲೇಸ್‌ಹೋಲ್ಡರ್ ಪ್ರಶ್ನೆ"


def test_latin_script_question_is_auto_detected_and_not_translated_if_english(sarvam):
    sarvam.detected = "en-IN"
    with answering_client(sarvam) as c:
        r = c.post("/api/query", json={"question": "What is the placeholder thing?", "language": "hi-IN"}).json()["data"]
    assert sarvam.calls("/translate")[0]["source_language_code"] == "auto"
    assert r["localization"]["question_translated"] is False
    assert r["localization"]["answer_text"].startswith("<hi-IN> "), "the answer still comes back in Hindi"


def test_changed_numbers_are_flagged(sarvam):
    sarvam.drop_numbers = True
    with answering_client(sarvam) as c:
        loc = c.post("/api/query", json={"question": "ಪ್ರಶ್ನೆ ಇಲ್ಲಿ", "language": "kn-IN"}).json()["data"]["localization"]
    assert "numbers_changed" in loc["warnings"]


def test_missing_evidence_is_explained_in_the_selected_language(sarvam):
    with make_client(sarvam=sarvam.client()) as c:  # knowledge base not configured
        r = c.post("/api/query", json={"question": "ತಮಿಳು ಪ್ರಶ್ನೆ", "language": "ta-IN"}).json()["data"]
    assert r["answer"] is None and r["localization"]["answer_text"] is None
    assert r["localization"]["status_message"].startswith("<ta-IN> ")
    assert all(n.startswith("<ta-IN> ") for n in r["localization"]["notices"])


def test_failed_answer_translation_falls_back_to_english_explicitly(sarvam):
    with answering_client(sarvam) as c:
        original = sarvam.handler

        def flaky(request):  # question translation works, answer translation fails
            body = json.loads(request.content) if request.url.path == "/translate" else {}
            if body.get("target_language_code") == "te-IN":
                return httpx.Response(500, json={"error": {"code": "internal_server_error"}})
            return original(request)

        sarvam.handler = flaky
        c.app.state.services.multilingual.client = SarvamClient("k", transport=httpx.MockTransport(flaky))
        loc = c.post("/api/query", json={"question": "తెలుగు ప్రశ్న", "language": "te-IN"}).json()["data"]["localization"]
    assert loc["complete"] is False and "partial_translation" in loc["warnings"]
    assert loc["answer_text"] == "ALPHA 5 mg"


def test_failed_question_translation_is_an_explicit_error(sarvam):
    sarvam.fail["/translate"] = (403, "invalid_api_key_error")
    with answering_client(sarvam) as c:
        r = c.post("/api/query", json={"question": "ಪ್ರಶ್ನೆ ಇಲ್ಲಿ", "language": "kn-IN"})
    assert r.status_code == 502 and r.json()["error"]["code"] == "voice_auth_failed"


def test_non_english_without_key_says_how_to_enable(sarvam):
    with answering_client(None) as c:
        r = c.post("/api/query", json={"question": "ಪ್ರಶ್ನೆ ಇಲ್ಲಿ", "language": "kn-IN"})
        langs = c.get("/api/languages").json()["data"]
    assert r.status_code == 503 and "SARVAM_API_KEY" in r.json()["error"]["message"]
    assert langs["multilingual_configured"] is False
    assert [l["code"] for l in langs["languages"] if l["available"]] == ["en-IN"]


def test_unsupported_language_is_rejected(sarvam):
    with answering_client(sarvam) as c:
        r = c.post("/api/query", json={"question": "Bonjour docteur", "language": "fr-FR"})
    assert r.status_code == 422


# ---------------------------------------------------------------- /api/transcribe
def upload(c, data=WAV, ctype="audio/webm", language="kn-IN"):
    return c.post("/api/transcribe", files={"file": ("q.webm", data, ctype)}, data={"language": language})


def test_transcribe_returns_editable_text(sarvam):
    with answering_client(sarvam) as c:
        r = upload(c)
    assert r.status_code == 200 and r.json()["data"]["transcript"] == sarvam.transcript


def test_transcribe_rejects_bad_uploads(sarvam):
    with answering_client(sarvam) as c:
        assert upload(c, data=b"tiny").json()["error"]["code"] == "empty_recording"
        assert upload(c, ctype="text/plain").status_code == 415
        assert upload(c, language="xx-XX").status_code == 422
        too_big = c.app.state.services.settings.max_audio_bytes + 10
        assert upload(c, data=b"\0" * too_big).status_code == 413
    assert sarvam.requests == []


def test_transcribe_handles_no_speech_and_api_errors(sarvam):
    with answering_client(sarvam) as c:
        sarvam.transcript = "  "
        assert upload(c).json()["error"]["code"] == "no_speech"
        sarvam.timeout.add("/speech-to-text")
        r = upload(c)
    assert r.status_code == 504 and r.json()["error"]["code"] == "voice_timeout"


def test_transcribe_without_key(sarvam):
    with answering_client(None) as c:
        assert upload(c).status_code == 503


# ---------------------------------------------------------------- /api/speech
def test_speech_returns_wav(sarvam):
    with answering_client(sarvam) as c:
        r = c.post("/api/speech", json={"text": "<kn-IN> ALPHA", "language": "kn-IN"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content == WAV


def test_speech_validation_and_errors(sarvam):
    with answering_client(sarvam) as c:
        assert c.post("/api/speech", json={"text": "  ", "language": "kn-IN"}).status_code == 422
        assert c.post("/api/speech", json={"text": "x" * 5000, "language": "kn-IN"}).status_code == 422
        sarvam.fail["/text-to-speech"] = (429, "rate_limit_exceeded_error")
        r = c.post("/api/speech", json={"text": "hello", "language": "hi-IN"})
    assert r.status_code == 429 and r.json()["error"]["code"] == "voice_rate_limited"


def test_inline_comment_after_key_is_ignored(caplog):
    from app.config import Settings
    from app.container import sarvam_client

    settings = Settings(_env_file=None, SARVAM_API_KEY="sk_abc123 # from the dashboard")
    client = sarvam_client(settings)
    assert client._http.headers["api-subscription-key"] == "sk_abc123"
    assert "sk_abc123" not in caplog.text and "dashboard" not in caplog.text
    assert sarvam_client(Settings(_env_file=None, SARVAM_API_KEY="   ")) is None
