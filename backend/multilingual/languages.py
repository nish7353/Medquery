"""Languages MedQuery accepts, and which Sarvam AI service supports each one.

Support was checked against docs.sarvam.ai (October 2026):

* speech-to-text  `saaras:v4`  - 22 Indian languages + en-IN
* translation     `mayura:v1`  - en-IN + 10 Indian languages (hi bn ta te gu kn ml mr pa od)
* text-to-speech  `bulbul:v3`  - the same 11 languages

To add a language, append it to LANGUAGES with the services that support it.
"""

from dataclasses import asdict, dataclass

# The language the knowledge base, BERT encoder and QA model work in.
PIPELINE_LANGUAGE = "en-IN"


@dataclass(frozen=True)
class Language:
    code: str           # BCP-47 code used by every Sarvam API
    name: str           # English name
    native_name: str    # name in its own script, for the language menu
    speech_to_text: bool = True
    translation: bool = True
    text_to_speech: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


LANGUAGES: dict[str, Language] = {lang.code: lang for lang in (
    Language("en-IN", "English", "English"),
    Language("hi-IN", "Hindi", "हिन्दी"),
    Language("kn-IN", "Kannada", "ಕನ್ನಡ"),
    Language("ta-IN", "Tamil", "தமிழ்"),
    Language("te-IN", "Telugu", "తెలుగు"),
)}


class UnsupportedLanguage(ValueError):
    def __init__(self, code: str, service: str | None = None):
        self.code, self.service = code, service
        what = f" for {service.replace('_', ' ')}" if service else ""
        super().__init__(f"Language '{code}' is not supported{what}.")


def get_language(code: str, service: str | None = None) -> Language:
    """Look up a language, checking that `service` ("speech_to_text", ...) supports it."""
    lang = LANGUAGES.get(code)
    if lang is None:
        raise UnsupportedLanguage(code)
    if service and not getattr(lang, service):
        raise UnsupportedLanguage(code, service)
    return lang


def needs_translation(code: str) -> bool:
    return code != PIPELINE_LANGUAGE
