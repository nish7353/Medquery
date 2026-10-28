"""Text cleaning and passage chunking.

Documents are kept as a list of blocks (headings and paragraphs). Chunks never
cross a section heading, are about `target_words` long, and overlap by a few
sentences so an answer near a chunk boundary is still retrievable.
"""

import html
import re
from html.parser import HTMLParser

from knowledge.db import Block

_WS = re.compile(r"[ \t\r\f\v\u00a0]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def clean(text: str) -> str:
    return _WS.sub(" ", html.unescape(text)).strip()


def paragraphs(text: str) -> list[Block]:
    """Split plain text on blank lines into paragraph blocks."""
    return [Block("para", p) for p in (clean(x.replace("\n", " ")) for x in re.split(r"\n\s*\n", text)) if p]


class _HtmlToBlocks(HTMLParser):
    _BREAKS = {"p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self._buf: list[str] = []
        self._kind = "para"
        self._skip = 0

    def _flush(self):
        text = clean("".join(self._buf))
        if text:
            self.blocks.append(Block(self._kind, text))
        self._buf, self._kind = [], "para"

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in self._BREAKS:
            self._flush()
            if tag.startswith("h"):
                self._kind = "heading"
            elif tag == "li":
                self._buf.append("- ")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in self._BREAKS:
            self._flush()

    def handle_data(self, data):
        if not self._skip:
            self._buf.append(data)

    def close(self):
        super().close()
        self._flush()


def html_to_blocks(markup: str) -> list[Block]:
    parser = _HtmlToBlocks()
    parser.feed(markup)
    parser.close()
    # Consecutive list items read better as one paragraph.
    merged: list[Block] = []
    for b in parser.blocks:
        if merged and b.kind == "para" and b.text.startswith("- ") and merged[-1].kind == "para":
            merged[-1] = Block("para", f"{merged[-1].text}\n{b.text}")
        else:
            merged.append(b)
    return merged


def _sentences(text: str, max_words: int) -> list[str]:
    out = []
    for line in text.split("\n"):
        for s in _SENTENCE_END.split(line):
            words = s.split()
            # Very long "sentences" (unpunctuated lists, tables) are cut into slices.
            out.extend(" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words))
    return out


def _join(sentences: list[str]) -> str:
    # List items keep their own line; ordinary sentences flow together.
    return "".join(("\n" if s.startswith("- ") and i else " " if i else "") + s for i, s in enumerate(sentences))


def _window(sentences: list[str], target: int, overlap: int) -> list[str]:
    chunks, current, words = [], [], 0
    for s in sentences:
        n = len(s.split())
        if current and words + n > target:
            chunks.append(_join(current))
            # Carry trailing sentences forward as overlap.
            carry, carried = [], 0
            for prev in reversed(current):
                carried += len(prev.split())
                if carried > overlap:
                    break
                carry.insert(0, prev)
            current, words = carry, sum(len(c.split()) for c in carry)
        current.append(s)
        words += n
    if current:
        chunks.append(_join(current))
    return chunks


def chunk_blocks(blocks: list[Block], target_words: int = 180, overlap_words: int = 30) -> list[tuple[str | None, str]]:
    """Return (section heading, chunk text) pairs. The heading is prefixed to the text."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for b in blocks:
        if b.kind == "heading":
            sections.append((b.text, []))
        else:
            sections[-1][1].extend(_sentences(b.text, target_words))

    chunks = []
    for heading, sentences in sections:
        for text in _window(sentences, target_words, overlap_words):
            chunks.append((heading, f"{heading}\n{text}" if heading else text))
    return chunks


# Sentences that carry no medical information: site navigation, attribution lines
# and GARD's explanation of its symptom-frequency table (repeated in ~2,200 answers).
# Kept deliberately narrow: repeated sentences such as inheritance patterns ARE answers.
_BOILERPLATE = re.compile(
    r"^(?:NIH: .*|Centers for Disease Control and Prevention"
    r"|SUPPLEMENTARY INFORMATION:.*|Supplementary (?:data|information) (?:are|is) available.*"
    r"|Much of this information comes from Orphanet.*|The Human Phenotype Ontology provides the following list.*"
    r"|If the information is available, the table below includes.*|The frequency of a sign or symptom is usually.*"
    r"|The frequency may also be listed as a fraction\.|The first number of the fraction is.*"
    r"|For example, a frequency of 25/25 means.*|Because these frequencies are based on a specific study.*"
    r"|Sometimes, no information on frequency is available\.|In these cases, the sign or symptom may be rare or common\."
    r"|You can use the MedlinePlus Medical Dictionary.*"
    r"|These have been retrieved from NCI's listing of clinical trials\.|(?:General i|I)nformation about clinical trials is available from the NCI website\."
    r"|For more specific results, refine the search.*|See the Treatment Options section that follows.*)$"
)


def _plain_sentences(text: str) -> list[str]:
    return [x.strip() for line in text.split("\n") for x in _SENTENCE_END.split(line) if x.strip()]


def strip_boilerplate(blocks: list[Block]) -> list[Block]:
    out = []
    for b in blocks:
        if b.kind == "para":
            kept = [s for s in _plain_sentences(b.text) if not _BOILERPLATE.match(s)]
            if not kept:
                continue
            b = Block("para", _join(kept))
        out.append(b)
    return out
