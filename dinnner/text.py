"""Text preparation (PDF extraction, sentence splitting) and result export."""

import io
import re
from typing import List

import pandas as pd

from .inference import SentenceResult

# --------------------------------------------------------------------------
# PDF extraction
# --------------------------------------------------------------------------

_START_RE = re.compile(r"(?im)^\s*(?:\d+\.?\s*)?(abstract|introduction)\b")
_END_RE = re.compile(r"(?im)^\s*(?:\d+\.?\s*)?(references|bibliography|literature cited)\s*$")
_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:\d+(?:\.\d+)*\.?[ \t]*)?"
    r"(abstract|introduction|background|methods?|materials and methods|results|"
    r"results and discussion|discussion|conclusions?|acknowledge?ments|keywords)[ \t]*:?[ \t]*$"
)


def extract_pdf_text(pdf_bytes: bytes, max_pages: int, body_only: bool = True):
    """Return (text, pages_read, total_pages) for a PDF.

    With body_only, text before the Abstract/Introduction heading and after the
    last References heading is dropped (same idea as the original demo, but
    the abstract is kept and the References match is anchored to a heading).
    """
    import pymupdf

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        total = doc.page_count
        raw = "\n".join(doc[i].get_text("text") for i in range(min(total, max_pages)))

    if body_only:
        start = _START_RE.search(raw)
        ends = list(_END_RE.finditer(raw))
        s = start.start() if start else 0
        e = ends[-1].start() if ends and ends[-1].start() > s else len(raw)
        raw = raw[s:e]

    # Re-join words hyphenated across line breaks ("pro-\ntein" -> "protein")
    # but keep real hyphens in names such as "IL-\n6" -> "IL-6".
    raw = re.sub(r"([a-z])-\n([a-z])", r"\1\2", raw)
    raw = re.sub(r"(\w)-\n(\w)", r"\1-\2", raw)
    # Section headings become paragraph breaks so they are not glued onto
    # the first sentence of the section ("Abstract Upon insulin ...").
    raw = _HEADING_RE.sub("\n\n", raw)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", raw)]
    text = "\n\n".join(p for p in paragraphs if p)
    return text, min(total, max_pages), total


# --------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------

_ABBREV = r"(?:e\.g|i\.e|et al|etc|Fig|Figs|Eq|Eqs|Ref|Refs|vs|approx|ca|cf|resp|No|Vol|Dr|Mr|Mrs|Ms|Prof)\."
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def _regex_split(text: str) -> List[str]:
    protected = re.sub(_ABBREV, lambda m: m.group().replace(".", "․"), text)
    return [s.replace("․", ".").strip() for s in _SPLIT_RE.split(protected) if s.strip()]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences; NLTK Punkt if available, regex otherwise."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    try:
        from nltk.tokenize import sent_tokenize

        sentences = [s for p in paragraphs for s in sent_tokenize(re.sub(r"\s+", " ", p))]
    except LookupError:
        sentences = [s for p in paragraphs for s in _regex_split(re.sub(r"\s+", " ", p))]
    return [s.strip() for s in sentences if s.strip()]


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def annotate_inline(res: SentenceResult) -> str:
    """Sentence with entities marked inline, e.g. '<Simple_chemical>ATP</Simple_chemical>'."""
    out, pos = [], 0
    for e in res.entities:
        out += [res.text[pos:e.start], f"<{e.type}>{e.text}</{e.type}>"]
        pos = e.end
    out.append(res.text[pos:])
    return "".join(out)


def entities_frame(docs: dict) -> pd.DataFrame:
    """One row per predicted entity. docs = {source name: [SentenceResult]}."""
    rows = [
        {
            "source": src,
            "sentence_id": r.index + 1,
            "entity": e.text,
            "type": e.type,
            "start_char": e.start,
            "end_char": e.end,
            "p_entity": round(e.p_entity, 4),
            "p_type": round(e.p_type, 4),
            "sentence": r.text,
        }
        for src, results in docs.items()
        for r in results
        for e in r.entities
    ]
    cols = ["source", "sentence_id", "entity", "type", "start_char", "end_char", "p_entity", "p_type", "sentence"]
    return pd.DataFrame(rows, columns=cols)


def sentences_frame(docs: dict) -> pd.DataFrame:
    """One row per sentence with an inline-annotated copy."""
    rows = [
        {
            "source": src,
            "sentence_id": r.index + 1,
            "sentence": r.text,
            "annotated_sentence": annotate_inline(r),
            "n_entities": len(r.entities),
        }
        for src, results in docs.items()
        for r in results
    ]
    return pd.DataFrame(rows, columns=["source", "sentence_id", "sentence", "annotated_sentence", "n_entities"])


def conll_text(docs: dict) -> str:
    """Token-level IOB output (word<TAB>label), blank line between sentences."""
    buf = io.StringIO()
    for src, results in docs.items():
        buf.write(f"# source = {src}\n")
        for r in results:
            for w, lab in zip(r.words, r.labels):
                buf.write(f"{w}\t{lab}\n")
            buf.write("\n")
    return buf.getvalue()
