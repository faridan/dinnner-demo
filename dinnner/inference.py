"""Turn raw sentences into DINNNER entity predictions.

Improvements over the original MACSYS demo code:
  * character offsets are kept for every word, so entities are shown exactly
    as written in the source text ("ITF-2b", not "ITF - 2b");
  * sentences longer than the 192-subword training window are split into
    chunks instead of being silently truncated;
  * sentences are batched and padded to the longest item, which is much
    faster on CPU than padding every sentence to 192;
  * all four BioNLP13PC entity types are returned, not only genes;
  * each entity carries the two DINNNER probabilities as confidence scores.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

WORD_RE = re.compile(r"\w+|[^\w\s]")


@dataclass
class Entity:
    text: str
    type: str
    start: int            # character offset within the sentence
    end: int
    p_entity: float       # mean p(entity present) over the entity's words
    p_type: float         # mean p(label) from the multiclass head


@dataclass
class SentenceResult:
    index: int
    text: str
    words: List[str] = field(default_factory=list)
    spans: List[tuple] = field(default_factory=list)    # (start, end) per word
    labels: List[str] = field(default_factory=list)     # IOB label per word
    entities: List[Entity] = field(default_factory=list)


def split_words(sentence: str):
    """Word-level tokenisation matching the original demo, with offsets."""
    matches = list(WORD_RE.finditer(sentence))
    return [m.group() for m in matches], [m.span() for m in matches]


def _chunk_words(words, tokenizer, max_subwords):
    """Split a word list into chunks that fit the model's subword window."""
    if not words:
        return []
    enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
    counts = [0] * len(words)
    for wid in enc.word_ids():
        if wid is not None:
            counts[wid] += 1

    chunks, start, used = [], 0, 0
    for i, n in enumerate(counts):
        n = max(n, 1)
        if used + n > max_subwords and i > start:
            chunks.append((start, i))
            start, used = i, 0
        used += n
    chunks.append((start, len(words)))
    return chunks


def decode_entities(words, spans, labels, p_ent, p_lab, sentence):
    """Group IOB labels into entity spans (lenient: a stray I- opens an entity)."""
    entities, cur = [], None

    def close():
        if cur is not None:
            s, e = spans[cur["first"]][0], spans[cur["last"]][1]
            idx = range(cur["first"], cur["last"] + 1)
            entities.append(Entity(
                text=sentence[s:e],
                type=cur["type"],
                start=s,
                end=e,
                p_entity=sum(p_ent[i] for i in idx) / len(idx),
                p_type=sum(p_lab[i] for i in idx) / len(idx),
            ))

    for i, lab in enumerate(labels):
        if lab == "O":
            close()
            cur = None
            continue
        tag, etype = lab.split("-", 1)
        if tag == "B" or cur is None or cur["type"] != etype:
            close()
            cur = {"type": etype, "first": i, "last": i}
        else:
            cur["last"] = i
    close()
    return entities


@torch.no_grad()
def predict(
    sentences: List[str],
    model,
    tokenizer,
    id2label: dict,
    device: torch.device,
    max_length: int = 192,
    batch_size: int = 16,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[SentenceResult]:
    results, jobs = [], []   # job = (sentence index, word start, word end)
    for si, sent in enumerate(sentences):
        words, spans = split_words(sent)
        n = len(words)
        results.append(SentenceResult(si, sent, words, spans, ["O"] * n))
        results[-1]._p_ent = [0.0] * n
        results[-1]._p_lab = [0.0] * n
        for a, b in _chunk_words(words, tokenizer, max_length - 2):
            jobs.append((si, a, b))

    for bstart in range(0, len(jobs), batch_size):
        batch = jobs[bstart:bstart + batch_size]
        word_lists = [results[si].words[a:b] for si, a, b in batch]
        enc = tokenizer(
            word_lists,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        logits_bin, logits_mc = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        p_bin = torch.sigmoid(logits_bin).squeeze(-1)          # (B, T)
        p_mc = torch.softmax(logits_mc, dim=-1)                # (B, T, C)
        joint = p_bin.unsqueeze(-1) * p_mc                     # DINNNER joint rule
        pred = joint.argmax(-1)
        p_pred = p_mc.gather(-1, pred.unsqueeze(-1)).squeeze(-1)

        for row, (si, a, _) in enumerate(batch):
            res, seen = results[si], set()
            for t, wid in enumerate(enc.word_ids(row)):
                if wid is None or wid in seen:
                    continue  # special token or non-first subword
                seen.add(wid)
                res.labels[a + wid] = id2label[int(pred[row, t])]
                res._p_ent[a + wid] = float(p_bin[row, t])
                res._p_lab[a + wid] = float(p_pred[row, t])

        if progress:
            progress(min(bstart + batch_size, len(jobs)), len(jobs))

    for res in results:
        res.entities = decode_entities(res.words, res.spans, res.labels, res._p_ent, res._p_lab, res.text)
        del res._p_ent, res._p_lab
    return results
