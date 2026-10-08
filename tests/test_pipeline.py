"""Offline tests using a tiny stand-in checkpoint (no internet needed).

    pip install pytest && pytest -q
"""

import os
import sys

import pandas as pd
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from dinnner.config import MODELS, ModelSpec  # noqa: E402
from dinnner.examples import EXAMPLES  # noqa: E402
from dinnner.inference import predict, split_words  # noqa: E402
from dinnner.model import load_model  # noqa: E402
from dinnner.text import conll_text, entities_frame, sentences_frame, split_sentences  # noqa: E402
from make_standin_model import LABELS, build, gazetteer_labels  # noqa: E402

SENTS = [s for t in EXAMPLES.values() for s in split_sentences(t)]


@pytest.fixture(scope="module")
def standin(tmp_path_factory):
    out = tmp_path_factory.mktemp("standin")
    enc_dir, ckpt = build(str(out), SENTS)
    spec = ModelSpec(
        key="standin", display_name="stand-in", dataset="test", encoder=enc_dir,
        id2label=dict(enumerate(LABELS)), weights_repo="none/none", weights_file="x.pth", local_path=ckpt,
    )
    model, tok = load_model(spec, torch.device("cpu"))
    return spec, model, tok


def test_registry_matches_training_labels():
    spec = MODELS["pubmedbert-bionlp13pc"]
    assert list(spec.id2label.values()) == LABELS
    assert spec.max_length == 192


def test_sentence_split():
    assert len(SENTS) == 6
    assert split_sentences("See Fig. 2 for details. AKT is active.") == ["See Fig. 2 for details.", "AKT is active."]


def test_recovers_labels_and_original_text(standin):
    spec, model, tok = standin
    res = predict(SENTS, model, tok, spec.id2label, torch.device("cpu"))
    for r in res:
        assert r.labels == gazetteer_labels(r.words), r.text
    texts = {e.text for r in res for e in r.entities}
    # hyphenated names keep their original form instead of "IRS - 1"
    assert {"IRS-1", "caspase-9", "BCL-2", "TNF-alpha", "PI3K complex"} <= texts
    for r in res:
        for e in r.entities:
            assert r.text[e.start:e.end] == e.text
            assert 0.0 <= e.p_entity <= 1.0 and 0.0 <= e.p_type <= 1.0


def test_chunking_matches_unchunked(standin):
    spec, model, tok = standin
    long_sentence = " ".join(SENTS)  # one very long "sentence"
    full = predict([long_sentence], model, tok, spec.id2label, torch.device("cpu"), max_length=256)[0]
    chunked = predict([long_sentence], model, tok, spec.id2label, torch.device("cpu"), max_length=24, batch_size=3)[0]
    assert len(chunked.labels) == len(split_words(long_sentence)[0])
    # chunk boundaries can only change words right at the cut; most must agree
    agree = sum(a == b for a, b in zip(full.labels, chunked.labels)) / len(full.labels)
    assert agree > 0.9
    assert "O" in chunked.labels and any(l != "O" for l in chunked.labels[-20:])  # tail not truncated


def test_empty_and_symbol_inputs(standin):
    spec, model, tok = standin
    res = predict(["", "???", "ATP"], model, tok, spec.id2label, torch.device("cpu"))
    assert res[0].entities == [] and res[0].labels == []
    assert len(res[2].labels) == 1


def test_exports(standin):
    spec, model, tok = standin
    docs = {"doc": predict(SENTS, model, tok, spec.id2label, torch.device("cpu"))}
    ents = entities_frame(docs)
    sents = sentences_frame(docs)
    assert len(sents) == len(SENTS)
    assert len(ents) == sum(sents["n_entities"])
    assert "<Simple_chemical>ATP</Simple_chemical>" in " ".join(sents["annotated_sentence"])
    # round trip through CSV keeps commas/quotes inside sentences intact
    import io
    back = pd.read_csv(io.StringIO(ents.to_csv(index=False)))
    assert back["sentence"].tolist() == ents["sentence"].tolist()
    iob = conll_text(docs)
    assert "IRS\tB-Gene_or_gene_product\n-\tI-Gene_or_gene_product\n1\tI-Gene_or_gene_product" in iob


def test_checkpoint_mismatch_is_reported(standin, tmp_path):
    spec, model, _ = standin
    bad = {k: v for k, v in model.state_dict().items() if "binary" not in k}
    path = tmp_path / "bad.pth"
    torch.save(bad, path)
    bad_spec = ModelSpec(**{**spec.__dict__, "local_path": str(path)})
    with pytest.raises(RuntimeError, match="does not match"):
        load_model(bad_spec, torch.device("cpu"))


def test_pdf_extraction():
    import pymupdf

    from dinnner.text import extract_pdf_text

    doc = pymupdf.open()
    page = doc.new_page()
    body = ("A Title\nAuthor, University\n\nAbstract\nAKT binds PIP3 at the plasma mem-\nbrane via IL-\n6.\n\n"
            "1. Introduction\nBAX acts on mitochondria.\n\nReferences\n[1] Dropped reference text.")
    y = 50
    for line in body.split("\n"):
        page.insert_text((50, y), line, fontsize=10)
        y += 14
    text, read, total = extract_pdf_text(doc.tobytes(), max_pages=10, body_only=True)
    sents = split_sentences(text)
    assert (read, total) == (1, 1)
    assert sents == ["AKT binds PIP3 at the plasma membrane via IL-6.", "BAX acts on mitochondria."]
    full, _, _ = extract_pdf_text(doc.tobytes(), max_pages=10, body_only=False)
    assert "Dropped reference" in full and "A Title" in full


def test_app_handlers(standin, monkeypatch):
    """Run the Gradio handlers end to end (no browser) with the stand-in model."""
    spec, model, tok = standin
    import app

    monkeypatch.setitem(app._MODELS, app.DEFAULT_MODEL, (model, tok))
    out = app.analyse_text(" ".join(EXAMPLES.values()), app.DEFAULT_MODEL, progress=lambda *a, **k: None)
    state = out[0]
    assert sum(len(r) for r in state["results"].values()) == 6
    annotated, table, summary, bars = out[6:10]
    assert "IRS-1" in annotated and "<sub>" in annotated
    assert len(table) == len(entities_frame(state["results"]))
    paths = [o["value"] for o in out[10:13]]
    assert all(os.path.exists(p) for p in paths)
    # filtering re-renders without re-running the model
    annotated2, table2, _, _ = app.render_views(state, ["Simple_chemical"], "Pasted text", True)
    assert set(table2["Type"]) == {"Simple chemical"}
    assert "Gene or gene product" not in annotated2
    with pytest.raises(app.gr.Error):
        app.analyse_text("   ", app.DEFAULT_MODEL, progress=lambda *a, **k: None)
