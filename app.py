"""DINNNER online tool: biomedical named entity recognition in the browser.

Runs as a Gradio app on a Hugging Face Space (ZeroGPU or CPU). Locally:  python app.py
"""

import os

# On ZeroGPU Spaces, `spaces` must be imported before torch touches CUDA.
try:
    import spaces
except ImportError:  # not installed locally; everything runs on CPU/GPU as usual
    spaces = None
ON_ZEROGPU = spaces is not None and os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "true")

import html  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from dinnner import DEFAULT_MODEL, MODELS, load_model, predict  # noqa: E402
from dinnner.config import BATCH_SIZE, MAX_PDF_FILES, MAX_PDF_PAGES, MAX_TEXT_CHARS  # noqa: E402
from dinnner.examples import EXAMPLES  # noqa: E402
from dinnner.text import conll_text, entities_frame, extract_pdf_text, sentences_frame, split_sentences  # noqa: E402

GITHUB_URL = os.environ.get("DINNNER_GITHUB_URL", "https://github.com/faridan/dinnner-demo")
PAPER_URL = os.environ.get("DINNNER_PAPER_URL", "")
SHOW_LIMIT = 300  # sentences rendered in the annotated view (all are in the downloads)

# Colour per entity type: (background, border). Readable in light and dark themes.
PALETTE = [
    ("rgba(59,130,246,0.18)", "#3b82f6"),   # blue
    ("rgba(16,185,129,0.20)", "#10b981"),   # green
    ("rgba(245,158,11,0.22)", "#f59e0b"),   # amber
    ("rgba(236,72,153,0.18)", "#ec4899"),   # pink
    ("rgba(139,92,246,0.18)", "#8b5cf6"),   # violet
    ("rgba(20,184,166,0.20)", "#14b8a6"),   # teal
]

CSS = """
.ent {padding: 0.05em 0.3em; border-radius: 0.3em; border-bottom: 2px solid; line-height: 2;}
.ent sub {font-size: 0.62em; font-weight: 600; letter-spacing: 0.02em; margin-left: 0.3em;
          opacity: 0.85; text-transform: uppercase; white-space: nowrap;}
.sent {margin: 0 0 0.6em 0; line-height: 1.9;}
.sent-id {display: inline-block; min-width: 2.4em; font-size: 0.75em; opacity: 0.55;}
.metrics {display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 0.25rem 0 0.5rem;}
.metric {flex: 1 1 8.5rem; padding: 0.6rem 0.8rem; border-radius: 0.6rem;
         border: 1px solid var(--border-color-primary); background: var(--block-background-fill);}
.metric .v {font-size: 1.6rem; font-weight: 600; line-height: 1.2;}
.metric .k {font-size: 0.8rem; opacity: 0.7;}
.metric.typed {border-left-width: 4px;}
.legend-row {margin-bottom: 0.5em;}
.legend-row small {display: block; opacity: 0.7; margin-top: 0.1em;}
.bars .row {display: grid; grid-template-columns: 11rem minmax(0, 32rem) 2.5rem; align-items: center; gap: 0.6rem; margin: 0.35rem 0;}
.bars .bar {height: 0.9rem; border-radius: 0 0.25rem 0.25rem 0;}
.bars .n {font-variant-numeric: tabular-nums; text-align: right;}
.muted {opacity: 0.7; font-size: 0.9em;}
"""


# --------------------------------------------------------------------------
# Model (loaded once at start-up so visitors do not wait on the first click)
# --------------------------------------------------------------------------

_MODELS, _LOAD_ERRORS = {}, {}
# On ZeroGPU the model is placed on "cuda" at start-up (an emulated device);
# a real GPU is attached only while a @spaces.GPU function runs.
DEVICE = torch.device("cuda" if ON_ZEROGPU or torch.cuda.is_available() else "cpu")


def get_model(key):
    if key not in _MODELS:
        try:
            _MODELS[key] = load_model(MODELS[key], DEVICE)
            _LOAD_ERRORS.pop(key, None)
        except Exception as exc:  # weights missing, network down, wrong checkpoint...
            _LOAD_ERRORS[key] = str(exc)
            raise gr.Error(f"The model could not be loaded: {exc}", duration=None)
    return _MODELS[key]


def _prepare_nltk():
    try:
        import nltk

        nltk.download("punkt_tab", quiet=True, raise_on_error=True)
    except Exception:
        pass  # the built-in regex sentence splitter is used instead


# --------------------------------------------------------------------------
# HTML helpers
# --------------------------------------------------------------------------

def type_colours(spec):
    types = list(spec.entity_info) or sorted({l[2:] for l in spec.id2label.values() if l != "O"})
    return {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(types)}


def pretty(etype):
    return etype.replace("_", " ")


def ent_html(text, etype, colours, tag=True):
    bg, border = colours.get(etype, PALETTE[-1])
    label = f"<sub>{html.escape(pretty(etype))}</sub>" if tag else ""
    return f'<span class="ent" style="background:{bg};border-color:{border}">{html.escape(text)}{label}</span>'


def sentence_html(res, colours, keep):
    out, pos = [], 0
    for e in res.entities:
        if e.type not in keep:
            continue
        out += [html.escape(res.text[pos:e.start]), ent_html(e.text, e.type, colours)]
        pos = e.end
    out.append(html.escape(res.text[pos:]))
    return f'<p class="sent"><span class="sent-id">{res.index + 1}</span>{"".join(out)}</p>'


def legend_html(spec):
    colours = type_colours(spec)
    return "".join(
        f'<div class="legend-row">{ent_html(pretty(t), t, colours, tag=False)}<small>{html.escape(d)}</small></div>'
        for t, d in spec.entity_info.items()
    )


# --------------------------------------------------------------------------
# Processing
# --------------------------------------------------------------------------

def _tag_docs(docs, model_key, progress_cb=None):
    """Run DINNNER over {source: [sentences]}; returns {source: [SentenceResult]}."""
    spec = MODELS[model_key]
    model, tokenizer = _MODELS[model_key]
    total = sum(len(s) for s in docs.values())
    results, done = {}, 0
    for src, sents in docs.items():
        def cb(d, n, base=done, k=len(sents), src=src):
            if progress_cb:
                progress_cb((base + k * d / max(n, 1)) / total, src)
        results[src] = predict(sents, model, tokenizer, spec.id2label, DEVICE,
                               max_length=spec.max_length, batch_size=BATCH_SIZE, progress=cb)
        done += len(sents)
    return results


def _gpu_seconds(docs, model_key):
    """GPU time to reserve on ZeroGPU: short requests get better queue priority."""
    n = sum(len(s) for s in docs.values())
    return int(min(120, 15 + n / 25))


def _tag_docs_gpu(docs, model_key):
    return _tag_docs(docs, model_key)


if ON_ZEROGPU:
    _tag_docs_gpu = spaces.GPU(duration=_gpu_seconds)(_tag_docs_gpu)


def _run(docs, model_key, notes, progress):
    if not any(docs.values()):
        raise gr.Error("No sentences were found in the input.")
    spec = MODELS[model_key]
    progress(0, desc="Loading model...")
    get_model(model_key)  # surfaces loading errors before any GPU is requested

    t0 = time.time()
    if ON_ZEROGPU:
        progress(0.05, desc="Tagging entities on GPU...")
        results = _tag_docs_gpu(docs, model_key)
    else:
        results = _tag_docs(docs, model_key, lambda f, src: progress(f, desc=f"Tagging entities in {src}"))

    # Write the downloads once per run (they always contain everything).
    out_dir = tempfile.mkdtemp(prefix="dinnner_")
    stem = "dinnner_" + (list(results)[0].rsplit(".", 1)[0][:40].replace(" ", "_") if len(results) == 1 else "results")
    paths = [os.path.join(out_dir, f"{stem}_{s}") for s in ("entities.csv", "sentences.csv", "iob.tsv")]
    entities_frame(results).to_csv(paths[0], index=False)
    sentences_frame(results).to_csv(paths[1], index=False)
    with open(paths[2], "w", encoding="utf-8") as f:
        f.write(conll_text(results))

    state = {"results": results, "model_key": model_key, "seconds": time.time() - t0, "notes": notes}
    types = list(type_colours(spec))
    sources = list(results)
    doc = sources[0]
    only_hits = len(results[doc]) > 40
    return (
        state,
        gr.update(visible=True),                                           # results column
        summary_html(state),
        gr.update(choices=[(pretty(t), t) for t in types], value=types),   # type filter
        gr.update(choices=sources, value=doc, visible=len(sources) > 1),   # document picker
        gr.update(value=only_hits),
        *render_views(state, types, doc, only_hits),
        *[gr.update(value=p, visible=True) for p in paths],
    )


def analyse_text(text, model_key, progress=gr.Progress()):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Please paste some text first, or pick an example.")
    notes = []
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        notes.append(f"Only the first {MAX_TEXT_CHARS:,} characters were processed.")
    return _run({"Pasted text": split_sentences(text)}, model_key, notes, progress)


def analyse_pdfs(files, body_only, model_key, progress=gr.Progress()):
    if not files:
        raise gr.Error("Please upload at least one PDF.")
    docs, notes = {}, []
    if len(files) > MAX_PDF_FILES:
        notes.append(f"Only the first {MAX_PDF_FILES} files were processed.")
    for path in files[:MAX_PDF_FILES]:
        path = getattr(path, "name", path)
        name = os.path.basename(path)
        try:
            with open(path, "rb") as f:
                text, read, total = extract_pdf_text(f.read(), MAX_PDF_PAGES, body_only)
        except Exception as exc:
            notes.append(f"Could not read {name}: {exc}")
            continue
        if not text:
            notes.append(f"No extractable text in {name} (it may be a scanned image).")
            continue
        if read < total:
            notes.append(f"{name}: only the first {read} of {total} pages were processed.")
        docs[name] = split_sentences(text)
    if not docs:
        raise gr.Error(" ".join(notes) or "No text could be read from the PDFs.")
    return _run(docs, model_key, notes, progress)


# --------------------------------------------------------------------------
# Views (re-rendered from stored results when filters change, no re-run)
# --------------------------------------------------------------------------

def summary_html(state):
    spec = MODELS[state["model_key"]]
    colours = type_colours(spec)
    ents = entities_frame(state["results"])
    n_sent = sum(len(r) for r in state["results"].values())
    cards = [("Sentences", n_sent, None), ("Entities", len(ents), None)]
    cards += [(pretty(t), int((ents["type"] == t).sum()), colours[t][1]) for t in colours]
    body = "".join(
        f'<div class="metric{" typed" if c else ""}"{f" style=border-left-color:{c}" if c else ""}>'
        f'<div class="v">{v:,}</div><div class="k">{html.escape(k)}</div></div>'
        for k, v, c in cards
    )
    notes = "".join(f"<div class='muted'>ℹ️ {html.escape(n)}</div>" for n in state["notes"])
    return (f'<div class="metrics">{body}</div>'
            f'<div class="muted">{html.escape(spec.display_name)} · processed in {state["seconds"]:.1f} s</div>{notes}')


def render_views(state, keep_types, doc, only_hits):
    if not state:
        return "", pd.DataFrame(), pd.DataFrame(), ""
    spec = MODELS[state["model_key"]]
    colours = type_colours(spec)
    keep = set(keep_types or [])
    results = state["results"]
    doc = doc if doc in results else list(results)[0]

    # Annotated text
    shown = [r for r in results[doc] if not only_hits or any(e.type in keep for e in r.entities)]
    annotated = "".join(sentence_html(r, colours, keep) for r in shown[:SHOW_LIMIT])
    if not shown:
        annotated = "<p class='muted'>No entities of the selected types were found.</p>"
    elif len(shown) > SHOW_LIMIT:
        annotated += f"<p class='muted'>Showing the first {SHOW_LIMIT} of {len(shown)} sentences. The downloads contain all of them.</p>"

    # Entity table
    ents = entities_frame(results)
    ents = ents[ents["type"].isin(keep)]
    table = pd.DataFrame({
        "Source": ents["source"],
        "Sent.": ents["sentence_id"],
        "Entity": ents["entity"],
        "Type": ents["type"].map(pretty),
        "p(entity)": ents["p_entity"].round(3),
        "p(type)": ents["p_type"].round(3),
        "Sentence": ents["sentence"],
    })
    if len(results) == 1:
        table = table.drop(columns="Source")

    # Per-entity summary and bar chart
    if ents.empty:
        summary, bars = pd.DataFrame(columns=["Entity", "Type", "Mentions", "Sentences", "Mean p(entity)"]), ""
    else:
        summary = (
            ents.assign(key=ents["entity"].str.lower())
            .groupby(["key", "type"], as_index=False)
            .agg(Entity=("entity", "first"), Mentions=("entity", "size"),
                 Sentences=("sentence_id", "nunique"), mean=("p_entity", "mean"))
            .sort_values(["Mentions", "Entity"], ascending=[False, True])
        )
        summary = pd.DataFrame({
            "Entity": summary["Entity"], "Type": summary["type"].map(pretty), "Mentions": summary["Mentions"],
            "Sentences": summary["Sentences"], "Mean p(entity)": summary["mean"].round(3),
        })
        counts = ents["type"].value_counts()
        top = counts.max()
        bars = "<div class='bars'>" + "".join(
            f"<div class='row'><span>{html.escape(pretty(t))}</span>"
            f"<div class='bar' style='width:{100 * n / top:.1f}%;background:{colours[t][1]}'></div>"
            f"<span class='n'>{n}</span></div>"
            for t, n in counts.items()
        ) + "</div>"
    return annotated, table, summary, bars


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------

spec0 = MODELS[DEFAULT_MODEL]
links = [f"[Source code]({GITHUB_URL})"] + ([f"[Paper]({PAPER_URL})"] if PAPER_URL else [])

with gr.Blocks(title="DINNNER | Biomedical NER") as demo:
    state = gr.State(None)

    gr.Markdown(
        "# 🧬 DINNNER: biomedical entity recognition\n"
        "Paste biomedical text or upload research articles (PDF). DINNNER tags **genes and gene products, "
        "simple chemicals, complexes and cellular components**. Results appear below and can be downloaded as CSV."
    )

    with gr.Sidebar(open=True, width=310, position="left"):
        gr.Markdown("## DINNNER\n<span class='muted'>Distribution-Informed Neural Network for Named Entity Recognition</span>")
        if len(MODELS) > 1:
            model_key = gr.Dropdown(
                choices=[(m.display_name, k) for k, m in MODELS.items()], value=DEFAULT_MODEL, label="Model",
            )
        else:
            model_key = gr.State(DEFAULT_MODEL)
            gr.Markdown(f"**Model**  \n{spec0.display_name}")
        gr.Markdown("**Entity types**")
        gr.HTML(legend_html(spec0))
        gr.Markdown(
            "---\n"
            "DINNNER splits token classification into two linked questions: *is this word part of an "
            "entity?* and *which entity type is it?* A shared PubMedBERT encoder feeds a binary head, "
            "p(entity present), and a multiclass head, p(label). The final label combines both, "
            "p(entity present) × p(label). Both probabilities are reported for every entity.\n\n"
            + " · ".join(links) + f"\n\n<span class='muted'>{spec0.description}</span>"
        )

    with gr.Tabs():
        with gr.Tab("✍️ Paste text"):
            text_in = gr.Textbox(
                lines=8, max_lines=20, max_length=MAX_TEXT_CHARS, show_label=False,
                placeholder="Paste an abstract, a paragraph or a full article here...",
            )
            gr.Examples(examples=[[t] for t in EXAMPLES.values()], inputs=[text_in],
                        example_labels=list(EXAMPLES), label="Try an example")
            run_text = gr.Button("Recognise entities", variant="primary")
        with gr.Tab("📄 Upload PDFs"):
            pdf_in = gr.File(
                file_count="multiple", file_types=[".pdf"], type="filepath",
                label=f"Up to {MAX_PDF_FILES} PDFs (first {MAX_PDF_PAGES} pages of each are processed)",
            )
            body_only = gr.Checkbox(
                value=True,
                label="Process the article body only (from the Abstract or Introduction up to the References)",
            )
            run_pdf = gr.Button("Recognise entities", variant="primary")

    with gr.Column(visible=False) as results_col:
        gr.Markdown("## Results")
        summary_out = gr.HTML()
        type_filter = gr.CheckboxGroup(label="Entity types to show", choices=[])
        with gr.Tabs():
            with gr.Tab("Annotated text"):
                with gr.Row():
                    doc_pick = gr.Dropdown(label="Document", choices=[], visible=False, scale=2)
                    only_hits = gr.Checkbox(label="Only show sentences that contain entities", scale=3)
                annotated_out = gr.HTML()
            with gr.Tab("Entity table"):
                table_out = gr.Dataframe(wrap=True, max_height=520, show_search="search", interactive=False)
            with gr.Tab("Entity summary"):
                bars_out = gr.HTML()
                entity_summary_out = gr.Dataframe(max_height=520, interactive=False)
            with gr.Tab("⬇️ Download"):
                gr.Markdown(
                    "Full results (all entity types, all sentences).\n\n"
                    "* **Entities (CSV):** one row per entity with type, character offsets, both probabilities and the sentence.\n"
                    "* **Annotated sentences (CSV):** one row per sentence, entities marked inline as `<Type>text</Type>`.\n"
                    "* **Token labels (TSV):** CoNLL-style word and IOB label per line."
                )
                with gr.Row():
                    dl_ents = gr.DownloadButton("Entities (CSV)", visible=False)
                    dl_sents = gr.DownloadButton("Annotated sentences (CSV)", visible=False)
                    dl_iob = gr.DownloadButton("Token labels (TSV)", visible=False)

    view_outputs = [annotated_out, table_out, entity_summary_out, bars_out]
    run_outputs = [state, results_col, summary_out, type_filter, doc_pick, only_hits,
                   *view_outputs, dl_ents, dl_sents, dl_iob]

    run_text.click(analyse_text, [text_in, model_key], run_outputs, api_name="tag_text")
    run_pdf.click(analyse_pdfs, [pdf_in, body_only, model_key], run_outputs, api_name="tag_pdfs")
    for comp in (type_filter, doc_pick, only_hits):
        comp.input(render_views, [state, type_filter, doc_pick, only_hits], view_outputs, api_visibility="private")


if __name__ == "__main__":
    _prepare_nltk()
    try:
        get_model(DEFAULT_MODEL)  # warm up so the first visitor does not wait
        print(f"DINNNER loaded on {DEVICE}")
    except gr.Error:
        print(f"WARNING: model not loaded at start-up: {_LOAD_ERRORS.get(DEFAULT_MODEL)}")
    demo.queue(default_concurrency_limit=2).launch(
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="teal",
                            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]),
        css=CSS,
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
    )
