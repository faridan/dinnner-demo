---
title: DINNNER Biomedical NER
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Biomedical named entity recognition with DINNNER
---

# DINNNER: online biomedical named entity recognition

[![tests](https://github.com/faridan/dinnner-demo/actions/workflows/tests.yml/badge.svg)](https://github.com/faridan/dinnner-demo/actions/workflows/tests.yml)
[![Open the online tool](https://img.shields.io/badge/🤗%20Online%20tool-DINNNER-teal)](https://huggingface.co/spaces/faridan/dinnner)

**Online tool:** https://huggingface.co/spaces/faridan/dinnner

DINNNER (Distribution-Informed Neural Network for Named Entity Recognition) recognises
biomedical entities in scientific text. This repository contains the web tool that
accompanies the paper. You can paste text or upload research articles as PDF, see the
entities highlighted in the browser, and download the results as CSV.

The released model is trained on BioNLP13PC and tags four entity types:
**Gene or gene product**, **Simple chemical**, **Complex** and **Cellular component**.

## How DINNNER labels a word

A shared transformer encoder (PubMedBERT) feeds two token-level heads:

* a binary head that estimates p(entity present), and
* a multiclass head that estimates p(label).

The predicted label is the argmax of p(entity present) × p(label), so entity typing is
conditioned on the evidence that an entity is there at all. The tool reports both
probabilities for every entity it finds.

## Using the tool

1. Paste text (or click an example), or upload up to 5 PDFs.
2. Click **Recognise entities**.
3. Browse the results in four views: annotated text, entity table, per-entity summary,
   and downloads. Use **Entity types to show** to filter.

For PDFs, the tool keeps the article body (from the Abstract or Introduction up to the
References) by default; untick the option to process everything. Scanned PDFs without a
text layer cannot be read.

### Downloads

| File | One row per | Columns |
|---|---|---|
| `*_entities.csv` | entity | `source`, `sentence_id`, `entity`, `type`, `start_char`, `end_char` (offsets within the sentence), `p_entity`, `p_type`, `sentence` |
| `*_sentences.csv` | sentence | `source`, `sentence_id`, `sentence`, `annotated_sentence` (entities marked as `<Type>text</Type>`), `n_entities` |
| `*_iob.tsv` | word | word and IOB label, blank line between sentences (CoNLL style) |

## Run it on your own machine

```bash
git clone https://github.com/faridan/dinnner-demo.git
cd dinnner-demo
pip install -r requirements.txt
python app.py
```

Then open http://localhost:7860. The weights are downloaded from the Hugging Face Hub on
first use. To use a local file instead, put it at
`weights/dinnner_pubmedbert_bionlp13pc.pth` or set `DINNNER_WEIGHTS_PATH`.

### Use DINNNER from Python

```python
import torch
from dinnner import MODELS, load_model, predict
from dinnner.text import split_sentences

spec = MODELS["pubmedbert-bionlp13pc"]
model, tokenizer = load_model(spec, torch.device("cpu"))
for sent in predict(split_sentences("AKT phosphorylates FOXO1 in the nucleus."),
                    model, tokenizer, spec.id2label, torch.device("cpu")):
    for e in sent.entities:
        print(e.text, e.type, round(e.p_entity, 3))
```

## Repository layout

```
app.py                     Gradio web interface
dinnner/config.py          model registry (labels, encoder, weight location) and limits
dinnner/model.py           DINNNER architecture and checkpoint loading
dinnner/inference.py       tokenisation with offsets, long-sentence chunking, batching, span decoding
dinnner/text.py            PDF extraction, sentence splitting, CSV/IOB export
scripts/upload_weights.py  publish a checkpoint to the Hugging Face Hub
tests/                     offline tests with a tiny stand-in model (pytest)
```

Adding a model trained on another corpus (for example BC5CDR or JNLPBA) only needs a new
entry in `dinnner/config.py` and an upload with `scripts/upload_weights.py`; it then
appears in the model selector.

## Deployment (maintainers)

The code lives on GitHub; the public site runs on a free Hugging Face Space
(Gradio SDK, ZeroGPU hardware), and every push to `main` redeploys it. On ZeroGPU the
app borrows a GPU only while it tags text (`@spaces.GPU` in `app.py`); the same code
also runs unchanged on CPU hardware.

1. **Publish the weights.** `hf auth login`, then
   `python scripts/upload_weights.py --checkpoint path/to/checkpoint.pth --repo faridan/dinnner-pubmedbert-bionlp13pc`
2. **Create the Space** at huggingface.co/new-space: name `dinnner`, SDK **Gradio**,
   template **Blank**, hardware **ZeroGPU**, public.
3. **Configure the Space** (Settings > Variables and secrets), as *variables*:
   `DINNNER_WEIGHTS_REPO` = `faridan/dinnner-pubmedbert-bionlp13pc`,
   `DINNNER_GITHUB_URL` = this repository's URL, and later `DINNNER_PAPER_URL`.
4. **Connect GitHub.** In this repository (Settings > Secrets and variables > Actions),
   add the secret `HF_TOKEN` (a Hugging Face token with write access) and the variable
   `HF_SPACE` = `faridan/dinnner`. Push to `main`; the
   *sync to Hugging Face Space* workflow mirrors the code and the Space rebuilds.

ZeroGPU gives each visitor a daily GPU allowance (a few minutes); one DINNNER request
uses only seconds of it. An idle Space sleeps and wakes on the next visit, which takes
a minute or two.

## Citation

If you use DINNNER, please cite:

```bibtex
@article{dinnner,
  title   = {TODO},
  author  = {TODO},
  journal = {Intelligent Systems with Applications},
  year    = {TODO}
}
```

## Acknowledgements

The interface builds on an earlier NER and relation extraction demo by Md Abul Bashar.
