"""One-off: publish a trained DINNNER checkpoint to the Hugging Face Hub.

The checkpoint is checked against the DINNNER architecture first, then
uploaded under the DINNNER file name with a short model card. Run it on the
machine that has the .pth file, after `hf auth login` (or with an
HF_TOKEN environment variable that has write access):

    python scripts/upload_weights.py \
        --checkpoint "dd_ner_model/sinnbioner_pubmedbert_bionlp13pc_epoch_19_f1_0.9057.pth" \
        --repo faridan/dinnner-pubmedbert-bionlp13pc
"""

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

from dinnner.config import MODELS  # noqa: E402
from dinnner.model import load_model  # noqa: E402

CARD = """---
license: {license}
language: en
library_name: pytorch
pipeline_tag: token-classification
tags: [biomedical, named-entity-recognition, ner, bionlp13pc, dinnner]
datasets: [bionlp13pc]
base_model: {encoder}
---

# DINNNER + PubMedBERT (BioNLP13PC)

Weights for **DINNNER** (Distribution-Informed Neural Network for Named Entity
Recognition), trained on the BioNLP 2013 Pathway Curation corpus.

DINNNER adds two token-level heads to a shared encoder: a binary head for
p(entity present) and a multiclass head for p(label). The predicted label is
the argmax of p(entity present) x p(label).

* Encoder: `{encoder}`
* Labels: {labels}
* Max subword length: {max_length}
* Validation micro F1: {f1}

Try it online and see the code: {github}

The file is a plain PyTorch `state_dict`; load it with the `DINNNER` class in
the demo repository (`dinnner/model.py`).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="local .pth file")
    ap.add_argument("--repo", required=True, help="e.g. your-username/dinnner-pubmedbert-bionlp13pc")
    ap.add_argument("--model", default="pubmedbert-bionlp13pc", choices=list(MODELS))
    ap.add_argument("--f1", default="0.9057", help="shown on the model card")
    ap.add_argument("--github", default="https://github.com/faridan/dinnner-demo")
    ap.add_argument("--license", default="other", help="licence id for the card, e.g. mit or cc-by-4.0")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    spec = MODELS[args.model]
    print("Checking the checkpoint against the DINNNER architecture...")
    load_model(spec.__class__(**{**spec.__dict__, "local_path": args.checkpoint}), torch.device("cpu"))
    print("OK")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    print(f"Uploading {args.checkpoint} as {spec.weights_file} ...")
    api.upload_file(path_or_fileobj=args.checkpoint, path_in_repo=spec.weights_file, repo_id=args.repo,
                    commit_message="Add DINNNER checkpoint")

    card = CARD.format(license=args.license, encoder=spec.encoder, max_length=spec.max_length, f1=args.f1,
                       github=args.github, labels=", ".join(spec.id2label.values()))
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(card)
    api.upload_file(path_or_fileobj=f.name, path_in_repo="README.md", repo_id=args.repo,
                    commit_message="Add model card")
    print(f"Done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
