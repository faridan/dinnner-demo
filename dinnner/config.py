"""Model registry and runtime settings for the DINNNER demo.

Each entry describes one trained DINNNER checkpoint. To add a checkpoint
trained on another dataset (e.g. BC5CDR or JNLPBA), add an entry with its
label map, upload the .pth file to the Hugging Face Hub, and it will appear
in the model selector automatically.
"""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    key: str                      # short id used internally
    display_name: str             # shown in the UI
    dataset: str                  # training corpus
    encoder: str                  # Hugging Face id of the pretrained encoder
    id2label: dict                # label index -> IOB label (must match training)
    weights_repo: str             # Hugging Face Hub repo holding the .pth file
    weights_file: str             # file name inside that repo
    local_path: str               # used instead of the Hub if the file exists
    max_length: int = 192         # max subword length used during training
    description: str = ""
    entity_info: dict = field(default_factory=dict)  # entity type -> description


# Set DINNNER_WEIGHTS_REPO as a Space variable (or edit the default below)
# to point at the Hugging Face model repo that holds your checkpoint.
_DEFAULT_REPO = os.environ.get("DINNNER_WEIGHTS_REPO", "faridan/dinnner-pubmedbert-bionlp13pc")

MODELS = {
    "pubmedbert-bionlp13pc": ModelSpec(
        key="pubmedbert-bionlp13pc",
        display_name="DINNNER + PubMedBERT (BioNLP13PC)",
        dataset="BioNLP13PC",
        encoder=os.environ.get("DINNNER_ENCODER", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"),
        id2label={
            0: "O",
            1: "B-Gene_or_gene_product",
            2: "I-Gene_or_gene_product",
            3: "B-Simple_chemical",
            4: "I-Simple_chemical",
            5: "B-Complex",
            6: "I-Complex",
            7: "B-Cellular_component",
            8: "I-Cellular_component",
        },
        weights_repo=_DEFAULT_REPO,
        weights_file="dinnner_pubmedbert_bionlp13pc.pth",
        local_path=os.environ.get("DINNNER_WEIGHTS_PATH", "weights/dinnner_pubmedbert_bionlp13pc.pth"),
        description="Trained on the BioNLP 2013 Pathway Curation corpus.",
        entity_info={
            "Gene_or_gene_product": "Genes, proteins and their products",
            "Simple_chemical": "Small molecules and chemical compounds",
            "Complex": "Molecular complexes formed by several components",
            "Cellular_component": "Cell structures and compartments",
        },
    ),
}

DEFAULT_MODEL = "pubmedbert-bionlp13pc"

# Limits that keep the public CPU Space responsive. Override with env vars.
MAX_TEXT_CHARS = int(os.environ.get("DINNNER_MAX_TEXT_CHARS", 50_000))
MAX_PDF_PAGES = int(os.environ.get("DINNNER_MAX_PDF_PAGES", 40))
MAX_PDF_FILES = int(os.environ.get("DINNNER_MAX_PDF_FILES", 5))
BATCH_SIZE = int(os.environ.get("DINNNER_BATCH_SIZE", 16))
