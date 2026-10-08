"""DINNNER architecture and checkpoint loading.

DINNNER shares one transformer encoder between two token-level heads:
  * a binary head estimating p(entity present), and
  * a multiclass head estimating p(label).
At inference the two are combined as p(entity present) * p(label), matching
the joint decision rule used in training (see ddner.py).
"""

import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .config import ModelSpec

# Buffers that some transformers versions save and others do not.
_IGNORABLE_KEYS = ("position_ids",)


class DINNNER(nn.Module):
    def __init__(self, encoder_name: str, num_classes: int):
        super().__init__()
        # Build the encoder from its config only: every weight is restored
        # from the DINNNER checkpoint, so downloading the pretrained encoder
        # weights as well would double start-up time for nothing.
        # The attribute is called `bert` so checkpoint keys match training.
        self.bert = AutoModel.from_config(AutoConfig.from_pretrained(encoder_name))
        hidden = self.bert.config.hidden_size
        self.binary_classifier = nn.Linear(hidden, 1)
        self.multiclass_classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.binary_classifier(h), self.multiclass_classifier(h)


def resolve_weights(spec: ModelSpec) -> str:
    """Return a local path to the checkpoint, downloading it if needed."""
    if os.path.exists(spec.local_path):
        return spec.local_path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=spec.weights_repo,
        filename=spec.weights_file,
        token=os.environ.get("HF_TOKEN"),  # only needed if the repo is private
    )


def load_model(spec: ModelSpec, device: torch.device):
    """Build DINNNER, restore the checkpoint and return (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(spec.encoder)
    model = DINNNER(spec.encoder, num_classes=len(spec.id2label))

    state = torch.load(resolve_weights(spec), map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith(_IGNORABLE_KEYS)]
    unexpected = [k for k in unexpected if not k.endswith(_IGNORABLE_KEYS)]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the DINNNER architecture.\n"
            f"Missing keys: {missing[:10]}\nUnexpected keys: {unexpected[:10]}"
        )

    model.to(device).eval()
    return model, tokenizer
