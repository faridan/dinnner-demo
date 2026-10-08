"""Build a tiny stand-in DINNNER checkpoint for offline testing.

It uses a 2-layer BERT with a locally built vocabulary and is overfitted on
gazetteer-labelled sentences, so tests can check that the pipeline recovers
known labels. It says nothing about the real model's accuracy.

    python tests/make_standin_model.py OUT_DIR
"""

import os
import sys

import torch
import torch.nn.functional as F
from transformers import BertConfig, BertTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dinnner.inference import split_words  # noqa: E402

GAZETTEER = {
    ("insulin", "receptor"): "Gene_or_gene_product", ("IRS", "-", "1"): "Gene_or_gene_product",
    ("PI3K",): "Gene_or_gene_product", ("AKT",): "Gene_or_gene_product", ("FOXO1",): "Gene_or_gene_product",
    ("APAF1",): "Gene_or_gene_product", ("caspase", "-", "9"): "Gene_or_gene_product",
    ("BCL", "-", "2"): "Gene_or_gene_product", ("BAX",): "Gene_or_gene_product", ("BAK",): "Gene_or_gene_product",
    ("cytochrome", "c"): "Gene_or_gene_product", ("TNF", "-", "alpha"): "Gene_or_gene_product",
    ("insulin",): "Gene_or_gene_product",
    ("PIP2",): "Simple_chemical", ("PIP3",): "Simple_chemical", ("ATP",): "Simple_chemical",
    ("Ca2",): "Simple_chemical", ("amino", "acids"): "Simple_chemical",
    ("PI3K", "complex"): "Complex", ("mTORC1",): "Complex", ("apoptosome",): "Complex",
    ("plasma", "membrane"): "Cellular_component", ("nucleus",): "Cellular_component",
    ("cytoplasm",): "Cellular_component", ("mitochondria",): "Cellular_component",
    ("outer", "mitochondrial", "membrane"): "Cellular_component",
    ("endoplasmic", "reticulum"): "Cellular_component",
}

LABELS = ["O", "B-Gene_or_gene_product", "I-Gene_or_gene_product", "B-Simple_chemical", "I-Simple_chemical",
          "B-Complex", "I-Complex", "B-Cellular_component", "I-Cellular_component"]
L2I = {l: i for i, l in enumerate(LABELS)}


def gazetteer_labels(words):
    labels, i = ["O"] * len(words), 0
    keys = sorted(GAZETTEER, key=len, reverse=True)
    while i < len(words):
        for k in keys:
            if tuple(words[i:i + len(k)]) == k:
                t = GAZETTEER[k]
                labels[i] = f"B-{t}"
                for j in range(i + 1, i + len(k)):
                    labels[j] = f"I-{t}"
                i += len(k)
                break
        else:
            i += 1
    return labels


def build(out_dir, sentences):
    from dinnner.model import DINNNER

    enc_dir = os.path.join(out_dir, "encoder")
    os.makedirs(enc_dir, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    words = sorted({w.lower() for s in sentences for w in split_words(s)[0]})
    # whole words plus single characters so unseen words still tokenise
    chars = sorted({c for w in words for c in w})
    vocab += words + [c for c in chars if c not in words] + ["##" + c for c in chars]
    with open(os.path.join(enc_dir, "vocab.txt"), "w") as f:
        f.write("\n".join(dict.fromkeys(vocab)))
    tok = BertTokenizerFast(os.path.join(enc_dir, "vocab.txt"), do_lower_case=True)
    tok.save_pretrained(enc_dir)
    BertConfig(vocab_size=len(tok), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
               intermediate_size=128, max_position_embeddings=256).save_pretrained(enc_dir)

    torch.manual_seed(0)
    model = DINNNER(enc_dir, num_classes=len(LABELS))
    data = []
    for s in sentences:
        w, _ = split_words(s)
        data.append((w, [L2I[l] for l in gazetteer_labels(w)]))

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for step in range(300):
        enc = tok([d[0] for d in data], is_split_into_words=True, padding=True, return_tensors="pt")
        y = torch.full(enc["input_ids"].shape, -100)
        for r, (_, labs) in enumerate(data):
            prev = None
            for t, wid in enumerate(enc.word_ids(r)):
                if wid is not None and wid != prev:
                    y[r, t] = labs[wid]
                prev = wid
        lb, lm = model(enc["input_ids"], enc["attention_mask"])
        yb = torch.where(y == -100, -100, (y > 0).long())
        pb = torch.sigmoid(lb)
        loss_b = F.nll_loss(torch.log(torch.cat([1 - pb, pb], -1) + 1e-10).view(-1, 2), yb.view(-1), ignore_index=-100)
        loss_m = F.cross_entropy(lm.view(-1, len(LABELS)), y.view(-1), ignore_index=-100)
        loss = 0.5 * loss_b + 0.5 * loss_m
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"final loss {loss.item():.4f}")

    ckpt = os.path.join(out_dir, "dinnner_standin.pth")
    torch.save(model.state_dict(), ckpt)
    return enc_dir, ckpt


if __name__ == "__main__":
    from dinnner.examples import EXAMPLES
    from dinnner.text import split_sentences

    sents = [s for t in EXAMPLES.values() for s in split_sentences(t)]
    print(build(sys.argv[1], sents))
