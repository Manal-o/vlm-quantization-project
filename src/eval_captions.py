"""Métriques de légendes : BLEU-4, CIDEr, ROUGE-L (+ METEOR si Java dispo)."""
import re
import pandas as pd
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge


def tok(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", str(s).lower()))


def evaluate(captions_csv, meta):
    """Retourne (dict de métriques, DataFrame CIDEr par image)."""
    df = pd.read_csv(captions_csv)
    refs = {m["id"]: m["references"] for m in meta}
    ids = [int(i) for i in df["id"]]
    gts = {i: [tok(r) for r in refs[i]] for i in ids}
    res = {i: [tok(c)] for i, c in zip(ids, df["caption"])}

    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    cider, cider_per_img = Cider().compute_score(gts, res)
    rouge, _ = Rouge().compute_score(gts, res)
    metrics = {"BLEU-4": bleu[3], "CIDEr": cider, "ROUGE-L": rouge}
    try:  # METEOR nécessite Java
        from pycocoevalcap.meteor.meteor import Meteor
        metrics["METEOR"], _ = Meteor().compute_score(gts, res)
    except Exception as e:
        print("METEOR ignoré (Java manquant ?) :", type(e).__name__)
    per_img = pd.DataFrame({"id": ids, "caption": df["caption"], "cider": cider_per_img})
    return metrics, per_img
