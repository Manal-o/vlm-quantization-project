"""Télécharge un sous-ensemble de Flickr8k dans data/raw/ (non versionné).

Source : jeu Flickr8k sur Hugging Face. VERIFIEZ l'identifiant, le nom des
colonnes et la licence avant usage ; adaptez ce script si besoin.
Pour l'option B, les notes humaines (Flickr8k-Expert) sont à récupérer à part.
"""
import argparse
import json
from pathlib import Path
from datasets import load_dataset

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=500)
p.add_argument("--out", default="data/raw")
p.add_argument("--hf_id", default="jxie/flickr8k")
args = p.parse_args()

out = Path(args.out)
(out / "images").mkdir(parents=True, exist_ok=True)
ds = load_dataset(args.hf_id, split="test").select(range(args.n))

meta = []
for i, ex in enumerate(ds):
    path = out / "images" / f"{i:05d}.jpg"
    ex["image"].convert("RGB").save(path)
    caps = [ex[k] for k in ex if k.startswith("caption")]
    meta.append({"id": i, "file": str(path), "references": caps})

(out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
print(f"{len(meta)} images enregistrées dans {out}")
