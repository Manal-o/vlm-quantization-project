"""Génère les légendes pour 2 modèles x 3 précisions sur les MÊMES images.
Reprend automatiquement : une configuration déjà terminée est ignorée.
"""
import json
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from src.models import load, free, generate
from src.metrics import peak_vram_gb


def run(cfg_path="configs/default.yaml", n_images=None, out_dir="results/tables"):
    cfg = yaml.safe_load(open(cfg_path))
    meta = json.load(open("data/raw/metadata.json"))
    meta = meta[: (n_images or cfg["data"]["n_images"])]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompt = cfg["prompts"]["caption"]
    max_new = cfg["generation"]["max_new_tokens"]

    for m in cfg["models"]:
        for prec in cfg["precisions"]:
            tag = f"{m['name']}_{prec}"
            f_cap, f_res = out / f"captions_{tag}.csv", out / f"resources_{tag}.json"
            if f_cap.exists() and f_res.exists():
                print(f"[skip] {tag} déjà fait")
                continue
            print(f"=== {tag} ===")
            torch.manual_seed(cfg["seed"])
            proc, model = load(m["hf_id"], prec)
            size_gb = model.get_memory_footprint() / 1024**3
            # échauffement (exclu des mesures)
            generate(proc, model, Image.open(meta[0]["file"]).convert("RGB"), prompt, max_new)
            torch.cuda.reset_peak_memory_stats()

            rows = []
            for ex in tqdm(meta):
                img = Image.open(ex["file"]).convert("RGB")
                text, lat = generate(proc, model, img, prompt, max_new)
                rows.append({"id": ex["id"], "caption": text, "latency_s": lat})

            pd.DataFrame(rows).to_csv(f_cap, index=False)
            lat_mean = sum(r["latency_s"] for r in rows) / len(rows)
            json.dump({"model": m["name"], "precision": prec, "n_images": len(rows),
                       "peak_vram_gb": peak_vram_gb(), "model_size_gb": size_gb,
                       "latency_mean_s": lat_mean}, open(f_res, "w"), indent=2)
            free(model)
            del proc
