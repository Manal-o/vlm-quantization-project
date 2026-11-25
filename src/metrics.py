"""Métriques : ressources (VRAM, latence) et corrélation avec les humains."""
import re
import torch
from scipy.stats import kendalltau, spearmanr


def peak_vram_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**3


def parse_score(text: str, lo: int = 1, hi: int = 4):
    """Extrait le premier chiffre valide ; None si non parsable."""
    m = re.search(rf"[{lo}-{hi}]", text)
    return int(m.group()) if m else None


def correlations(vlm_scores, human_scores):
    """Kendall tau et Spearman rho, en ignorant les réponses non parsables."""
    pairs = [(v, h) for v, h in zip(vlm_scores, human_scores) if v is not None]
    if len(pairs) < 3:
        return {"kendall_tau": None, "spearman_rho": None, "n": len(pairs)}
    v, h = zip(*pairs)
    return {
        "kendall_tau": kendalltau(v, h).statistic,
        "spearman_rho": spearmanr(v, h).statistic,
        "n": len(pairs),
        "unparsable_rate": 1 - len(pairs) / len(vlm_scores),
    }
