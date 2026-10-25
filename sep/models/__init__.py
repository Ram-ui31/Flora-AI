"""Model factory for Flora's production SCNet + FiLM separator."""
from .scnet_film import SCNetFiLM


def count_params(model):
    """Return the parameter count in millions (used by training logs)."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def build_model(a: dict, n_queries=4):
    """Build SCNet from training arguments, including existing SCNet checkpoints."""
    arch = a.get("arch", "scnet")
    if arch != "scnet":
        raise ValueError(f"Unsupported architecture {arch!r}; Flora uses SCNet + FiLM.")
    width = a.get("scnet_width", 128)
    return SCNetFiLM(n_queries=n_queries, dims=(4, width // 4, width // 2, width),
                     num_dplayer=a.get("layers", 6))
