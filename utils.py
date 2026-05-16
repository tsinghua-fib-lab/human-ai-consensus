from pathlib import Path


def find_hf_model_dir(cache_dir):
    """Return a Hugging Face model directory from a cache root or model path."""
    root = Path(cache_dir)
    if (root / "config.json").exists():
        return str(root)

    snapshots = list(root.glob("models--*/*/*"))
    for candidate in snapshots:
        if (candidate / "config.json").exists():
            return str(candidate)

    for candidate in root.rglob("config.json"):
        return str(candidate.parent)

    raise FileNotFoundError(
        f"No Hugging Face model directory with config.json found under {cache_dir}"
    )
