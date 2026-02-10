import re
from pathlib import Path
CHECKPOINT_ROOT = Path("outputs/checkpoints")


def get_latest_model_dir(model_name: str) -> Path:
    """
    Cherche les dossiers du type ModelName_XX et retourne celui avec le XX le plus grand.
    """
    pattern = re.compile(rf"^{re.escape(model_name)}_(\d+)$")
    candidates = []
    for d in CHECKPOINT_ROOT.iterdir():
        if d.is_dir():
            match = pattern.match(d.name)
            if match:
                candidates.append((int(match.group(1)), d))
    if not candidates:
        raise FileNotFoundError(f"Aucun checkpoint trouvé pour '{model_name}'")

    return max(candidates, key=lambda x: x[0])[1]


def get_checkpoint_files(model_dir: Path):
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml introuvable dans {model_dir}")

    pt_files = list(model_dir.glob("*.pt"))

    if len(pt_files) != 1:
        raise RuntimeError(f"Il doit y avoir exactement un fichier .pt dans {model_dir}")

    return config_path, pt_files[0]