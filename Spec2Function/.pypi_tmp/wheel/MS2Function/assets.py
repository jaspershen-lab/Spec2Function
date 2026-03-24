from __future__ import annotations

from pathlib import Path
import os
from typing import Iterable, Optional

REQUIRED_ASSETS = (
    "models/best_model.pth",
    "models/config.json",
    "data/hmdb_subsections_WITH_NAME.jsonl",
    "data/all_jsonl_embeddings.pt",
)


def _default_cache_dir() -> Path:
    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or str(Path.home())
        return Path(base) / "MS2Function"
    xdg_cache = os.getenv("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "MS2Function"
    return Path.home() / ".cache" / "MS2Function"


def _has_required_files(root: Path, files: Iterable[str] = REQUIRED_ASSETS) -> bool:
    return all((root / rel).exists() for rel in files)


def _download_assets(root: Path, repo_id: str, files: Iterable[str]) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required to download MS2Function assets. "
            "Install with `pip install huggingface_hub` or set MS2FUNCTION_ASSET_DIR."
        ) from exc

    root.mkdir(parents=True, exist_ok=True)
    for rel in files:
        local_path = root / rel
        if local_path.exists():
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=rel,
            local_dir=root,
            local_dir_use_symlinks=False,
        )


def resolve_assets_root(project_root: Optional[Path]) -> Path:
    env_dir = os.getenv("MS2FUNCTION_ASSET_DIR")
    if env_dir:
        root = Path(env_dir)
        if not _has_required_files(root):
            raise FileNotFoundError(
                f"MS2Function assets not found in MS2FUNCTION_ASSET_DIR: {root}"
            )
        return root

    if project_root:
        project_root = Path(project_root)
        if _has_required_files(project_root):
            return project_root

    repo_id = os.getenv("MS2FUNCTION_ASSET_REPO", "cgxjdzz/ms2function-assets")
    cache_root = _default_cache_dir()
    _download_assets(cache_root, repo_id, REQUIRED_ASSETS)
    if not _has_required_files(cache_root):
        raise FileNotFoundError(
            f"Failed to download MS2Function assets from {repo_id} into {cache_root}"
        )
    return cache_root
