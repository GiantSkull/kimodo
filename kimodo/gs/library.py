# Copyright Giant Skull: daniel@giantskull.com
"""Tool settings and the studio character library.

The library is a folder of skinned FBX files exported from the game project (export them from the engine, not from
source art: the engine asset's proportions are what animations play on). Each FBX may have a sidecar JSON with the
same name:

    {
      "display_name": "Barbarian",
      "skeleton_hips_height": 89.513,      # hips height of the engine Skeleton's reference pose, in file units (cm)
      "bones": {"spine_05": "Chest", "prop_r": null}   # bone map overrides (null excludes a bone)
    }
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

SETTINGS_DEFAULTS = {
    "library_dir": os.environ.get("GS_KIMODO_LIBRARY", ""),
    "export_dir": str(Path.home() / "Documents" / "GSKimodo" / "Exports"),
    "last_character": "",
}


def settings_path() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "GSKimodo"
    root.mkdir(parents=True, exist_ok=True)
    return root / "settings.json"


def load_settings() -> Dict[str, str]:
    data = dict(SETTINGS_DEFAULTS)
    try:
        with open(settings_path(), "r", encoding="utf-8") as f:
            data.update({k: v for k, v in json.load(f).items() if isinstance(v, str)})
    except (OSError, ValueError):
        pass
    if os.environ.get("GS_KIMODO_LIBRARY"):
        data["library_dir"] = os.environ["GS_KIMODO_LIBRARY"]
    return data


def save_settings(**changes: str) -> None:
    data = load_settings()
    data.update(changes)
    with open(settings_path(), "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)


@dataclass
class LibraryEntry:
    display_name: str
    fbx_path: str
    skeleton_hips_height: Optional[float] = None
    bones: Optional[Dict[str, Optional[str]]] = None


def read_sidecar(fbx_path: str) -> dict:
    path = Path(fbx_path).with_suffix(".json")
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def scan_library(library_dir: str) -> List[LibraryEntry]:
    if not library_dir or not os.path.isdir(library_dir):
        return []
    entries = []
    for fbx in sorted(Path(library_dir).rglob("*.fbx"), key=lambda p: p.stem.lower()):
        meta = read_sidecar(str(fbx))
        entries.append(
            LibraryEntry(
                display_name=str(meta.get("display_name") or fbx.stem),
                fbx_path=str(fbx),
                skeleton_hips_height=meta.get("skeleton_hips_height"),
                bones=meta.get("bones"),
            )
        )
    # Unique display names for the dropdown.
    seen: Dict[str, int] = {}
    for e in entries:
        n = seen.get(e.display_name, 0)
        seen[e.display_name] = n + 1
        if n:
            e.display_name = f"{e.display_name} ({n + 1})"
    return entries
