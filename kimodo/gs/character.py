# Copyright Giant Skull: daniel@giantskull.com
"""Studio characters: skinned FBX files loaded (through ``fbx_bpy`` in a child process) into numpy arrays."""

import hashlib
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import bonemap


def cache_dir() -> Path:
    root = os.environ.get("GS_KIMODO_CACHE") or os.path.join(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()), "GSKimodo", "cache")
    path = Path(root) / "characters"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_bpy(*args: str, timeout: float = 600.0) -> None:
    """Runs ``python -m kimodo.gs.fbx_bpy <args>`` with this interpreter; raises with Blender's output on failure."""
    proc = subprocess.run(
        [sys.executable, "-m", "kimodo.gs.fbx_bpy", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
        raise RuntimeError(f"FBX processing failed ({' '.join(args[:2])}):\n{tail}")


@dataclass
class CharacterRig:
    name: str
    fbx_path: str
    armature_name: str
    bone_names: List[str]
    bone_parents: np.ndarray  # (B,)
    rest_arm: np.ndarray  # (B, 4, 4) armature space
    armature_world: np.ndarray  # (4, 4) armature -> Blender world (Z-up, metres)
    verts: np.ndarray  # (V, 3) armature space
    faces: np.ndarray  # (F, 3)
    skin_weights: np.ndarray  # (V, 4)
    skin_indices: np.ndarray  # (V, 4)
    bone_map: Dict[str, str] = field(default_factory=dict)  # target bone -> SOMA joint

    @property
    def missing_required(self) -> List[str]:
        return bonemap.missing_required(self.bone_map)


def _file_key(path: Path) -> str:
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|v1"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def load_character(fbx_path: str, name: Optional[str] = None, preset: Optional[Dict[str, Optional[str]]] = None) -> CharacterRig:
    """Loads a skinned FBX (cached by path, size and modification time) and resolves its bone map, with ``preset``
    ({bone: SOMA joint or None}) overriding the guess."""
    path = Path(fbx_path)
    if not path.is_file():
        raise FileNotFoundError(f"Character FBX not found: {fbx_path}")
    cached = cache_dir() / f"{path.stem}_{_file_key(path)}.npz"
    if not cached.exists():
        tmp = cached.with_suffix(".tmp.npz")
        run_bpy("read", str(path), str(tmp))
        os.replace(tmp, cached)
    data = np.load(cached, allow_pickle=False)
    names = [str(n) for n in data["bone_names"]]
    parents = data["bone_parents"].astype(np.int64)

    return CharacterRig(
        name=name or path.stem,
        fbx_path=str(path),
        armature_name=str(data["armature_name"]),
        bone_names=names,
        bone_parents=parents,
        rest_arm=data["rest_arm"],
        armature_world=data["armature_world"],
        verts=data["verts"],
        faces=data["faces"],
        skin_weights=data["skin_weights"],
        skin_indices=data["skin_indices"].astype(np.int64),
        bone_map=bonemap.resolve_bone_map(names, list(parents), preset),
    )


def write_fbx(rig: CharacterRig, bone_arm: np.ndarray, root_arm: np.ndarray, fps: float, out_path: str, take_name: str) -> None:
    """Writes an animation-only FBX on ``rig``'s skeleton from armature-space bone matrices (T, B, 4, 4) and the
    armature object's root motion (T, 3) (see ``Retargeter.to_armature``)."""
    with tempfile.TemporaryDirectory(prefix="gskimodo_") as tmp:
        anim = os.path.join(tmp, "anim.npz")
        np.savez(
            anim,
            bone_names=np.array(rig.bone_names),
            bone_arm=bone_arm,
            root_arm=root_arm,
            fps=np.float64(fps),
            take_name=np.array(take_name),
        )
        run_bpy("write", rig.fbx_path, anim, out_path)

