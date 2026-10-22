# Copyright Giant Skull: daniel@giantskull.com
"""The "Giant Skull" folder of the Kimodo demo: pick a studio character, see motions on it, export to the game.

Kept out of ``kimodo/demo/ui.py`` so the fork stays easy to merge with upstream; ui.py and app.py only call
``create_gui``, ``register_project_io``, ``character_skin`` and ``forget_client``.
"""

import json
import os
import re
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import torch

import viser
from kimodo.exports.bvh import save_motion_bvh
from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77

from . import character as gs_character
from . import library
from .retarget import CharacterSkin, Retargeter, soma77_globals

SOMA_CHOICE = "Kimodo default (SOMA)"
PROJECT_FILE = "gs_character.json"


@dataclass
class _ClientState:
    rig: Optional[gs_character.CharacterRig] = None
    retargeter: Optional[Retargeter] = None
    skeleton_hips_height: Optional[float] = None
    entries: Dict[str, library.LibraryEntry] = field(default_factory=dict)
    save_project: Optional[Callable] = None
    load_project: Optional[Callable] = None
    busy: bool = False


_STATES: Dict[int, _ClientState] = {}
_RIG_CACHE: Dict[str, gs_character.CharacterRig] = {}
_RIG_LOCK = threading.Lock()


def forget_client(client_id: int) -> None:
    _STATES.pop(client_id, None)


def register_project_io(client_id: int, save: Callable, load: Callable) -> None:
    state = _STATES.setdefault(client_id, _ClientState())
    state.save_project = save
    state.load_project = load


def character_skin(client_id: int, skeleton) -> Optional[CharacterSkin]:
    """The selected studio character's skin for the viewer, or None to show Kimodo's own body."""
    state = _STATES.get(client_id)
    if state is None or state.retargeter is None or not isinstance(skeleton, (SOMASkeleton30, SOMASkeleton77)):
        return None
    return CharacterSkin(state.retargeter, skeleton)


def _load_rig(fbx_path: str, name: str, bones) -> gs_character.CharacterRig:
    key = f"{os.path.abspath(fbx_path)}|{json.dumps(bones, sort_keys=True)}"
    with _RIG_LOCK:
        rig = _RIG_CACHE.get(key)
        if rig is None:
            rig = gs_character.load_character(fbx_path, name=name, preset=bones)
            _RIG_CACHE[key] = rig
    return rig


def _safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", (name or "").strip()).strip("_")
    return name or "A_Kimodo"


def _notify(client, title, body, color=None, seconds=6.0):
    client.add_notification(title=title, body=body, auto_close_seconds=seconds, color=color, with_close_button=True)


def create_gui(demo, client: viser.ClientHandle) -> None:
    client_id = client.client_id
    state = _STATES.setdefault(client_id, _ClientState())
    settings = library.load_settings()

    with client.gui.add_folder("Giant Skull", expand_by_default=True):
        with client.gui.add_folder("Character", expand_by_default=True):
            gui_character = client.gui.add_dropdown(
                "Character",
                options=[SOMA_CHOICE],
                initial_value=SOMA_CHOICE,
                hint="Studio character the animation is made for. Motions are shown on it and exported to its skeleton.",
            )
            gui_upload = client.gui.add_upload_button(
                "Load FBX...",
                icon=viser.Icon.UPLOAD,
                mime_type=".fbx",
                hint="Use any skinned FBX character that is not in the studio library.",
            )
            gui_status = client.gui.add_markdown("")
            with client.gui.add_folder("Library settings", expand_by_default=False):
                gui_library_dir = client.gui.add_text(
                    "Library folder",
                    initial_value=settings["library_dir"],
                    hint="Folder of character FBX files exported from the game project.",
                )
                gui_refresh = client.gui.add_button("Refresh Library", icon=viser.Icon.REFRESH)

        with client.gui.add_folder("Export to Game", expand_by_default=True):
            gui_name = client.gui.add_text(
                "Animation name",
                initial_value="A_Placeholder",
                hint="File name of the exported animation (no extension).",
            )
            gui_root_motion = client.gui.add_checkbox(
                "Root motion",
                initial_value=True,
                hint="Move the root bone with the character. Turn off for in-place loops.",
            )
            gui_bvh = client.gui.add_checkbox("Also save BVH (Kimodo skeleton)", initial_value=False)
            gui_project = client.gui.add_checkbox(
                "Also save project",
                initial_value=True,
                hint="Saves prompts, constraints and settings so the animation can be reopened and changed.",
            )
            gui_export_dir = client.gui.add_text("Export folder", initial_value=settings["export_dir"])
            gui_export = client.gui.add_button("Export Animation", color="green", icon=viser.Icon.DOWNLOAD)
            gui_open_folder = client.gui.add_button("Open Export Folder", icon=viser.Icon.FOLDER)
            gui_project_path = client.gui.add_text(
                "Project to open",
                initial_value="",
                hint="A project folder saved by Export Animation (ends in _project).",
            )
            gui_open_project = client.gui.add_button("Open Project", icon=viser.Icon.FOLDER_OPEN)

    def _session():
        if not demo.client_active(client_id):
            return None
        return demo.client_sessions[client_id]

    def _set_status():
        rig = state.rig
        if rig is None:
            gui_status.content = "Showing Kimodo's own body. Pick a studio character to see and export motions on it."
            return
        missing = rig.missing_required
        mapped = len(rig.bone_map)
        text = f"**{rig.name}**: {len(rig.bone_names)} bones, {mapped} driven by Kimodo."
        if missing:
            text += f"\n\nNot found: {', '.join(missing)}. Add them to the character's .json sidecar (see README)."
        gui_status.content = text

    def _refresh_library():
        current = str(gui_character.value)
        kept = state.entries.get(current)
        state.entries = {e.display_name: e for e in library.scan_library(gui_library_dir.value)}
        if kept is not None and current not in state.entries:
            state.entries[current] = kept  # an uploaded character stays selectable
        gui_character.options = [SOMA_CHOICE] + list(state.entries)

    def _rebuild_motions():
        """Re-creates the shown motions so they pick up the new character skin."""
        session = _session()
        if session is None:
            return
        old = [(m.joints_pos, m.joints_rot, m.foot_contacts) for m in session.motions.values()]
        demo.clear_motions(client_id)
        if not old:
            demo.add_character_motion(client, session.skeleton)
        for joints_pos, joints_rot, foot_contacts in old:
            demo.add_character_motion(client, session.skeleton, joints_pos, joints_rot, foot_contacts)
        demo.set_frame(client_id, session.frame_idx)

    def _apply_character(display_name: str, fbx_path: Optional[str], bones, skeleton_hips_height):
        current = state.rig.fbx_path if state.rig is not None else None
        if fbx_path == current and (fbx_path is None or state.rig.name == display_name):
            return
        if state.busy:
            _notify(client, "Please wait", "A character is still loading.")
            return
        state.busy = True
        loading = client.add_notification(
            title="Loading character...",
            body=f"{display_name}: the first load of a character can take up to a minute.",
            loading=True,
            with_close_button=False,
        )
        try:
            if fbx_path is None:
                state.rig = None
                state.retargeter = None
                state.skeleton_hips_height = None
            else:
                rig = _load_rig(fbx_path, display_name, bones)
                state.retargeter = Retargeter(rig, device=demo.device)
                state.rig = rig
                state.skeleton_hips_height = skeleton_hips_height
                if re.fullmatch(r"A_(\w+_)?Placeholder", gui_name.value):
                    gui_name.value = f"A_{_safe_name(display_name)}_Placeholder"
            _rebuild_motions()
            library.save_settings(last_character=display_name if fbx_path else "")
        except Exception as e:  # noqa: BLE001 - shown to the user
            traceback.print_exc()
            _notify(client, "Could not load character", str(e), color="red", seconds=15.0)
            state.rig = None
            state.retargeter = None
            state.busy = False
            gui_character.value = SOMA_CHOICE
        finally:
            state.busy = False
            loading.remove()
            _set_status()

    # Selecting a dropdown value (by the user or by code) is the one way a character gets applied.
    @gui_character.on_update
    def _(_event) -> None:
        name = str(gui_character.value)
        if name == SOMA_CHOICE:
            _apply_character(name, None, None, None)
            return
        entry = state.entries.get(name)
        if entry is not None:
            _apply_character(name, entry.fbx_path, entry.bones, entry.skeleton_hips_height)

    def _select(entry: library.LibraryEntry) -> None:
        """Selects a character that may not be in the library (uploaded FBX, project from elsewhere)."""
        state.entries.setdefault(entry.display_name, entry)
        if entry.display_name not in gui_character.options:
            gui_character.options = list(gui_character.options) + [entry.display_name]
        gui_character.value = entry.display_name

    @gui_upload.on_upload
    def _(event) -> None:
        upload = event.target.value
        uploads = gs_character.cache_dir().parent / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        path = uploads / os.path.basename(upload.name)
        path.write_bytes(upload.content)
        _select(library.LibraryEntry(display_name=path.stem, fbx_path=str(path)))

    @gui_refresh.on_click
    def _(_event) -> None:
        library.save_settings(library_dir=gui_library_dir.value)
        _refresh_library()
        _notify(client, "Library refreshed", f"{len(state.entries)} characters found.")

    def _export_motion(session) -> Optional[object]:
        motions = list(session.motions.values())
        if len(motions) != 1:
            _notify(
                client,
                "Pick one result first",
                "Several samples are shown: click the one you want to keep, then export.",
                color="yellow",
            )
            return None
        return motions[0]

    @gui_export.on_click
    def _(event) -> None:
        session = _session()
        if session is None:
            return
        if state.retargeter is None:
            _notify(client, "No character", "Pick a studio character (or Load FBX...) before exporting.", color="yellow")
            return
        motion = _export_motion(session)
        if motion is None:
            return
        name = _safe_name(gui_name.value)
        out_dir = Path(gui_export_dir.value or library.SETTINGS_DEFAULTS["export_dir"])
        exporting = client.add_notification(title="Exporting...", body=name, loading=True, with_close_button=False)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            library.save_settings(export_dir=str(out_dir))
            joints_pos = motion.joints_pos.clone()
            # Start the clip at the origin.
            start = joints_pos[0, session.skeleton.root_idx].clone()
            joints_pos[..., 0] -= start[0]
            joints_pos[..., 2] -= start[2]
            rot, pos = soma77_globals(session.skeleton, motion.joints_rot, joints_pos, rot_is_global=True)
            rt = state.retargeter
            with torch.no_grad():
                c_rot, c_pos = rt.retarget(rot, pos, root_motion=bool(gui_root_motion.value))
            bone_arm, root_arm = rt.to_armature(
                c_rot, c_pos, root_motion=bool(gui_root_motion.value), skeleton_hips_height=state.skeleton_hips_height
            )
            fbx_path = out_dir / f"{name}.fbx"
            gs_character.write_fbx(state.rig, bone_arm, root_arm, float(session.model_fps), str(fbx_path), name)
            written = [fbx_path.name]
            if gui_bvh.value:
                bvh_path = out_dir / f"{name}_SOMA.bvh"
                save_motion_bvh(
                    str(bvh_path),
                    motion.joints_local_rot,
                    joints_pos[:, session.skeleton.root_idx, :],
                    skeleton=session.skeleton,
                    fps=float(session.model_fps),
                )
                written.append(bvh_path.name)
            if gui_project.value and state.save_project is not None:
                project_dir = out_dir / f"{name}_project"
                if project_dir.exists():
                    import shutil

                    shutil.rmtree(project_dir)
                state.save_project(client, session, str(project_dir))
                with open(project_dir / PROJECT_FILE, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(
                        {
                            "character": state.rig.name,
                            "fbx_path": state.rig.fbx_path,
                            "skeleton_hips_height": state.skeleton_hips_height,
                            "root_motion": bool(gui_root_motion.value),
                            "animation_name": name,
                        },
                        f,
                        indent=2,
                    )
                written.append(project_dir.name)
                gui_project_path.value = str(project_dir)
            _notify(client, "Exported!", f"Saved {', '.join(written)} to {out_dir}", color="green", seconds=10.0)
        except Exception as e:  # noqa: BLE001 - shown to the user
            traceback.print_exc()
            _notify(client, "Export failed", str(e), color="red", seconds=15.0)
        finally:
            exporting.remove()

    @gui_open_folder.on_click
    def _(_event) -> None:
        out_dir = Path(gui_export_dir.value)
        out_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(str(out_dir))  # noqa: S606 - local tool, opens Explorer

    @gui_open_project.on_click
    def _(_event) -> None:
        project_dir = Path(gui_project_path.value.strip().strip('"'))
        if not project_dir.is_dir() or state.load_project is None:
            _notify(client, "Project not found", f"No project folder at {project_dir}", color="red")
            return
        info_path = project_dir / PROJECT_FILE
        if info_path.is_file():
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            display = info.get("character", "")
            if display in state.entries:
                gui_character.value = display
            elif info.get("fbx_path") and os.path.isfile(info["fbx_path"]):
                _select(library.LibraryEntry(display, info["fbx_path"], info.get("skeleton_hips_height")))
            gui_name.value = info.get("animation_name", gui_name.value)
            gui_root_motion.value = bool(info.get("root_motion", True))
        state.load_project(client, str(project_dir))

    _refresh_library()
    _set_status()
    last = settings.get("last_character")
    if last and last in state.entries:
        # Restore the last character once the session exists (create_gui runs before it).
        def _restore():
            for _ in range(100):
                if _session() is not None:
                    break
                threading.Event().wait(0.1)
            gui_character.value = last

        threading.Thread(target=_restore, daemon=True).start()
