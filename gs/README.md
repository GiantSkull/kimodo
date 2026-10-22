# GS Kimodo

Giant Skull's fork of [NVIDIA Kimodo](https://github.com/nv-tlabs/kimodo): type what a character should do, get a
placeholder animation already on one of our characters, and export it as an FBX that imports straight onto that
character's skeleton in Unreal (or Maya).

## For artists

1. Double-click `GSKimodo.bat`. The first time, it installs everything (about 10 GB of downloads) and asks for the
   studio Hugging Face token: ask your lead.
2. Your browser opens GS Kimodo. In **Giant Skull > Character**, pick the character (or **Load FBX...** for any
   skinned FBX).
3. Type what should happen in the timeline prompt (e.g. "a person walks forward and waves"), set the length and press
   **Generate**. With several samples, click the one you like to keep it.
4. In **Export to Game**, give the animation a name and press **Export Animation**. The FBX (and a project folder you
   can reopen with **Open Project** to change it later) goes to the export folder.
5. In Unreal, import the FBX with the character's Skeleton selected (Import Mesh off).

Tips: one action per prompt works best; chain several prompts on the timeline for sequences. Turn **Root motion** off
for loops that should stay in place. Constraints (Generate tab) pin hands, feet, poses or the walking path.

## Character library

The dropdown lists every `.fbx` under the library folder (Library settings, or the `GS_KIMODO_LIBRARY` environment
variable). Export library characters FROM UNREAL (right-click the Skeletal Mesh > Asset Actions > Export, FBX), not
from source art: animations play on the engine asset's proportions. An optional JSON next to the FBX with the same
name tunes it:

```json
{
  "display_name": "Barbarian",
  "skeleton_hips_height": 89.513,
  "bones": {"spine_05": "Chest", "prop_r": null}
}
```

- `skeleton_hips_height`: pelvis height (cm) of the Unreal Skeleton asset's reference pose when it differs from this
  mesh's. Our characters share `SKEL_GS_metahuman_base` (89.513), which scales pelvis motion per mesh; without it the
  character floats or sinks in Unreal.
- `bones`: bone map overrides onto Kimodo's SOMA joints (`Hips`, `Spine1`, `Spine2`, `Chest`, `Neck1`, `Neck2`, `Head`,
  `LeftShoulder`, `LeftArm`, `LeftForeArm`, `LeftHand`, `LeftHandIndex1`..`4`, `LeftLeg`, `LeftShin`, `LeftFoot`,
  `LeftToeBase`, and the Right equivalents); `null` excludes a bone. Unreal, Mixamo and HumanIK names are recognised
  automatically.

## How it works (maintainers)

- `kimodo/gs/fbx_bpy.py`: FBX read/write with Blender's `bpy` module, run in a child process.
- `kimodo/gs/bonemap.py`: bone name recognition. `kimodo/gs/retarget.py`: SOMA -> character retarget (global rotation
  transfer with a per-limb rest alignment; character bone lengths; hips scaled by height) and skinning for the viewer.
- `kimodo/gs/panel.py`: the Giant Skull UI folder. Upstream files only call into it (`kimodo/demo/ui.py`, `app.py`,
  `kimodo/viz/scene.py`; look for "Giant Skull"), so upstream merges stay small.
- Unreal FBX traps (verified): an FBX root node named `Armature` is dropped by Unreal, so its rotation is baked into the
  bones; Unreal exports load with the `root` bone as the armature object, so root motion is keyed on the object in
  file units.
