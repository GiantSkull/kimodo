# Copyright Giant Skull: daniel@giantskull.com
"""FBX reading and writing with Blender's ``bpy`` module.

Run as a separate process (see ``kimodo.gs.fbx``) so bpy never shares a process with torch / viser threads, and a bad
FBX cannot take the app down:

    python -m kimodo.gs.fbx_bpy read <character.fbx> <out.npz>
    python -m kimodo.gs.fbx_bpy write <character.fbx> <anim.npz> <out.fbx>

Bone matrices and vertices in the npz files are in ARMATURE space (bone ``matrix_local``: unit-scale rotations, the
file's own length units). ``armature_world`` maps armature space to Blender world space (Z-up, metres), so callers
derive the up axis and unit scale from it.
"""

import sys

import numpy as np

import bpy  # noqa: E402  (only available inside the bpy wheel)
from mathutils import Matrix  # noqa: E402

MAX_INFLUENCES = 4


def _reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _import_fbx(path):
    _reset_scene()
    bpy.ops.import_scene.fbx(
        filepath=path,
        use_anim=False,
        ignore_leaf_bones=True,
        automatic_bone_orientation=False,
        use_custom_props=False,
    )
    armatures = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError(f"No skeleton (armature) found in {path}")
    # The deforming armature is the one most meshes are skinned to; fall back to the biggest.
    def _score(arm):
        users = sum(1 for o in bpy.context.scene.objects if o.type == "MESH" and _mesh_armature(o) is arm)
        return (users, len(arm.data.bones))

    armature = max(armatures, key=_score)
    return armature


def _mesh_armature(obj):
    for mod in obj.modifiers:
        if mod.type == "ARMATURE" and mod.object is not None:
            return mod.object
    return obj.parent if obj.parent is not None and obj.parent.type == "ARMATURE" else None


def _bone_order(armature):
    """Bones ordered parents first, with parent indices."""
    names = []
    parents = []
    index = {}

    def _visit(bone, parent_idx):
        index[bone.name] = len(names)
        names.append(bone.name)
        parents.append(parent_idx)
        for child in bone.children:
            _visit(child, index[bone.name])

    for bone in armature.data.bones:
        if bone.parent is None:
            _visit(bone, -1)
    return names, parents


def read(fbx_path, out_path):
    armature = _import_fbx(fbx_path)
    names, parents = _bone_order(armature)
    bone_index = {n: i for i, n in enumerate(names)}
    arm_world = np.array(armature.matrix_world)
    arm_world_inv = np.linalg.inv(arm_world)
    rest_arm = np.array([np.array(armature.data.bones[n].matrix_local) for n in names], dtype=np.float64)

    verts_all = []
    faces_all = []
    weights_all = []
    indices_all = []
    vert_offset = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or _mesh_armature(obj) is not armature:
            continue
        # Rest-pose geometry: disable the armature modifier while evaluating.
        for mod in obj.modifiers:
            if mod.type == "ARMATURE":
                mod.show_viewport = False
        depsgraph.update()
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        mesh.calc_loop_triangles()
        nv = len(mesh.vertices)
        co = np.empty(nv * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", co)
        co = co.reshape(nv, 3)
        to_arm = arm_world_inv @ np.array(obj.matrix_world)
        co = co @ to_arm[:3, :3].T + to_arm[:3, 3]
        tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
        mesh.loop_triangles.foreach_get("vertices", tris)
        tris = tris.reshape(-1, 3)

        group_to_bone = {g.index: bone_index.get(g.name, -1) for g in obj.vertex_groups}
        w = np.zeros((nv, MAX_INFLUENCES), dtype=np.float32)
        j = np.zeros((nv, MAX_INFLUENCES), dtype=np.int32)
        for v in mesh.vertices:
            infl = [(g.weight, group_to_bone.get(g.group, -1)) for g in v.groups]
            infl = sorted([x for x in infl if x[1] >= 0 and x[0] > 0.0], reverse=True)[:MAX_INFLUENCES]
            total = sum(x[0] for x in infl)
            if total <= 0.0:
                continue
            for k, (wt, b) in enumerate(infl):
                w[v.index, k] = wt / total
                j[v.index, k] = b
        eval_obj.to_mesh_clear()

        verts_all.append(co)
        faces_all.append(tris + vert_offset)
        weights_all.append(w)
        indices_all.append(j)
        vert_offset += nv

    if verts_all:
        verts = np.concatenate(verts_all)
        faces = np.concatenate(faces_all)
        weights = np.concatenate(weights_all)
        indices = np.concatenate(indices_all)
    else:
        verts = np.zeros((0, 3))
        faces = np.zeros((0, 3), dtype=np.int64)
        weights = np.zeros((0, MAX_INFLUENCES), dtype=np.float32)
        indices = np.zeros((0, MAX_INFLUENCES), dtype=np.int32)

    np.savez(
        out_path,
        armature_name=np.array(armature.name),
        bone_names=np.array(names),
        bone_parents=np.array(parents, dtype=np.int32),
        rest_arm=rest_arm,
        armature_world=arm_world,
        verts=verts,
        faces=faces,
        skin_weights=weights,
        skin_indices=indices,
    )


def write(fbx_path, anim_path, out_path):
    """Key the character's skeleton with armature-space bone matrices and export an animation-only FBX."""
    anim = np.load(anim_path, allow_pickle=False)
    names = [str(n) for n in anim["bone_names"]]
    pose = anim["bone_arm"]  # (T, B, 4, 4) armature space
    fps = float(anim["fps"])
    take_name = str(anim["take_name"])

    armature = _import_fbx(fbx_path)
    # Animation only: drop every mesh so the FBX imports onto an existing skeleton.
    for obj in list(bpy.context.scene.objects):
        if obj.type != "ARMATURE" or obj is not armature:
            bpy.data.objects.remove(obj, do_unlink=True)

    scene = bpy.context.scene
    num_frames = pose.shape[0]
    scene.render.fps = int(round(fps))
    scene.render.fps_base = scene.render.fps / fps
    scene.frame_start = 0
    scene.frame_end = num_frames - 1

    root = anim["root_arm"]  # (T, 3) armature-space root motion of the armature object
    obj_world = np.array(armature.matrix_world)
    if armature.name == "Armature":
        # Unreal skips the transform of an FBX node named "Armature", and Blender keeps the Y-up -> Z-up rotation
        # there. Bake that rotation into the bones so the root bone carries it, and re-express the poses in the new
        # armature space. The unit scale stays on the object: Unreal ignores it too, so translations stay in file units.
        obj_rot = obj_world[:3, :3] / np.linalg.norm(obj_world[:3, 0])
        bpy.ops.object.select_all(action="DESELECT")
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        pose = pose.copy()
        pose[..., :3, :3] = np.einsum("ij,tbjk->tbik", obj_rot, pose[..., :3, :3])
        pose[..., :3, 3] = pose[..., :3, 3] @ obj_rot.T
        root = root @ obj_rot.T

    bones = armature.data.bones
    action = bpy.data.actions.new(take_name)
    armature.animation_data_create()
    armature.animation_data.action = action

    frames = np.arange(num_frames, dtype=np.float32)

    def _key(data_path, values, group):
        for axis in range(values.shape[1]):
            fc = action.fcurves.new(data_path, index=axis, action_group=group)
            fc.keyframe_points.add(num_frames)
            co = np.empty(num_frames * 2, dtype=np.float32)
            co[0::2] = frames
            co[1::2] = values[:, axis]
            fc.keyframe_points.foreach_set("co", co)
            fc.keyframe_points.foreach_set("interpolation", [1] * num_frames)  # LINEAR
            fc.update()

    if np.abs(root).max() > 0.0:
        # The object is the skeleton's root bone (e.g. "root" in Unreal exports): root motion moves the object. Unreal
        # ignores this node's unit scale, so key it in file units like the bones (rotation only, no scale).
        loc0 = np.array(armature.location)
        obj_rot = obj_world[:3, :3] / np.linalg.norm(obj_world[:3, 0])
        _key("location", loc0[None] + root @ obj_rot.T, "Object Transforms")
    for b_idx, name in enumerate(names):
        bone = bones.get(name)
        if bone is None:
            continue
        pose_arm = pose[:, b_idx]
        rest_arm = np.array(bone.matrix_local)
        if bone.parent is not None:
            parent_pose_arm = pose[:, names.index(bone.parent.name)]
            rest_rel = np.linalg.inv(np.array(bone.parent.matrix_local)) @ rest_arm
            pose_rel = np.linalg.inv(parent_pose_arm) @ pose_arm
        else:
            rest_rel = rest_arm
            pose_rel = pose_arm
        basis = np.linalg.inv(rest_rel)[None] @ pose_rel  # == pose_bone.matrix_basis

        locs = basis[:, :3, 3]
        quats = np.array([Matrix(m.tolist()).to_quaternion() for m in basis])
        # Keep quaternions continuous so interpolation never flips.
        for t in range(1, num_frames):
            if np.dot(quats[t], quats[t - 1]) < 0.0:
                quats[t] = -quats[t]

        pb = armature.pose.bones[name]
        pb.rotation_mode = "QUATERNION"
        _key(pb.path_from_id("location"), locs, name)
        _key(pb.path_from_id("rotation_quaternion"), quats, name)

    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_scene.fbx(
        filepath=out_path,
        use_selection=True,
        object_types={"ARMATURE"},
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_simplify_factor=0.0,
        armature_nodetype="NULL",
        use_armature_deform_only=False,
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_UNITS",
    )


def main(argv):
    if len(argv) >= 3 and argv[0] == "read":
        read(argv[1], argv[2])
    elif len(argv) >= 4 and argv[0] == "write":
        write(argv[1], argv[2], argv[3])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
