# Copyright Giant Skull: daniel@giantskull.com
"""Retarget Kimodo SOMA motion onto a studio character, and skin the character for the viewer.

Kimodo motion is Y-up, +Z forward, metres, with global joint rotations relative to SOMA's standard T-pose (identity
at rest). Each mapped character bone gets the SOMA joint's global rotation, applied after an alignment rotation that
turns the character's rest pose (A-pose or anything else) into SOMA's T-pose direction for that limb. Unmapped bones
(twist, helpers, face) keep their rest pose relative to their parent. Bone lengths always stay the character's own;
only the hips translation is taken from Kimodo, scaled by the hips height ratio.
"""

from typing import Optional, Tuple

import numpy as np
import torch

from kimodo.skeleton import SOMASkeleton77, batch_rigid_transform, global_rots_to_local_rots

from .character import CharacterRig

# Children used to aim each limb when computing the rest alignment: (target-side role is found through the bone map).
_AIM = {
    "Shoulder": "Arm",
    "Arm": "ForeArm",
    "ForeArm": "Hand",
    "Hand": "HandMiddle1",
    "Leg": "Shin",
    "Shin": "Foot",
    "Foot": "ToeBase",
}
_FINGER_NEXT = {"1": "2", "2": "3", "3": "4"}


def _normalize(v):
    return v / max(np.linalg.norm(v), 1e-9)


def _min_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation turning direction a onto direction b."""
    a = _normalize(a)
    b = _normalize(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        axis = _normalize(np.cross(a, [1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.cross(a, [0.0, 1.0, 0.0]))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _aim_child(soma_name: str) -> Optional[str]:
    for side in ("Left", "Right"):
        if soma_name.startswith(side):
            part = soma_name[len(side):]
            if part in _AIM:
                return side + _AIM[part]
            if part.startswith("Hand") and part[-1] in _FINGER_NEXT:
                return side + part[:-1] + _FINGER_NEXT[part[-1]]
    return None


class Retargeter:
    """Precomputed SOMA -> character retarget; ``retarget`` runs batched over frames."""

    def __init__(self, rig: CharacterRig, device="cpu"):
        self.rig = rig
        self.device = torch.device(device)
        self.soma = SOMASkeleton77(load=True)
        soma_names = [n for n, _ in SOMASkeleton77.bone_order_names_with_parents]
        soma_idx = {n: i for i, n in enumerate(soma_names)}
        src_rest = self.soma.neutral_joints.detach().cpu().numpy().astype(np.float64)  # (77, 3) T-pose

        names = rig.bone_names
        parents = rig.bone_parents
        B = len(names)
        bone_idx = {n: i for i, n in enumerate(names)}
        rest_r = rig.rest_arm[:, :3, :3]
        rest_p = rig.rest_arm[:, :3, 3]

        # Armature space -> Kimodo space: rotation C, uniform scale s (to metres), offset o.
        aw = rig.armature_world
        s = float(np.linalg.norm(aw[:3, 0]))
        up_t = _normalize(aw[:3, :3].T @ np.array([0.0, 0.0, 1.0]) / s**2)
        inv_map = {v: k for k, v in rig.bone_map.items()}

        def _side_vec(get_pos, left_names, right_names, up):
            vecs = []
            for ln, rn in zip(left_names, right_names):
                lp, rp = get_pos(ln), get_pos(rn)
                if lp is not None and rp is not None:
                    vecs.append(lp - rp)
            v = np.sum(vecs, axis=0)
            v = v - np.dot(v, up) * up
            return _normalize(v)

        lr = (["LeftLeg", "LeftArm", "LeftShoulder"], ["RightLeg", "RightArm", "RightShoulder"])
        left_t = _side_vec(lambda n: rest_p[bone_idx[inv_map[n]]] if n in inv_map else None, *lr, up_t)
        up_s = np.array([0.0, 1.0, 0.0])
        left_s = _side_vec(lambda n: src_rest[soma_idx[n]], *lr, up_s)
        basis_t = np.stack([left_t, up_t, np.cross(left_t, up_t)], axis=1)
        basis_s = np.stack([left_s, up_s, np.cross(left_s, up_s)], axis=1)
        C = basis_s @ basis_t.T

        k_rest_r = np.einsum("ij,bjk->bik", C, rest_r)
        k_rest_p = s * rest_p @ C.T
        if len(rig.verts):
            k_verts = s * rig.verts @ C.T
            floor = k_verts[:, 1].min()
        else:
            k_verts = np.zeros((0, 3))
            floor = k_rest_p[:, 1].min()
        hips_name = inv_map.get("Hips")
        if hips_name is None:
            raise ValueError(f"{rig.name}: no bone is mapped to the hips")
        hips = bone_idx[hips_name]
        o = np.array([-k_rest_p[hips, 0], -floor, -k_rest_p[hips, 2]])
        k_rest_p = k_rest_p + o
        k_verts = k_verts + o

        # Hips height ratio for the root translation.
        src_floor = src_rest[:, 1].min()
        self.hips_scale = float(k_rest_p[hips, 1] / max(src_rest[soma_idx["Hips"], 1] - src_floor, 1e-6))
        self.src_floor = float(src_floor)

        # Per-bone rest alignment A (Kimodo space): rotates the character's rest limb direction onto SOMA's T-pose.
        align = np.tile(np.eye(3), (B, 1, 1))
        src_of = np.full(B, -1, dtype=np.int64)
        for b in range(B):
            soma_name = rig.bone_map.get(names[b])
            if soma_name is None or soma_name not in soma_idx:
                continue
            src_of[b] = soma_idx[soma_name]
        for b in range(B):
            if src_of[b] < 0:
                continue
            soma_name = soma_names[src_of[b]]
            child = _aim_child(soma_name)
            if child and child in inv_map and child in soma_idx:
                d_t = k_rest_p[bone_idx[inv_map[child]]] - k_rest_p[b]
                d_s = src_rest[soma_idx[child]] - src_rest[src_of[b]]
                if np.linalg.norm(d_t) > 1e-6 and np.linalg.norm(d_s) > 1e-6:
                    align[b] = _min_rotation(d_t, d_s)
                    continue
            # No aim child (hands without fingers, toes, head): inherit the nearest mapped ancestor's alignment.
            p = parents[b]
            while p >= 0 and src_of[p] < 0:
                p = parents[p]
            if p >= 0 and soma_name not in ("Hips", "Head") and not soma_name.startswith(("Spine", "Chest", "Neck")):
                align[b] = align[p]

        # Bones above the hips (root etc.) stay at rest, optionally following the hips on the ground.
        above_hips = np.zeros(B, dtype=bool)
        p = parents[hips]
        while p >= 0:
            above_hips[p] = True
            p = parents[p]

        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)  # noqa: E731
        self.C = C
        self.s = s
        self.o = o
        self.hips = hips
        self.src_hips = soma_idx["Hips"]
        self.parents = [int(x) for x in parents]
        self.src_of = [int(x) for x in src_of]
        self.above_hips = above_hips
        self.k_rest_r = f(k_rest_r)
        self.k_rest_p = f(k_rest_p)
        self.align_rest = f(np.einsum("bij,bjk->bik", align, k_rest_r))  # A_b @ R_b
        # Rest-relative rotation and offset of each bone in its parent's frame.
        rel_r = np.tile(np.eye(3), (B, 1, 1))
        rel_p = np.zeros((B, 3))
        for b in range(B):
            pb = parents[b]
            if pb >= 0:
                rel_r[b] = k_rest_r[pb].T @ k_rest_r[b]
                rel_p[b] = k_rest_r[pb].T @ (k_rest_p[b] - k_rest_p[pb])
        self.rel_r = f(rel_r)
        self.rel_p = f(rel_p)
        self.k_verts = f(k_verts)
        self.faces = rig.faces.astype(np.int32)
        self.skin_w = f(rig.skin_weights)
        self.skin_j = torch.as_tensor(rig.skin_indices, dtype=torch.long, device=self.device)
        self.inv_bind_r = self.k_rest_r.transpose(-1, -2)
        self.inv_bind_p = -torch.einsum("bij,bj->bi", self.inv_bind_r, self.k_rest_p)

    def retarget(self, soma_rot: torch.Tensor, soma_pos: torch.Tensor, root_motion: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """SOMA77 global rotations (T, 77, 3, 3) and positions (T, 77, 3) -> character globals in Kimodo space."""
        soma_rot = soma_rot.to(self.device, torch.float32)
        soma_pos = soma_pos.to(self.device, torch.float32)
        T = soma_rot.shape[0]
        B = len(self.parents)
        rot = [None] * B
        pos = [None] * B
        # Kimodo motion stands on y = 0, like the character after ``o``.
        hips_pos = soma_pos[:, self.src_hips] * self.hips_scale
        for b in range(B):
            if self.above_hips[b]:
                rot[b] = self.k_rest_r[b].expand(T, 3, 3)
                p = self.k_rest_p[b].expand(T, 3)
                if root_motion:
                    p = p + hips_pos * torch.tensor([1.0, 0.0, 1.0], device=self.device)
                pos[b] = p
                continue
            pb = self.parents[b]
            if self.src_of[b] >= 0:
                rot[b] = soma_rot[:, self.src_of[b]] @ self.align_rest[b]
            elif pb >= 0:
                rot[b] = rot[pb] @ self.rel_r[b]
            else:
                rot[b] = self.k_rest_r[b].expand(T, 3, 3)
            if b == self.hips:
                pos[b] = hips_pos
            elif pb >= 0:
                pos[b] = pos[pb] + torch.einsum("tij,j->ti", rot[pb], self.rel_p[b])
            else:
                pos[b] = self.k_rest_p[b].expand(T, 3)
        return torch.stack(rot, dim=1), torch.stack(pos, dim=1)

    def skin(self, rot: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Linear blend skinning of the character for character globals (T, B, 3, 3), (T, B, 3) -> (T, V, 3)."""
        skin_r = rot @ self.inv_bind_r  # (T, B, 3, 3)
        skin_p = torch.einsum("tbij,bj->tbi", rot, self.inv_bind_p) + pos
        out = torch.zeros(rot.shape[0], self.k_verts.shape[0], 3, device=self.device)
        for k in range(self.skin_j.shape[1]):
            j = self.skin_j[:, k]
            w = self.skin_w[:, k][None, :, None]
            out += w * (torch.einsum("tvij,vj->tvi", skin_r[:, j], self.k_verts) + skin_p[:, j])
        return out

    @property
    def root_is_object(self) -> bool:
        """True when the skeleton's root bone is the armature object itself (Unreal FBX exports: object "root",
        first bone "pelvis"), so root motion must move the object. Unreal skips an object named "Armature"."""
        return not self.above_hips.any() and self.rig.armature_name != "Armature"

    @property
    def rest_hips_height(self) -> float:
        """Hips height of the character's rest pose in the file's units (e.g. cm)."""
        return float(self.k_rest_p[self.hips, 1]) / self.s

    def to_armature(
        self, rot: torch.Tensor, pos: torch.Tensor, root_motion: bool = True, skeleton_hips_height: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Character globals in Kimodo space -> armature-space bone matrices (T, B, 4, 4) for FBX export, and the
        armature object's root motion as armature-space offsets (T, 3) (zero unless ``root_is_object``).

        ``skeleton_hips_height`` (file units): the hips height of the engine Skeleton's reference pose, when it differs
        from this mesh's. Unreal shares animations per Skeleton and scales the pelvis translation by mesh / skeleton
        (AnimationScaled), so the exported hips motion is authored at the skeleton's proportions."""
        rot = rot.detach().cpu().numpy().astype(np.float64)
        pos = pos.detach().cpu().numpy().astype(np.float64).copy()
        T, B = rot.shape[:2]
        rest_hips = self.k_rest_p[self.hips].cpu().numpy()
        ground = pos[:, self.hips] * np.array([1.0, 0.0, 1.0]) if root_motion else np.zeros((T, 3))
        if skeleton_hips_height:
            factor = skeleton_hips_height / self.rest_hips_height
            delta = (pos[:, self.hips] - ground) * (factor - 1.0)
            below = np.array([not self.above_hips[b] for b in range(B)])
            pos[:, below] += delta[:, None]
        root = np.zeros((T, 3))
        if root_motion and self.root_is_object:
            root[:, [0, 2]] = ground[:, [0, 2]] - rest_hips[[0, 2]]
        out = np.tile(np.eye(4), (T, B, 1, 1))
        out[..., :3, :3] = np.einsum("ji,tbjk->tbik", self.C, rot)
        out[..., :3, 3] = ((pos - root[:, None] - self.o) @ self.C) / self.s
        return out, (root @ self.C) / self.s


def soma77_globals(skeleton, joints_rot: torch.Tensor, joints_pos: torch.Tensor, rot_is_global: bool = True):
    """Normalise Kimodo viewer poses (SOMA30 or SOMA77, local or global) to SOMA77 global rotations and positions."""
    if joints_rot.shape[1] == 77 and rot_is_global:
        return joints_rot, joints_pos
    skel77 = SOMASkeleton77(load=True)
    device = joints_rot.device
    if joints_rot.shape[1] != 77:
        local = global_rots_to_local_rots(joints_rot, skeleton) if rot_is_global else joints_rot
        local = skeleton.to_SOMASkeleton77(local)
        root = joints_pos[:, skeleton.root_idx : skeleton.root_idx + 1]
    else:
        local = joints_rot
        root = joints_pos[:, skel77.root_idx : skel77.root_idx + 1]
    neutral = skel77.neutral_joints[None].repeat((joints_rot.shape[0], 1, 1)).to(device)
    pos, rot = batch_rigid_transform(local, neutral, skel77.joint_parents.to(device), skel77.root_idx)
    return rot, pos + root


class CharacterSkin:
    """Drop-in for Kimodo's SOMASkin in the viewer: takes SOMA poses and returns the retargeted character's vertices."""

    def __init__(self, retargeter: Retargeter, skeleton):
        self.retargeter = retargeter
        self.skeleton = skeleton
        self.bind_vertices = retargeter.k_verts
        self.faces = torch.as_tensor(retargeter.faces)

    def skin(self, joint_rotmat, joint_pos, rot_is_global=False):
        rot, pos = soma77_globals(self.skeleton, joint_rotmat, joint_pos, rot_is_global)
        c_rot, c_pos = self.retargeter.retarget(rot, pos)
        return self.retargeter.skin(c_rot, c_pos)
