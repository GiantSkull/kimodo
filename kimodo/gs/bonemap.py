# Copyright Giant Skull: daniel@giantskull.com
"""Map a character's bones onto Kimodo's SOMA skeleton.

``guess_bone_map`` recognises the common naming conventions (Unreal mannequin / UEFN, Mixamo, HumanIK, most Maya and
Blender rigs) from bone names and the hierarchy. The ``bones`` of a library character's sidecar JSON (see
``kimodo.gs.library``, ``{"target bone": "SOMA joint", ...}``) override or extend the guess; ``null`` excludes a bone.
"""

import re
from typing import Dict, List, Optional, Sequence

SOMA_SPINE = ["Spine1", "Spine2", "Chest"]
SOMA_NECK = ["Neck1", "Neck2"]
SOMA_FINGERS = {
    "Thumb": ["Thumb1", "Thumb2", "Thumb3"],
    "Index": ["Index1", "Index2", "Index3", "Index4"],
    "Middle": ["Middle1", "Middle2", "Middle3", "Middle4"],
    "Ring": ["Ring1", "Ring2", "Ring3", "Ring4"],
    "Pinky": ["Pinky1", "Pinky2", "Pinky3", "Pinky4"],
}

# Name tokens of bones that never take part in retargeting (they follow their parent).
_IGNORE_TOKENS = {
    "twist", "roll", "ik", "corrective", "helper", "socket", "weapon", "prop", "attach", "end", "nub", "dummy", "null",
    "offset", "ctrl", "cloth", "hair", "eye", "jaw", "tongue", "teeth", "brow", "lid", "lip", "cheek", "nose", "ear",
}


def _ignored(base: str) -> bool:
    return any(re.sub(r"\d+$", "", tok) in _IGNORE_TOKENS for tok in base.split("_"))

_SIDE_TOKENS = [
    (re.compile(r"(^|[_\.\s\-])(l|lf|lft|left)(?=$|[_\.\s\-\d])", re.I), "L"),
    (re.compile(r"(^|[_\.\s\-])(r|rt|rgt|right)(?=$|[_\.\s\-\d])", re.I), "R"),
    (re.compile(r"^left", re.I), "L"),
    (re.compile(r"^right", re.I), "R"),
    (re.compile(r"left$", re.I), "L"),
    (re.compile(r"right$", re.I), "R"),
]

_ROLES = [
    ("hips", re.compile(r"^(pelvis|hips?|cog|root_?hips?)$")),
    ("spine", re.compile(r"^(spine|chest|upperchest|upper_chest|torso|back)(_?\d+)?$")),
    ("neck", re.compile(r"^neck(_?\d+)?$")),
    ("head", re.compile(r"^head$")),
    ("clavicle", re.compile(r"^(clavicle|shoulder|collar|collarbone)$")),
    ("upperarm", re.compile(r"^(upperarm|upper_arm|uparm|arm|humerus)$")),
    ("lowerarm", re.compile(r"^(lowerarm|lower_arm|forearm|fore_arm|elbow)$")),
    ("hand", re.compile(r"^(hand|wrist)$")),
    ("thigh", re.compile(r"^(thigh|upleg|up_leg|upperleg|upper_leg|femur)$")),
    ("leg", re.compile(r"^leg$")),  # thigh or calf, resolved from the hierarchy
    ("calf", re.compile(r"^(calf|shin|lowerleg|lower_leg|knee|tibia)$")),
    ("foot", re.compile(r"^(foot|ankle)$")),
    ("toe", re.compile(r"^(ball|toe|toes|toebase|toe_base)(_?\d+)?$")),
    ("finger", re.compile(r"^(thumb|index|middle|ring|pinky|pinkie|little)[_\-]?(\d+|metacarpal|meta)?$")),
]


def _strip_namespace(name: str) -> str:
    return re.split(r"[:|]", name)[-1]


def _side_and_base(name: str):
    base = _strip_namespace(name)
    side = None
    for pattern, s in _SIDE_TOKENS:
        if pattern.search(base):
            side = s
            base = pattern.sub(lambda m: m.group(1) if m.lastindex and m.group(1) else "", base, count=1)
            break
    # CamelCase -> snake for the role patterns (UpperArm -> upper_arm), then collapse separators.
    base = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", base).lower()
    base = re.sub(r"[\.\s\-]+", "_", base).strip("_")
    return side, base


def _role(base: str) -> Optional[str]:
    # Mixamo / SOMA fingers: LeftHandIndex1 -> hand_index1
    if base.startswith("hand_") and _ROLES[-1][1].match(base[5:]):
        return "finger"
    compact = base.replace("_", "")
    for role, pattern in _ROLES:
        if pattern.match(base) or pattern.match(compact):
            return role
    return None


def _depths(parents: Sequence[int]) -> List[int]:
    depth = []
    for p in parents:
        depth.append(0 if p < 0 else depth[p] + 1)
    return depth


def _is_ancestor(parents: Sequence[int], a: int, b: int) -> bool:
    while b >= 0:
        b = parents[b]
        if b == a:
            return True
    return False


def guess_bone_map(names: Sequence[str], parents: Sequence[int]) -> Dict[str, str]:
    """Returns {target bone name: SOMA joint name} for the bones it recognises."""
    depth = _depths(parents)
    info = []
    for i, name in enumerate(names):
        side, base = _side_and_base(name)
        role = None if _ignored(base) else _role(base)
        if role == "finger" and base.startswith("hand_"):
            base = base[5:]
        info.append((side, base, role))

    def _pick(role, side=None):
        cands = [i for i, (s, _, r) in enumerate(info) if r == role and (side is None or s == side)]
        return sorted(cands, key=lambda i: depth[i])

    result: Dict[str, str] = {}
    hips = _pick("hips")
    hips_idx = hips[0] if hips else None
    if hips_idx is not None:
        result[names[hips_idx]] = "Hips"

    def _chain(role, soma_chain):
        bones = [i for i in _pick(role) if hips_idx is None or _is_ancestor(parents, hips_idx, i)]
        if not bones:
            return
        n = len(bones)
        for k, i in enumerate(bones):
            src = soma_chain[0] if n == 1 else soma_chain[int(round(k * (len(soma_chain) - 1) / (n - 1)))]
            result[names[i]] = src

    _chain("spine", SOMA_SPINE)
    _chain("neck", SOMA_NECK)
    head = _pick("head")
    if head:
        result[names[head[0]]] = "Head"

    for side, soma_side in (("L", "Left"), ("R", "Right")):
        for role, soma in (("clavicle", "Shoulder"), ("upperarm", "Arm"), ("lowerarm", "ForeArm"), ("hand", "Hand"),
                           ("thigh", "Leg"), ("calf", "Shin"), ("foot", "Foot")):
            bones = _pick(role, side)
            if bones:
                result[names[bones[0]]] = soma_side + soma
        # "Leg": Mixamo / HumanIK use UpLeg + Leg (Leg = calf); a lone "Leg" chain is thigh then calf.
        legs = _pick("leg", side)
        if legs:
            has_thigh = any(soma_side + "Leg" == v for v in result.values())
            result[names[legs[0]]] = soma_side + ("Shin" if has_thigh else "Leg")
            if not has_thigh and len(legs) > 1:
                result[names[legs[1]]] = soma_side + "Shin"
        toes = _pick("toe", side)
        if toes:
            result[names[toes[0]]] = soma_side + "ToeBase"

        hand = _pick("hand", side)
        for finger, soma_chain in SOMA_FINGERS.items():
            key = {"Pinky": ("pinky", "pinkie", "little")}.get(finger, (finger.lower(),))
            bones = [i for i, (s, b, r) in enumerate(info) if r == "finger" and s == side and b.startswith(key)]
            if hand:
                bones = [i for i in bones if _is_ancestor(parents, hand[0], i)]
            bones.sort(key=lambda i: depth[i])
            # Align from the tip: a 3-bone finger maps onto the last three SOMA joints (the 4th is the metacarpal).
            # The thumb has no metacarpal joint in SOMA, so it aligns from the base.
            if finger == "Thumb":
                pairs = zip(bones, soma_chain)
            else:
                pairs = zip(reversed(bones), reversed(soma_chain))
            for i, src in pairs:
                result[names[i]] = soma_side + "Hand" + src
    return result




def resolve_bone_map(names: Sequence[str], parents: Sequence[int], preset: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, str]:
    bone_map = guess_bone_map(names, parents)
    if preset:
        for target, src in preset.items():
            if src is None:
                bone_map.pop(target, None)
            elif target in names:
                bone_map[target] = src
    return bone_map


REQUIRED_SOMA = ["Hips", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftLeg", "RightLeg", "LeftShin", "RightShin"]


def missing_required(bone_map: Dict[str, str]) -> List[str]:
    mapped = set(bone_map.values())
    return [j for j in REQUIRED_SOMA if j not in mapped]
