from __future__ import annotations

from typing import Dict, List, Set

ACTION_GROUPS: Dict[str, List[str]] = {
    "break_open": [
        "break", "breaking", "break open", "breaking open",
        "crack", "cracking", "split", "splitting", "split open",
        "open", "opening", "shell", "shelling",
    ],
    "separate_egg_parts": [
        "separate", "separating", "yolk", "white", "egg white", "egg yolk",
    ],
    "peel_remove_outer": [
        "peel", "peeling", "remove peel", "removing peel",
        "remove shell", "removing shell", "strip", "stripping",
    ],
    "cut_divide": [
        "cut", "cutting", "slice", "slicing", "chop", "chopping",
        "dice", "dicing", "halve", "halving",
    ],
    "mix_agitate": [
        "mix", "mixing", "stir", "stirring", "whisk", "whisking",
        "beat", "beating", "blend", "blending",
    ],
    "pour_transfer": [
        "pour", "pouring", "add", "adding", "transfer", "transferring",
        "empty", "emptying",
    ],
    "hold_pick_place": [
        "hold", "holding", "pick", "picking", "pick up", "picking up",
        "place", "placing", "put", "putting", "grab", "grabbing",
    ],
    "squeeze_press": [
        "squeeze", "squeezing", "press", "pressing", "pinch", "pinching",
    ],
}

def action_groups_as_sets() -> Dict[str, Set[str]]:
    return {group: set(values) for group, values in ACTION_GROUPS.items()}