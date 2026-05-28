"""Generate one (input, answer) PNG pair per PaintBench task for the website.

Saves to <OUT_DIR>/ex_<task>_{input,answer}.png. Set OUT_DIR below or pass --out-dir.
"""
from __future__ import annotations
import hashlib
import importlib
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.background import BackgroundSpec
from core.colors import STANDARD_PALETTE

OUT_DIR = os.environ.get("OUT_DIR", "website_examples")

TASKS = [
    ("tasks.translation",      "translation"),
    ("tasks.rotation",         "rotation"),
    ("tasks.reflection",       "reflection"),
    ("tasks.scaling",          "scaling"),
    ("tasks.shearing",         "shearing"),
    ("tasks.construction",     "construction"),
    ("tasks.removal",          "removal"),
    ("tasks.copying",          "copying"),
    ("tasks.border",           "border"),
    ("tasks.cropping",         "cropping"),
    ("tasks.recolor",          "recolor"),
    ("tasks.flood_fill",       "flood_fill"),
    ("tasks.blending",         "blending"),
    ("tasks.gradient",         "gradient"),
    ("tasks.point_operations", "point_operations"),
    ("tasks.comparison",       "comparison"),
    ("tasks.ordering",         "ordering"),
    ("tasks.pattern",          "pattern"),
    ("tasks.counting",         "counting"),
    ("tasks.legend",           "legend"),
]

def _make_seed(task: str, slot: int = 0) -> int:
    key = f"paintbench|{task}|baseline|default|{slot}|0"
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)


def _color_split(palette: dict, seed: int):
    rng = random.Random(seed ^ 0xC0FFEE)
    items = list(palette.items())
    rng.shuffle(items)
    return items[0][1], items[1][1], [rgb for _, rgb in items[2:]]


def generate_one(module_path: str, task_name: str, slot: int = 3):
    seed = _make_seed(task_name, slot)
    bg_rgb, holdout_rgb, obj_colors = _color_split(STANDARD_PALETTE, seed)
    W, H = 512, 512  # smaller for website thumbnails
    bg_spec = BackgroundSpec(colors=[bg_rgb])
    mod = importlib.import_module(module_path)
    prob = mod.generate(seed=seed, bg_spec=bg_spec, W=W, H=H, obj_colors=obj_colors)
    return prob


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for module_path, task_name in TASKS:
        out_input  = os.path.join(OUT_DIR, f"ex_{task_name}_input.png")
        out_answer = os.path.join(OUT_DIR, f"ex_{task_name}_answer.png")
        if os.path.exists(out_input) and os.path.exists(out_answer):
            print(f"  skip {task_name} (already exists)")
            continue
        print(f"  generating {task_name}...")
        try:
            prob = generate_one(module_path, task_name)
            prob.input_image.save(out_input)
            prob.answer_image.save(out_answer)
            print(f"    -> saved ({prob.input_image.size})")
        except Exception as e:
            print(f"    ERROR: {e}")
            # Try different slots
            for slot in range(1, 8):
                try:
                    prob = generate_one(module_path, task_name, slot)
                    prob.input_image.save(out_input)
                    prob.answer_image.save(out_answer)
                    print(f"    -> saved slot={slot}")
                    break
                except Exception:
                    pass

    print("done")


if __name__ == "__main__":
    main()
