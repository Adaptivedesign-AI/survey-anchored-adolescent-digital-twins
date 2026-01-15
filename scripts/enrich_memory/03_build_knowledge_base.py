#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_build_knowledge_base.py

Deterministically merge per-video reviewer outputs into ONE merged_knowledge_base.json.

Defaults:
- Input reviewer outputs: ./extracted_data/**/**_reviewer_output.json
- Output file: ./knowledge_base/merged_knowledge_base.json
- source_json field: relative path starting with "extracted_data/..."

Output schema matches your provided merged_knowledge_base.json example:
{
  "total_files": <int>,
  "total_scenes": <int>,
  "data": [
    {
      "video_id": "...",
      "source_json": "extracted_data/<video>_extracted_data/<video>_reviewer_output.json",
      "identifiedCategory": "...",
      "title": "...",
      "metadata_description": "...",
      "text": [
        {
          "id": "Teen1",
          ... other teen fields ...,
          "scenes": [
            {
              "transcript_full": "...",
              "what_helped": "...",
              "other_notes": "...",
              "description": "...",
              "kb_id": "KB_000001"
            }
          ]
        }
      ]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


VIDEO_KEYS = [
    "video_id",
    "source_json",
    "identifiedCategory",
    "title",
    "metadata_description",
    "text",
]

SCENE_KEYS = [
    "transcript_full",
    "what_helped",
    "other_notes",
    "description",
    "kb_id",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def infer_video_id(fp: Path) -> str:
    # .../<video_id>_reviewer_output.json
    name = fp.name
    suffix = "_reviewer_output.json"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    # fallback: parent folder "<video_id>_extracted_data"
    parent = fp.parent.name
    if parent.endswith("_extracted_data"):
        return parent[: -len("_extracted_data")]
    return fp.stem


def relpath_posix(path: Path, base_dir: Path) -> str:
    rel = path.relative_to(base_dir)
    return str(rel).replace("\\", "/")


def ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def coerce_scene(scene: Any) -> Dict[str, Any]:
    """
    Keep ONLY the 5 required scene keys.
    Fill missing keys with "unknown" (except kb_id which will be overwritten later).
    """
    if not isinstance(scene, dict):
        scene = {}

    out = {}
    for k in SCENE_KEYS:
        if k == "kb_id":
            out[k] = ""  # will be overwritten
        else:
            v = scene.get(k)
            out[k] = v if (isinstance(v, str) and v.strip()) else "unknown"
    return out


def coerce_teen(teen: Any, teen_index: int) -> Dict[str, Any]:
    """
    Keep teen fields as-is, ensure it has:
    - id: "Teen{idx}" if missing
    - scenes: list of coerced scenes
    """
    if not isinstance(teen, dict):
        teen = {}

    out = dict(teen)

    # ensure id
    tid = out.get("id")
    if not (isinstance(tid, str) and tid.strip()):
        out["id"] = f"Teen{teen_index}"

    # ensure scenes
    scenes = out.get("scenes")
    if not isinstance(scenes, list):
        scenes = []
    out["scenes"] = [coerce_scene(s) for s in scenes]

    return out


def coerce_video_record(raw: Any, fp: Path, base_dir: Path) -> Dict[str, Any]:
    """
    Produce a video record with EXACT 6 keys:
    video_id, source_json, identifiedCategory, title, metadata_description, text
    """
    vid = infer_video_id(fp)
    src_rel = relpath_posix(fp, base_dir)

    # raw can be dict (preferred) or list (fallback)
    if isinstance(raw, dict):
        r = raw
    elif isinstance(raw, list):
        r = {"text": raw}
    else:
        r = {}

    identified = r.get("identifiedCategory")
    title = r.get("title")
    meta_desc = r.get("metadata_description")
    text = r.get("text")

    # hard defaults for missing keys
    identified = identified if (isinstance(identified, str) and identified.strip()) else "unknown"
    title = title if (isinstance(title, str) and title.strip()) else "unknown"
    meta_desc = meta_desc if (isinstance(meta_desc, str) and meta_desc.strip()) else "unknown"

    teens = ensure_list(text)
    coerced_teens = [coerce_teen(t, i + 1) for i, t in enumerate(teens)]

    out = {
        "video_id": vid,
        "source_json": src_rel,
        "identifiedCategory": identified,
        "title": title,
        "metadata_description": meta_desc,
        "text": coerced_teens,
    }

    # Ensure exact key set and order
    return {k: out[k] for k in VIDEO_KEYS}


def assign_kb_ids(data: List[Dict[str, Any]]) -> int:
    """
    Deterministically overwrite ALL scene kb_id as:
    KB_000001 ... in this fixed order:
    - sorted files by their reviewer_output path
    - teen list order
    - scene list order
    Returns total scene count.
    """
    idx = 1
    for v in data:
        for teen in v["text"]:
            scenes = teen.get("scenes", [])
            for sc in scenes:
                sc["kb_id"] = f"KB_{idx:06d}"
                idx += 1
    return idx - 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help="Repo root. Run this script from repo root, or pass --base_dir.",
    )
    ap.add_argument(
        "--extracted_dir",
        type=str,
        default="extracted_data",
        help="Folder containing *_extracted_data subfolders.",
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="**/*_reviewer_output.json",
        help="Glob pattern under extracted_dir.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="knowledge_base/merged_knowledge_base.json",
        help="Output path (relative to base_dir).",
    )
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    extracted_dir = (base_dir / args.extracted_dir).resolve()
    out_path = (base_dir / args.out).resolve()

    if not extracted_dir.exists():
        raise FileNotFoundError(f"extracted_dir not found: {extracted_dir}")

    fps = sorted(extracted_dir.glob(args.glob))
    if not fps:
        raise FileNotFoundError(f"No files matched: {extracted_dir}/{args.glob}")

    data: List[Dict[str, Any]] = []
    for fp in fps:
        raw = read_json(fp)
        rec = coerce_video_record(raw=raw, fp=fp, base_dir=base_dir)

        # enforce source_json to start with "extracted_data/"
        # by construction it is relative to base_dir, but we ensure it explicitly here
        if not rec["source_json"].startswith("extracted_data/"):
            # if extracted_dir is not "extracted_data", rewrite to keep compatibility with your example
            # by stripping everything before the first "extracted_data/"
            s = rec["source_json"]
            marker = "extracted_data/"
            pos = s.find(marker)
            if pos >= 0:
                rec["source_json"] = s[pos:]
            else:
                # last-resort: force a canonical-looking value
                # (this should not happen if extracted_dir default is used)
                rec["source_json"] = f"extracted_data/{rec['video_id']}_extracted_data/{rec['video_id']}_reviewer_output.json"

        data.append(rec)

    total_scenes = assign_kb_ids(data)

    merged = {
        "total_files": len(data),
        "total_scenes": total_scenes,
        "data": data,
    }

    write_json(out_path, merged)
    print(f"[OK] Wrote: {out_path}")
    print(f"[OK] total_files={merged['total_files']} total_scenes={merged['total_scenes']}")


if __name__ == "__main__":
    main()