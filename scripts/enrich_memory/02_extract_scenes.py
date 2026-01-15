
"""
Step 02: Extract structured micro-scenes from YouTube videos using Gemini,
with category-specific prompts + JSON schemas.

Inputs (relative to repo root, configurable in configs/paths.yaml):
- data/sources/youtube_corpus/api_metadata/{youtube_video_id}.json
- (optional) local video file (downloaded in a private cache, not committed)

Prompts & schemas (relative to repo root, configurable in configs/paths.yaml):
- prompts live under:   data/sources/youtube_corpus/prompts/
- schemas live under:   data/sources/youtube_corpus/schemas/

Outputs:
- data/sources/youtube_corpus/extracted_scenes/{youtube_video_id}/
    - {youtube_video_id}_gemini_output.json
    - {youtube_video_id}_reviewer_output.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import yaml
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from google import genai
from google.genai import types


# ---------------------------
# Category routing
# ---------------------------
VALID_CATEGORIES = [
    "Interviews",
    "Documentary & Awareness",
    "Personal Storytelling",
    "News & Reporting",
    "Research & Investigative",
    "Creative (Drama, TV)",
]

FILEMAP = {
    "Interviews": {
        "prompt": "Interviews.prompt.md",
        "schema": "Interviews.schema.json",
    },
    "Documentary & Awareness": {
        "prompt": "Documentary_Awareness.prompt.md",
        "schema": "Documentary_Awareness.schema.json",
    },
    "Personal Storytelling": {
        "prompt": "Personal_Storytelling.prompt.md",
        "schema": "Personal_Storytelling.schema.json",
    },
    "News & Reporting": {
        "prompt": "News_Reporting.prompt.md",
        "schema": "News_Reporting.schema.json",
    },
    "Research & Investigative": {
        "prompt": "Research_Investigative.prompt.md",
        "schema": "Research_Investigative.schema.json",
    },
    "Creative (Drama, TV)": {
        "prompt": "Creative_Drama_TV.prompt.md",
        "schema": "Creative_Drama_TV.schema.json",
    },
}


# ---------------------------
# Helpers
# ---------------------------
def repo_root() -> Path:
    # src/pipelines/02_extract_scenes.py -> repo root = parents[2]
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_configs() -> tuple[dict, dict]:
    root = repo_root()
    paths_cfg = load_yaml(root / "configs" / "paths.yaml")
    models_cfg = load_yaml(root / "configs" / "models.yaml")
    return paths_cfg, models_cfg


def resolve_metadata_path(video_id: str, paths_cfg: dict) -> Path:
    root = repo_root()
    meta_dir = root / paths_cfg["youtube"]["api_metadata_dir"]
    # in Step 01 we saved per-video metadata as {vid}.json
    return meta_dir / f"{video_id}.json"


def resolve_prompts_and_schemas(paths_cfg: dict, category: str) -> Tuple[Path, Path, str]:
    root = repo_root()
    prompts_dir = root / paths_cfg["youtube"]["prompts_dir"]
    schemas_dir = root / paths_cfg["youtube"]["schemas_dir"]

    resolved = category if category in FILEMAP else "Personal Storytelling"
    if resolved != category:
        print(f"[WARN] Unknown category '{category}', fallback to '{resolved}'")

    prompt_name = FILEMAP[resolved]["prompt"]
    schema_name = FILEMAP[resolved]["schema"]

    prompt_path = prompts_dir / prompt_name
    schema_path = schemas_dir / schema_name

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    return prompt_path, schema_path, resolved


def resolve_output_dir(video_id: str, paths_cfg: dict) -> Path:
    root = repo_root()
    out_root = root / paths_cfg["youtube"]["extracted_scenes_dir"]
    return ensure_dir(out_root / video_id)


# ---------------------------
# Gemini client + upload
# ---------------------------
def init_gemini_client(models_cfg: dict) -> genai.Client:
    """
    Supports two modes via configs/models.yaml:

    gemini:
      provider: ai_studio | vertex
      api_key_env: GEMINI_API_KEY          # for ai_studio
      project_env: GOOGLE_CLOUD_PROJECT    # for vertex
      location_env: GOOGLE_CLOUD_LOCATION  # for vertex
      model_extract: gemini-2.5-flash
    """
    gcfg = models_cfg.get("gemini", {})
    provider = (gcfg.get("provider") or "ai_studio").strip().lower()

    if provider == "vertex":
        project_env = gcfg.get("project_env", "GOOGLE_CLOUD_PROJECT")
        location_env = gcfg.get("location_env", "GOOGLE_CLOUD_LOCATION")
        project = os.environ.get(project_env)
        location = os.environ.get(location_env, "us-central1")
        if not project:
            raise RuntimeError(f"Missing env var {project_env} for Vertex mode.")
        return genai.Client(vertexai=True, project=project, location=location)

    # default: AI Studio key
    api_key_env = gcfg.get("api_key_env", "GEMINI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing env var {api_key_env} for AI Studio mode.")
    return genai.Client(api_key=api_key)


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((TimeoutError, RuntimeError)),
)
def wait_until_active(client: genai.Client, file_obj, *, timeout: int = 300, poll: int = 5):
    start = time.time()
    name = file_obj.name
    while True:
        f = client.files.get(name=name)
        st = getattr(f.state, "name", None) or str(f.state)
        if st == "ACTIVE":
            return f
        if st == "FAILED":
            raise RuntimeError(f"File processing failed: {f}")
        if time.time() - start > timeout:
            raise TimeoutError(f"File stuck in {st} after {timeout}s: {name}")
        time.sleep(poll)


def gemini_extract(
    client: genai.Client,
    *,
    video_id: str,
    metadata: dict,
    prompt_text: str,
    schema_obj: dict,
    model_name: str,
    video_path: Optional[Path] = None,
) -> dict:
    """
    If video_path is provided, upload the video and let Gemini extract from video content.
    If video_path is None, fall back to metadata-only extraction (still deterministic but weaker).
    """
    contents = [
        prompt_text,
        "JSON metadata:\n\n" + json.dumps(metadata, ensure_ascii=False),
    ]

    if video_path is not None:
        uploaded = client.files.upload(file=str(video_path))
        file_active = wait_until_active(client, uploaded)
        contents.append(file_active)
    else:
        contents.append(
            "NOTE: No local video file provided. Extract using metadata only."
        )

    resp = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema_obj,
        ),
    )

    # resp.text is JSON string
    try:
        return json.loads(resp.text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Gemini returned invalid JSON for video {video_id}: {e}\n{resp.text[:2000]}") from e


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Step 02: Extract micro-scenes with Gemini")
    parser.add_argument("--video_id", type=str, required=True, help="YouTube video id (11 chars)")
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Override metadata json path (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Optional local video file path (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs for this video id.",
    )
    args = parser.parse_args()

    paths_cfg, models_cfg = load_configs()
    root = repo_root()

    # Resolve metadata path
    if args.metadata_path:
        meta_path = Path(args.metadata_path)
        meta_path = meta_path if meta_path.is_absolute() else (root / meta_path)
    else:
        meta_path = resolve_metadata_path(args.video_id, paths_cfg)

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    metadata = read_json(meta_path)
    category = metadata.get("identifiedCategory")
    if not category:
        raise ValueError(f"identifiedCategory missing in metadata: {meta_path}")

    prompt_path, schema_path, resolved_category = resolve_prompts_and_schemas(paths_cfg, category)
    prompt_text = read_text(prompt_path)
    schema_obj = read_json(schema_path)

    out_dir = resolve_output_dir(args.video_id, paths_cfg)
    gemini_out = out_dir / f"{args.video_id}_gemini_output.json"
    reviewer_out = out_dir / f"{args.video_id}_reviewer_output.json"
    scenes_out = out_dir / "scenes.jsonl"

    if gemini_out.exists() and not args.force:
        print(f"[SKIP] Output already exists. Use --force to overwrite: {gemini_out}")
        return

    video_path: Optional[Path] = None
    if args.video_path:
        vp = Path(args.video_path)
        vp = vp if vp.is_absolute() else (root / vp)
        if not vp.exists():
            raise FileNotFoundError(f"Video file not found: {vp}")
        video_path = vp

    client = init_gemini_client(models_cfg)
    model_name = models_cfg.get("gemini", {}).get("model_extract", "gemini-2.5-flash")

    print(f"[INFO] video_id: {args.video_id}")
    print(f"[INFO] category: {resolved_category}")
    print(f"[INFO] prompt: {prompt_path}")
    print(f"[INFO] schema: {schema_path}")
    print(f"[INFO] model: {model_name}")
    print(f"[INFO] metadata: {meta_path}")
    if video_path:
        print(f"[INFO] video: {video_path}")
    else:
        print(f"[INFO] video: (none)")

    data = gemini_extract(
        client,
        video_id=args.video_id,
        metadata=metadata,
        prompt_text=prompt_text,
        schema_obj=schema_obj,
        model_name=model_name,
        video_path=video_path,
    )

    write_json(gemini_out, data)
    write_json(reviewer_out, data)

    # Adjust the key name below to match your schema (e.g., "scenes", "micro_scenes", etc.)
    scenes = None
    for k in ["scenes", "micro_scenes", "chunks"]:
        if isinstance(data.get(k), list):
            scenes = data[k]
            break

    if scenes is not None:
        # Add video_id to each scene for downstream joins
        rows = []
        for i, s in enumerate(scenes):
            if isinstance(s, dict):
                s2 = dict(s)
                s2.setdefault("youtube_video_id", args.video_id)
                s2.setdefault("scene_index", i)
                rows.append(s2)
        if rows:
            write_jsonl(scenes_out, rows)

    print(f"[OK] Wrote Gemini output:   {gemini_out}")
    print(f"[OK] Wrote reviewer file:  {reviewer_out}")
    if scenes_out.exists():
        print(f"[OK] Wrote scenes JSONL:  {scenes_out}")


if __name__ == "__main__":
    main()