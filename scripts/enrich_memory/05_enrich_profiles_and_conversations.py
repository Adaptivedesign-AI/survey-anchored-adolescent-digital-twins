#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from google import genai
from google.genai import types


REPO_ROOT = Path(__file__).resolve().parents[2]

KB_PATH = REPO_ROOT / "data" / "knowledge_base" / "merged_knowledge_base.json"
DT_CSV_PATH = REPO_ROOT / "data" / "digital_twins" / "baseline_profiles" / "1000_DTs_prompts.csv"
MATCHES_PATH = REPO_ROOT / "outputs" / "dt_memory_retrieval" / "04_matched_memories_top2.json"

OUT_DIR = REPO_ROOT / "outputs" / "dt_profile_enrichment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL_PATH = OUT_DIR / "05_enriched_profiles_with_conversations.jsonl"
OUT_META_PATH = OUT_DIR / "05_run_meta.json"

MODEL_NAME = "gemini-2.5-pro"

TOP_K = 2

DOMAINS = [
    "Injury, Violence, and Adverse Experiences",
    "Mental Health and Suicide Behaviors",
    "Substance Use (Tobacco, Alcohol, and Other Drugs)",
    "Sexual Health and Sexual Orientation",
    "Nutrition, Weight, and Health Behaviors",
]


def init_client() -> genai.Client:
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true", "yes"}
    if use_vertex:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip() or "us-central1"
        if not project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is required when GOOGLE_GENAI_USE_VERTEXAI=True")
        return genai.Client(vertexai=True, project=project, location=location)

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or set GOOGLE_GENAI_USE_VERTEXAI=True with Vertex env vars).")
    return genai.Client(api_key=api_key)


def normalize_kb_id(kb_id: str) -> str:
    if not kb_id:
        return kb_id
    kb_id = str(kb_id).strip()
    if "_" not in kb_id:
        return kb_id
    prefix, num = kb_id.split("_", 1)
    return f"{prefix}_{str(num).zfill(6)}"


def load_dt_profiles(limit: Optional[int] = None) -> pd.DataFrame:
    if not DT_CSV_PATH.exists():
        raise FileNotFoundError(f"DT CSV not found: {DT_CSV_PATH}")
    df = pd.read_csv(DT_CSV_PATH)
    if "id" not in df.columns or "prompt" not in df.columns:
        raise ValueError("DT CSV must have columns: id, prompt")
    df = df[["id", "prompt"]].copy()
    df["id"] = df["id"].astype(str)
    df["prompt"] = df["prompt"].astype(str)
    if limit is not None:
        df = df.head(int(limit))
    return df


def load_matches() -> Dict[str, Any]:
    if not MATCHES_PATH.exists():
        raise FileNotFoundError(f"Matches file not found: {MATCHES_PATH}")
    with open(MATCHES_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "meta" not in payload or "results" not in payload:
        raise ValueError("04_matched_memories_top2.json must contain keys: meta, results")
    if not isinstance(payload["results"], dict):
        raise ValueError("04_matched_memories_top2.json key 'results' must be an object mapping dt_id -> result")

    return payload


def load_kb_scene_map() -> Dict[str, Dict[str, Any]]:
    """
    Build map: kb_id (KB_000xxx) -> full scene context
    """
    if not KB_PATH.exists():
        raise FileNotFoundError(f"KB not found: {KB_PATH}")
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)

    data = kb.get("data", [])
    if not isinstance(data, list) or not data:
        raise ValueError("merged_knowledge_base.json must contain a non-empty list at key 'data'.")

    scene_map: Dict[str, Dict[str, Any]] = {}

    for entry in data:
        if not isinstance(entry, dict):
            continue
        video_id = str(entry.get("video_id", "")).strip()
        title = str(entry.get("title", "")).strip()
        identified_category = str(entry.get("identifiedCategory", "")).strip()

        teens = entry.get("text", [])
        if not isinstance(teens, list):
            continue

        for teen in teens:
            if not isinstance(teen, dict):
                continue
            teen_info = {
                "age_estimate": teen.get("age_estimate", "unknown"),
                "sex": teen.get("sex", "unknown"),
                "ethnicity": teen.get("ethnicity", "unknown"),
                "mental_health_diagnosis": teen.get("mental_health_diagnosis", "unknown"),
                "hobbies": teen.get("hobbies", "unknown"),
            }
            scenes = teen.get("scenes", [])
            if not isinstance(scenes, list):
                continue

            for sc in scenes:
                if not isinstance(sc, dict):
                    continue
                kb_id = normalize_kb_id(sc.get("kb_id", ""))
                if not kb_id or not kb_id.startswith("KB_"):
                    continue

                scene_map[kb_id] = {
                    "kb_id": kb_id,
                    "video_id": video_id,
                    "title": title,
                    "identifiedCategory": identified_category,
                    "teen_info": teen_info,
                    "scene": {
                        "kb_id": kb_id,
                        "description": str(sc.get("description", "")).strip(),
                        "transcript_full": str(sc.get("transcript_full", "")).strip(),
                        "what_helped": str(sc.get("what_helped", "")).strip(),
                        "other_notes": str(sc.get("other_notes", "")).strip(),
                    },
                }

    if not scene_map:
        raise ValueError("No scene-level kb_id found in merged_knowledge_base.json")
    return scene_map


def build_domain_enrichment_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "enhanced_profile_domain": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "overall_domain_narrative": {"type": "string"},
                    "used_kb_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["domain", "overall_domain_narrative", "used_kb_ids"],
                "additionalProperties": False,
            }
        },
        "required": ["enhanced_profile_domain"],
        "additionalProperties": False,
    }


def build_conversations_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "daily_conversations": {
                "type": "array",
                "minItems": 5,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "conversation_id": {"type": "integer"},
                        "setting": {"type": "string"},
                        "participants": {"type": "array", "items": {"type": "string"}},
                        "dialogue": {
                            "type": "array",
                            "minItems": 2,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "speaker": {"type": "string"},
                                    "text": {"type": "string"},
                                },
                                "required": ["speaker", "text"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["conversation_id", "setting", "participants", "dialogue"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["daily_conversations"],
        "additionalProperties": False,
    }


def format_memories_for_prompt(
    chosen_snippets: List[Dict[str, Any]],
    kb_scene_map: Dict[str, Dict[str, Any]],
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """
    Returns:
      - memories_text: formatted string for prompt
      - used_kb_ids: list of normalized kb_ids
      - memory_records: list for output traceability
    """
    blocks: List[str] = []
    used_ids: List[str] = []
    memory_records: List[Dict[str, Any]] = []

    for i, snip in enumerate(chosen_snippets, 1):
        kb_id_raw = str(snip.get("kb_id", "")).strip()
        kb_id = normalize_kb_id(kb_id_raw)
        used_ids.append(kb_id)

        full = kb_scene_map.get(kb_id)
        if full is None:
            excerpt = str(snip.get("text_excerpt", "")).strip()
            blocks.append(
                f"Memory {i} [{kb_id}]:\n"
                f"(no full scene found in knowledge base, using excerpt)\n"
                f"Excerpt: {excerpt}\n"
            )
            memory_records.append({
                "kb_id": kb_id,
                "found_full_scene": False,
                "excerpt": excerpt,
                "similarity": snip.get("similarity", None),
                "title": snip.get("title", ""),
                "video_id": snip.get("video_id", ""),
                "identifiedCategory": snip.get("identifiedCategory", ""),
            })
            continue

        scene = full.get("scene", {}) or {}
        teen_info = full.get("teen_info", {}) or {}

        blocks.append(
            f"Memory {i} [{kb_id}]:\n"
            f"Video title: {full.get('title','')}\n"
            f"Teen: age {teen_info.get('age_estimate','unknown')}, sex {teen_info.get('sex','unknown')}, ethnicity {teen_info.get('ethnicity','unknown')}\n"
            f"Scene description: {scene.get('description','')}\n"
            f"Transcript: {scene.get('transcript_full','')}\n"
            f"What helped: {scene.get('what_helped','')}\n"
            f"Notes: {scene.get('other_notes','')}\n"
        )

        memory_records.append({
            "kb_id": kb_id,
            "found_full_scene": True,
            "video_id": full.get("video_id", ""),
            "title": full.get("title", ""),
            "identifiedCategory": full.get("identifiedCategory", ""),
            "teen_info": teen_info,
            "scene": scene,
            "similarity": snip.get("similarity", None),
            "excerpt": str(snip.get("text_excerpt", "")).strip(),
        })

    return "\n\n".join(blocks).strip(), used_ids, memory_records


def construct_domain_prompt(baseline_profile: str, domain_name: str, memories_text: str, used_kb_ids: List[str]) -> str:
    return (
        "You are writing a teen's profile enrichment for exactly one domain.\n\n"
        "Write in second person ('you'). Write 3 to 4 sentences. It must feel realistic and specific.\n"
        "Use the baseline facts as the grounded foundation, then weave in the retrieved memories.\n"
        "Make only reasonable inferences supported by baseline and memories.\n"
        "Do not list survey items. Do not write generic advice.\n\n"
        f"DOMAIN:\n{domain_name}\n\n"
        f"BASELINE PROFILE (survey facts, second person):\n{baseline_profile}\n\n"
        f"RETRIEVED MEMORIES:\n{memories_text}\n\n"
        "Return JSON only in the schema below.\n"
        "The field used_kb_ids must be exactly the kb_id list provided.\n\n"
        "JSON schema target:\n"
        "{\n"
        '  "enhanced_profile_domain": {\n'
        '    "domain": "<same as DOMAIN>",\n'
        '    "overall_domain_narrative": "You ...",\n'
        '    "used_kb_ids": ["KB_000001", "..."]\n'
        "  }\n"
        "}\n\n"
        f"used_kb_ids to return verbatim:\n{json.dumps(used_kb_ids, ensure_ascii=False)}\n"
    )


def generate_domain_enrichment(
    client: genai.Client,
    baseline_profile: str,
    domain_name: str,
    chosen_snippets: List[Dict[str, Any]],
    kb_scene_map: Dict[str, Dict[str, Any]],
    sleep_s: float,
) -> Dict[str, Any]:
    memories_text, used_kb_ids, memory_records = format_memories_for_prompt(chosen_snippets, kb_scene_map)

    if not memories_text:
        return {
            "domain": domain_name,
            "overall_domain_narrative": "No retrieved memories available for this domain.",
            "used_kb_ids": [],
            "memories": [],
            "error": "no_memories",
        }

    prompt = construct_domain_prompt(baseline_profile, domain_name, memories_text, used_kb_ids)

    schema = build_domain_enrichment_schema()
    start = time.time()

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    elapsed = round(time.time() - start, 2)
    obj = json.loads(resp.text)

    out = obj["enhanced_profile_domain"]
    out["generation_time_seconds"] = elapsed
    out["memories"] = memory_records

    if sleep_s > 0:
        time.sleep(sleep_s)

    return out


def construct_conversations_prompt(baseline_profile: str, domain_narratives: List[Dict[str, Any]]) -> str:
    narr = []
    for d in domain_narratives:
        narr.append(f"{d.get('domain','')}: {d.get('overall_domain_narrative','')}")
    narr_text = "\n".join(narr).strip()

    return (
        "You are generating realistic daily conversations for a teen.\n\n"
        "You must follow the baseline profile and the enriched domain narratives.\n"
        "Write exactly 5 conversations.\n"
        "Each conversation must be 2 to 6 exchanges.\n"
        "Language should feel like real teen speech, natural and specific.\n"
        "Avoid therapy tone. Avoid generic motivational lines.\n"
        "Keep everything consistent with the profile.\n\n"
        f"BASELINE PROFILE:\n{baseline_profile}\n\n"
        f"ENRICHED DOMAIN NARRATIVES:\n{narr_text}\n\n"
        "Return JSON only with the following schema:\n"
        "{\n"
        '  "daily_conversations": [\n'
        "    {\n"
        '      "conversation_id": 1,\n'
        '      "setting": "...",\n'
        '      "participants": ["You", "..."],\n'
        '      "dialogue": [{"speaker":"...","text":"..."}]\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def generate_conversations(
    client: genai.Client,
    baseline_profile: str,
    domain_narratives: List[Dict[str, Any]],
    sleep_s: float,
) -> Tuple[List[Dict[str, Any]], float]:
    prompt = construct_conversations_prompt(baseline_profile, domain_narratives)
    schema = build_conversations_schema()
    start = time.time()

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.8,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    elapsed = round(time.time() - start, 2)
    obj = json.loads(resp.text)
    convs = obj.get("daily_conversations", [])

    if sleep_s > 0:
        time.sleep(sleep_s)

    return convs, elapsed


def load_existing_dt_ids(jsonl_path: Path) -> set:
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                dt_id = str(obj.get("dt_id", "")).strip()
                if dt_id:
                    done.add(dt_id)
            except Exception:
                continue
    return done


def write_jsonl_line(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_all(
    client: genai.Client,
    limit: Optional[int],
    dt_id: Optional[str],
    resume: bool,
    overwrite: bool,
    sleep_s: float,
) -> None:
    if overwrite and OUT_JSONL_PATH.exists():
        OUT_JSONL_PATH.unlink()

    dt_df = load_dt_profiles(limit=limit)
    matches_payload = load_matches()
    kb_scene_map = load_kb_scene_map()

    matches_meta = matches_payload.get("meta", {})
    matches_results = matches_payload.get("results", {})

    if dt_id is not None:
        dt_id = str(dt_id).strip()
        dt_df = dt_df[dt_df["id"].astype(str) == dt_id].copy()
        if dt_df.empty:
            raise ValueError(f"dt_id not found in CSV: {dt_id}")

    done_ids = set()
    if resume and OUT_JSONL_PATH.exists():
        done_ids = load_existing_dt_ids(OUT_JSONL_PATH)

    run_meta = {
        "step": "05_enrich_profiles_and_conversations",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "domains": DOMAINS,
        "top_k": TOP_K,
        "paths": {
            "kb_path": str(KB_PATH.relative_to(REPO_ROOT)),
            "dt_csv_path": str(DT_CSV_PATH.relative_to(REPO_ROOT)),
            "matches_path": str(MATCHES_PATH.relative_to(REPO_ROOT)),
            "out_jsonl": str(OUT_JSONL_PATH.relative_to(REPO_ROOT)),
        },
        "input_meta_from_step04": matches_meta,
        "dt_count_planned": int(len(dt_df)),
        "kb_scene_count": int(len(kb_scene_map)),
        "resume": bool(resume),
        "overwrite": bool(overwrite),
        "sleep_seconds_per_call": float(sleep_s),
    }

    with open(OUT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    for _, row in tqdm(dt_df.iterrows(), total=len(dt_df), desc="05 Enrich DTs + conversations"):
        cur_id = str(row["id"]).strip()
        baseline = str(row["prompt"]).strip()

        if resume and cur_id in done_ids:
            continue

        per_dt_match = matches_results.get(cur_id)
        if per_dt_match is None:
            rec = {
                "dt_id": cur_id,
                "timestamp": datetime.now().isoformat(),
                "model": MODEL_NAME,
                "error": "no_match_record_from_step04",
            }
            write_jsonl_line(OUT_JSONL_PATH, rec)
            continue

        domains_block = per_dt_match.get("domains", {})
        enriched_domains: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        dt_start = time.time()

        for dom in DOMAINS:
            chosen_snippets = []
            try:
                dom_obj = domains_block.get(dom, {})
                chosen_snippets = (dom_obj.get("chosen_snippets", []) or [])
                if not isinstance(chosen_snippets, list):
                    chosen_snippets = []

                enrichment = generate_domain_enrichment(
                    client=client,
                    baseline_profile=baseline,
                    domain_name=dom,
                    chosen_snippets=chosen_snippets,
                    kb_scene_map=kb_scene_map,
                    sleep_s=sleep_s,
                )
                enriched_domains.append(enrichment)
            except Exception as e:
                errors.append({
                    "domain": dom,
                    "error": str(e),
                    "chosen_snippets_count": len(chosen_snippets) if isinstance(chosen_snippets, list) else 0,
                })
                enriched_domains.append({
                    "domain": dom,
                    "overall_domain_narrative": "Error generating enrichment for this domain.",
                    "used_kb_ids": [],
                    "memories": [],
                    "error": str(e),
                })

        conversations: List[Dict[str, Any]] = []
        conv_time = 0.0
        try:
            conversations, conv_time = generate_conversations(
                client=client,
                baseline_profile=baseline,
                domain_narratives=enriched_domains,
                sleep_s=sleep_s,
            )
        except Exception as e:
            errors.append({"stage": "conversations", "error": str(e)})

        total_time = round(time.time() - dt_start, 2)

        out_obj = {
            "dt_id": cur_id,
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "inputs": {
                "baseline_profile": baseline,
                "matches_source": str(MATCHES_PATH.relative_to(REPO_ROOT)),
            },
            "domains": enriched_domains,
            "daily_conversations": conversations,
            "timing": {
                "conversation_generation_time_seconds": conv_time,
                "total_dt_time_seconds": total_time,
            },
        }

        if errors:
            out_obj["errors"] = errors

        write_jsonl_line(OUT_JSONL_PATH, out_obj)

    print(f"[OK] Wrote JSONL: {OUT_JSONL_PATH.relative_to(REPO_ROOT)}")
    print(f"[OK] Wrote meta:  {OUT_META_PATH.relative_to(REPO_ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dt_id", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    client = init_client()
    run_all(
        client=client,
        limit=args.limit,
        dt_id=args.dt_id,
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
        sleep_s=float(args.sleep),
    )


if __name__ == "__main__":
    main()