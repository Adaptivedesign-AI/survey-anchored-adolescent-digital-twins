#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import math
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd
from tqdm import tqdm

from google import genai
from google.genai import types


REPO_ROOT = Path(__file__).resolve().parents[2]

KB_PATH = REPO_ROOT / "data" / "knowledge_base" / "merged_knowledge_base.json"
DT_CSV_PATH = REPO_ROOT / "data" / "digital_twins" / "baseline_profiles" / "1000_DTs_prompts.csv"

OUT_DIR = REPO_ROOT / "outputs" / "dt_memory_retrieval"
CACHE_DIR = REPO_ROOT / "outputs" / "kb_cache"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "04_matched_memories_top2.json"

KB_DOMAIN_CACHE_PATH = CACHE_DIR / "kb_domain_labels.jsonl"
KB_EMBED_CACHE_PATH = CACHE_DIR / "kb_embeddings.jsonl"

MODEL_CLASSIFY = "gemini-2.5-flash"
EMBED_MODEL = "text-embedding-004"

TOP_K = 2

DOMAINS = [
    "Injury, Violence, and Adverse Experiences",
    "Mental Health and Suicide Behaviors",
    "Substance Use (Tobacco, Alcohol, and Other Drugs)",
    "Sexual Health and Sexual Orientation",
    "Nutrition, Weight, and Health Behaviors",
]


def _cosine(u: List[float], v: List[float]) -> float:
    if not u or not v or len(u) != len(v):
        return -1.0
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for a, b in zip(u, v):
        dot += a * b
        nu += a * a
        nv += b * b
    if nu <= 0.0 or nv <= 0.0:
        return -1.0
    return dot / (math.sqrt(nu) * math.sqrt(nv))


def normalize_kb_id(kb_id: str) -> str:
    if not kb_id:
        return kb_id
    kb_id = str(kb_id).strip()
    if "_" not in kb_id:
        return kb_id
    prefix, num = kb_id.split("_", 1)
    return f"{prefix}_{str(num).zfill(6)}"


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


def load_kb_items() -> List[Dict[str, Any]]:
    """
    把 merged_knowledge_base.json 展平成 scene-level items
    kb_id 使用 scene.kb_id（KB_000xxx），保证后续 enrichment 能回查 full content
    """
    if not KB_PATH.exists():
        raise FileNotFoundError(f"KB not found: {KB_PATH}")
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)

    data = kb.get("data", [])
    if not isinstance(data, list) or not data:
        raise ValueError("merged_knowledge_base.json must contain a non-empty list at key 'data'.")

    items: List[Dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue

        video_id = str(entry.get("video_id", "")).strip()
        source_json = str(entry.get("source_json", "")).strip()
        title = str(entry.get("title", "")).strip()
        identified_category = str(entry.get("identifiedCategory", "")).strip()

        teens = entry.get("text", [])
        if not isinstance(teens, list):
            continue

        for teen in teens:
            if not isinstance(teen, dict):
                continue
            scenes = teen.get("scenes", [])
            if not isinstance(scenes, list):
                continue

            teen_info = {
                "age_estimate": teen.get("age_estimate", "unknown"),
                "sex": teen.get("sex", "unknown"),
                "ethnicity": teen.get("ethnicity", "unknown"),
                "mental_health_diagnosis": teen.get("mental_health_diagnosis", "unknown"),
                "hobbies": teen.get("hobbies", "unknown"),
            }

            for sc in scenes:
                if not isinstance(sc, dict):
                    continue
                kb_id = normalize_kb_id(sc.get("kb_id", ""))
                if not kb_id or not kb_id.startswith("KB_"):
                    continue

                items.append({
                    "kb_id": kb_id,
                    "video_id": video_id,
                    "source_json": source_json,
                    "identifiedCategory": identified_category,
                    "title": title,
                    "teen_info": teen_info,
                    "scene": {
                        "kb_id": kb_id,
                        "description": str(sc.get("description", "")).strip(),
                        "transcript_full": str(sc.get("transcript_full", "")).strip(),
                        "what_helped": str(sc.get("what_helped", "")).strip(),
                        "other_notes": str(sc.get("other_notes", "")).strip(),
                    }
                })

    if not items:
        raise ValueError("No valid scene-level KB items parsed from merged_knowledge_base.json")
    return items


def read_jsonl_as_dict(path: Path, key_field: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            k = str(obj.get(key_field, "")).strip()
            if k:
                out[k] = obj
    return out


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_domain_classifier_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {"domain": {"type": "string", "enum": DOMAINS}},
        "required": ["domain"],
        "additionalProperties": False,
    }


def classify_domain(client: genai.Client, kb_item: Dict[str, Any]) -> str:
    schema = build_domain_classifier_schema()
    sc = kb_item.get("scene", {}) or {}
    prompt = (
        "You are labeling a youth mental-health related scene into exactly one domain.\n\n"
        "Choose exactly one domain from this list:\n"
        f"{json.dumps(DOMAINS, ensure_ascii=False)}\n\n"
        "Return JSON only with key 'domain'.\n\n"
        "SCENE:\n"
        f"Video title: {kb_item.get('title','')}\n"
        f"Scene description: {sc.get('description','')}\n"
        f"Transcript: {str(sc.get('transcript_full',''))[:3500]}\n"
    )

    resp = client.models.generate_content(
        model=MODEL_CLASSIFY,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    obj = json.loads(resp.text)
    dom = str(obj["domain"]).strip()
    if dom not in DOMAINS:
        raise ValueError(f"Invalid domain returned: {dom}")
    return dom


def embed_text(client: genai.Client, text: str) -> List[float]:
    text = text or ""
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
    )

    emb = None
    if hasattr(resp, "embeddings") and resp.embeddings:
        e0 = resp.embeddings[0]
        if hasattr(e0, "values"):
            emb = list(e0.values)
        elif isinstance(e0, dict) and "values" in e0:
            emb = list(e0["values"])

    if emb is None and isinstance(resp, dict):
        if "embeddings" in resp and resp["embeddings"]:
            e0 = resp["embeddings"][0]
            if isinstance(e0, dict) and "values" in e0:
                emb = list(e0["values"])

    if emb is None or not isinstance(emb, list) or not emb:
        raise RuntimeError("Failed to parse embedding response.")
    return emb


def ensure_kb_cache(client: genai.Client, kb_items: List[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, List[float]]]:
    domain_cache = read_jsonl_as_dict(KB_DOMAIN_CACHE_PATH, "kb_id")
    embed_cache = read_jsonl_as_dict(KB_EMBED_CACHE_PATH, "kb_id")

    kb_id_to_domain: Dict[str, str] = {}
    kb_id_to_emb: Dict[str, List[float]] = {}

    for kb_id, obj in domain_cache.items():
        dom = str(obj.get("domain", "")).strip()
        if dom in DOMAINS:
            kb_id_to_domain[kb_id] = dom

    for kb_id, obj in embed_cache.items():
        vec = obj.get("embedding", None)
        if isinstance(vec, list) and vec:
            kb_id_to_emb[kb_id] = vec

    for it in tqdm(kb_items, desc="Cache KB scenes (domain + embeddings)"):
        kb_id = it["kb_id"]

        if kb_id not in kb_id_to_domain:
            dom = classify_domain(client, it)
            kb_id_to_domain[kb_id] = dom
            append_jsonl(KB_DOMAIN_CACHE_PATH, {"kb_id": kb_id, "domain": dom})
            time.sleep(0.05)

        if kb_id not in kb_id_to_emb:
            sc = it.get("scene", {}) or {}
            teen = it.get("teen_info", {}) or {}
            text_for_embed = " | ".join([
                it.get("title", "").strip(),
                str(sc.get("description", "")).strip(),
                str(sc.get("transcript_full", "")).strip(),
                str(sc.get("what_helped", "")).strip(),
                str(sc.get("other_notes", "")).strip(),
                f"Teen: age {teen.get('age_estimate','unknown')}, sex {teen.get('sex','unknown')}, ethnicity {teen.get('ethnicity','unknown')}",
            ]).strip(" | ")

            vec = embed_text(client, text_for_embed[:12000])
            kb_id_to_emb[kb_id] = vec
            append_jsonl(KB_EMBED_CACHE_PATH, {"kb_id": kb_id, "embed_model": EMBED_MODEL, "embedding": vec})
            time.sleep(0.02)

    return kb_id_to_domain, kb_id_to_emb


def build_domain_index(kb_items: List[Dict[str, Any]], kb_id_to_domain: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    dom2items: Dict[str, List[Dict[str, Any]]] = {d: [] for d in DOMAINS}
    for it in kb_items:
        dom = kb_id_to_domain.get(it["kb_id"], None)
        if dom in dom2items:
            dom2items[dom].append(it)
    return dom2items


def retrieve_topk_for_domain(
    client: genai.Client,
    domain_items: List[Dict[str, Any]],
    kb_id_to_emb: Dict[str, List[float]],
    query_text: str,
    k: int,
) -> List[Dict[str, Any]]:
    qvec = embed_text(client, query_text[:12000])
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for it in domain_items:
        kb_id = it["kb_id"]
        vec = kb_id_to_emb.get(kb_id)
        if not vec:
            continue
        sim = _cosine(qvec, vec)
        scored.append((sim, it))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]

    out = []
    for sim, it in top:
        sc = it.get("scene", {}) or {}
        excerpt = str(sc.get("transcript_full", "")).strip()
        if not excerpt:
            excerpt = str(sc.get("description", "")).strip()
        excerpt = excerpt[:800]
        out.append({
            "kb_id": it["kb_id"],
            "video_id": it.get("video_id", ""),
            "identifiedCategory": it.get("identifiedCategory", ""),
            "title": it.get("title", ""),
            "text_excerpt": excerpt,
            "similarity": sim,
        })
    return out


def run_all(client: genai.Client, limit: Optional[int]) -> None:
    kb_items = load_kb_items()
    dt_df = load_dt_profiles(limit=limit)

    kb_id_to_domain, kb_id_to_emb = ensure_kb_cache(client, kb_items)
    dom2items = build_domain_index(kb_items, kb_id_to_domain)

    meta = {
        "model_embed": EMBED_MODEL,
        "model_domain_classifier": MODEL_CLASSIFY,
        "top_k": TOP_K,
        "domains": DOMAINS,
        "kb_path": str(KB_PATH.relative_to(REPO_ROOT)),
        "dt_csv_path": str(DT_CSV_PATH.relative_to(REPO_ROOT)),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dt_count": int(len(dt_df)),
        "kb_scene_count": int(len(kb_items)),
    }

    results: Dict[str, Any] = {}

    for _, row in tqdm(dt_df.iterrows(), total=len(dt_df), desc="Retrieve top-2 per DT per domain"):
        dt_id = str(row["id"]).strip()
        baseline = str(row["prompt"]).strip()

        per_dt = {"domains": {}}

        for dom in DOMAINS:
            query = (
                f"Domain: {dom}\n"
                "Teen baseline profile (survey facts in second person):\n"
                f"{baseline}\n"
                "Task: retrieve the most relevant real teen experiences for this domain."
            )

            chosen = retrieve_topk_for_domain(
                client=client,
                domain_items=dom2items.get(dom, []),
                kb_id_to_emb=kb_id_to_emb,
                query_text=query,
                k=TOP_K,
            )

            per_dt["domains"][dom] = {"chosen_snippets": chosen}

        results[dt_id] = per_dt

    payload = {"meta": meta, "results": results}

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote: {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"[OK] KB caches at: {CACHE_DIR.relative_to(REPO_ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    client = init_client()
    run_all(client, limit=args.limit)


if __name__ == "__main__":
    main()