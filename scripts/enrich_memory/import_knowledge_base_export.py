#!/usr/bin/env python3
"""Convert the author-supplied mixed-format export for the existing RAG loaders.

The source export is retained unchanged. This is format conversion, not a model
rerun, content correction, deduplication, or confirmation of historical results.
Existing scene IDs are retained when video/profile and scene content match.
"""
import argparse
from collections import defaultdict, deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SCENE_FIELDS = ('description', 'transcript_full', 'what_helped', 'other_notes')
SCENE_EXTRA = ('clinically_plausible',)

def parse_fields(text):
    matches = list(re.finditer(r'(?m)^([a-z_]+):[ \t]*', text))
    result = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        key, value = m.group(1), text[m.end():end].strip()
        if key in result:
            raise ValueError(f'Duplicate field {key!r}; refusing to overwrite it')
        result[key] = value
    return result

def parse_profiles(text):
    # Source exports end with a standalone profile-count line.
    text = re.sub(r'\n\s*\d+\s*$', '', text)
    ids = list(re.finditer(r'(?m)^id:[ \t]*', text))
    profiles = []
    for i, start in enumerate(ids):
        end = ids[i + 1].start() if i + 1 < len(ids) else len(text)
        block = text[start.start():end]
        scenes = list(re.finditer(r'(?m)^(?:scenes:[ \t]*)?description:[ \t]*', block))
        profile = parse_fields(block[:scenes[0].start()] if scenes else block)
        profile['scenes'] = []
        for j, scene_start in enumerate(scenes):
            scene_end = scenes[j + 1].start() if j + 1 < len(scenes) else len(block)
            raw = block[scene_start.start():scene_end]
            fields = parse_fields(re.sub(r'^scenes:[ \t]*', '', raw))
            if not all(key in fields for key in SCENE_FIELDS):
                raise ValueError(f'Incomplete scene for profile {profile.get("id")}')
            unexpected = set(fields) - set(SCENE_FIELDS + SCENE_EXTRA)
            if unexpected:
                raise ValueError(f'Unexpected scene fields: {unexpected}')
            fields['source_locator'] = {'text_offset': start.start() + scene_start.start()}
            profile['scenes'].append(fields)
        profiles.append(profile)
    # Detect content that a parser would otherwise silently omit.
    expected = len(re.findall(r'(?m)^transcript_full:', text))
    if sum(len(p['scenes']) for p in profiles) != expected:
        raise ValueError('Not all transcript records were parsed')
    return profiles

def identity(video, profile, scene):
    return (video['video_id'], profile.get('id'), *(scene.get(k, 'unknown').strip() for k in SCENE_FIELDS if k != 'description'))

def convert(source, previous=None):
    previous_ids = defaultdict(deque)
    max_id = 0
    if previous:
        for v in previous['data']:
            for p in v.get('text', []):
                for s in p.get('scenes', []):
                    previous_ids[identity(v, p, s)].append(s['kb_id'])
                    max_id = max(max_id, int(s['kb_id'].split('_')[1]))
    records, skipped = [], []
    for ri, raw in enumerate(source):
        if 'video_id' not in raw:
            if 'scenarios' not in raw or 'build_meta' not in raw:
                raise ValueError(f'Unrecognized source record at {ri}')
            skipped.append({'source_record_index': ri, 'reason': 'Derived scenario bundle; original video record is retained'})
            continue
        v = deepcopy(raw)
        v['source_record_index'] = ri
        if isinstance(v.get('text'), str):
            v['text'] = parse_profiles(v['text'])
        elif not isinstance(v.get('text'), list):
            raise ValueError(f'Unexpected text type at {ri}')
        for pi, p in enumerate(v['text']):
            for si, s in enumerate(p.get('scenes', [])):
                s.setdefault('source_locator', {'profile_index': pi, 'scene_index': si})
                prior = previous_ids[identity(v, p, s)]
                if prior:
                    s['kb_id'] = prior.popleft()
                else:
                    max_id += 1
                    s['kb_id'] = f'KB_{max_id:06d}'
        records.append(v)
    remaining = [x for ids in previous_ids.values() for x in ids]
    if remaining:
        raise ValueError(f'Previous scenes not matched; refusing to reassign IDs: {remaining}')
    ids = [s['kb_id'] for v in records for p in v['text'] for s in p.get('scenes', [])]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate scene IDs')
    return {'total_files': len(records), 'total_scenes': len(ids),
            'conversion': {'source': 'knowledge_base_all.json', 'deduplicated': False,
                           'description': 'Format conversion of author-confirmed source export; not an experiment rerun.',
                           'skipped_records': skipped}, 'data': records}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=ROOT / 'data/knowledge_base/knowledge_base_all.json')
    ap.add_argument('--out', type=Path, default=ROOT / 'data/knowledge_base/merged_knowledge_base.json')
    ap.add_argument('--previous', type=Path, help='Existing structured KB whose scene IDs must be retained; defaults to --out if it exists')
    args = ap.parse_args()
    prior_path = args.previous or args.out
    previous = json.loads(prior_path.read_text()) if prior_path.exists() else None
    raw = args.source.read_bytes()
    result = convert(json.loads(raw), previous)
    result['conversion']['source_sha256'] = hashlib.sha256(raw).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f"Converted {result['total_files']} videos and {result['total_scenes']} scene entries")

if __name__ == '__main__':
    main()
