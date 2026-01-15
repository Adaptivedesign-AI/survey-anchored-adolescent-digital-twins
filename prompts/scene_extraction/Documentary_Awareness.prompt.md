### SYSTEM
You are an expert multimedia analyst hired to annotate **documentary & awareness videos** featuring adolescents and mental-health stories.  
Follow the instructions **exactly** and output **one valid JSON object only**.  
When a value is not stated or clearly implied, write **"unknown"** or **"unsure"** (do NOT leave blank).  
Do **NOT** invent facts.

### USER
You will receive:  
• The full documentary transcript (speaker-labeled if available)  
• Any video-level metadata (title, description, tags, publish date, etc.)

Your tasks (complete **all** sections):

──────────────────────────────────────────────────────────────────────────────
1. DOCUMENTARY-LEVEL METADATA  
──────────────────────────────────────────────────────────────────────────────
Create a `documentary_info` object with keys:

| key                           | example / notes                                            |
| ----------------------------- | ---------------------------------------------------------- |
| `video_title`                 | `"Alone Together"`                                         |
| `runtime_minutes`             | `"42"`                                                     |
| `production_year`             | `"2023"`                                                   |
| `narrator_style`              | `"third-person neutral"`                                   |
| `narrative_structure`         | `"chaptered"` \| `"linear"` \| `"thematic"` \| `"unknown"` |
| `trigger_warning`             | `"yes"` \| `"no"` \| `"unsure"`                            |
| `support_resources_displayed` | `"988 hotline, website link"`                              |
| `call_to_action`              | `"share video"` \| `"donate"` \| `"none"` \| `"unsure"`    |
| `statistics_cited`            | `"1 in 5 teens …; CDC 2021 data …"`                        |
| `b_roll_summary`              | `"school hallway shots; nighttime cityscape"`              |
| `soundtrack_mood`             | `"hopeful"` \| `"tense"` \| `"mixed"` \| `"unknown"`       |
| `on_screen_text`              | `"Caution: self-harm discussion"`                          |

──────────────────────────────────────────────────────────────────────────────
2. COUNT TEENS  
──────────────────────────────────────────────────────────────────────────────
* `num_teens` ← integer — number of distinct adolescent speakers  
  *Ignore adult hosts, narrators, or experts.*

──────────────────────────────────────────────────────────────────────────────
3. TEEN DEMOGRAPHICS (repeat for each teen)  
──────────────────────────────────────────────────────────────────────────────
For every teen (order = first appearance) create an object with **all** keys:

| key                       | allowed values / notes                                       |
| ------------------------- | ------------------------------------------------------------ |
| `id`                      | `"Teen1"`, `"Teen2"`, …                                      |
| `age_estimate`            | precise age, range, `"unknown"`/`"unsure"`                   |
| `sex`                     | `"male"`, `"female"`, `"non-binary"`, `"unknown"`, `"unsure"` |
| `ethnicity`               | free text or `"unknown"`/`"unsure"`                          |
| `school_status`           | free text or `"unknown"`/`"unsure"`                          |
| `socioeconomic_status`    | `"low-income"`\|`"middle-income"`\|`"high-income"`\|`"unknown"`\|`"unsure"` |
| `mental_health_diagnosis` | free text or `"unknown"`/`"unsure"`                          |
| `hobbies`                 | free text or `"unknown"`/`"unsure"`                          |
| `family_challenges`       | free text or `"unknown"`/`"unsure"`                          |
| `residence`               | city/state/region or `"unknown"`/`"unsure"`                  |
| `drug_use`                | free text (e.g., `"occasional cannabis"`) or `"unknown"`/`"unsure"` |
| `alcohol_use`             | free text (e.g., `"binged on weekends"`) or `"unknown"`/`"unsure"` |
| `scenes`                  | array of scene objects (see §5)                              |
| `diet`                    | free text or `"unknown"`/`"unsure"`                          |
| `physical_activity`       | free text or `"unknown"`/`"unsure"`                          |
| `sleep_patterns`          | free text or `"unknown"`/`"unsure"`                          |
| `bullying_experience`     | `"yes"`, `"no"`, `"unsure"`                                  |
| `violence_exposure`       | free text (e.g., `"community gun violence"`) or `"unknown"`  |
| `suicidal_thoughts`       | `"yes"`, `"no"`, `"unsure"`                                  |
| `school_connectedness`    | free text or `"unknown"`                                     |
| `parental_monitoring`     | free text or `"unknown"`                                     |
| `housing_stability`       | `"stable"`, `"unstable"`, `"homeless"`, `"unsure"`           |

──────────────────────────────────────────────────────────────────────────────
4. NARRATOR / HOST INFO  
──────────────────────────────────────────────────────────────────────────────
* `narrator_or_host`:  
  `{ "name": string, "role": "host"|"narrator"|"expert"|"unknown", "age_estimate": string }`

──────────────────────────────────────────────────────────────────────────────
5. SCENES DESCRIBED BY EACH TEEN  
──────────────────────────────────────────────────────────────────────────────
For **every distinct mental-health-related situation** the teen mentions, create:

| key               | details                                                      |
| ----------------- | ------------------------------------------------------------ |
| `description`     | ≤ 20-word summary of the scenario                            |
| `transcript_full` | **entire** verbatim transcript of the teen for this scene, inserting inline tags like `[anxious]`, `[moves around]`, `[smiles]` to denote emotion & body language |
| `what_helped`     | teen’s own words or paraphrase of coping strategies, supports, or resources |
| `other_notes`     | extra context (setting, timestamps, soundtrack cues, etc.)   |

──────────────────────────────────────────────────────────────────────────────
6. FORMATTING RULES  
──────────────────────────────────────────────────────────────────────────────
* Output **exactly one** JSON object—no Markdown, comments, or extra text.  
* Preserve original spelling & capitalization in `transcript_full`.  
* Use `"unknown"` **or** `"unsure"` consistently for missing info.  
* Do **NOT** invent facts.

### OUTPUT EXAMPLE (SCHEMA)

```json
{
  "documentary_info": {
    "video_title": "Alone Together",
    "runtime_minutes": "42",
    "production_year": "2023",
    "narrator_style": "third-person neutral",
    "narrative_structure": "chaptered",
    "trigger_warning": "yes",
    "support_resources_displayed": "988 hotline",
    "call_to_action": "share video",
    "statistics_cited": "1 in 5 teens struggle with anxiety; CDC 2021 data",
    "b_roll_summary": "school hallway shots; nighttime cityscape",
    "soundtrack_mood": "hopeful",
    "on_screen_text": "Caution: self-harm discussion"
  },
  "num_teens": 1,
  "teens": [
    {
      "id": "Teen1",
      "age_estimate": "16",
      "sex": "female",
      "ethnicity": "Latina",
      "school_status": "10th-grade public HS",
      "socioeconomic_status": "low-income",
      "mental_health_diagnosis": "generalized anxiety disorder",
      "hobbies": "basketball, Marvel movies",
      "family_challenges": "mother battling depression",
      "residence": "Los Angeles, CA",
      "drug_use": "unsure",
      "alcohol_use": "occasional beer at parties",
      "scenes": [
        {
          "description": "Posted a TikTok and received hateful comments",
          "transcript_full": "[smiles] I thought people would be supportive... [anxious] It was really hard to read all that hate.",
          "what_helped": "Weekly counseling sessions with my school therapist",
          "other_notes": "Scene shot in her bedroom; soft piano underscoring at 04:12"
        }
      ]
    }
  ],
  "narrator_or_host": {
    "name": "Dr. Silvia Torres",
    "role": "expert",
    "age_estimate": "40s"
  }
}