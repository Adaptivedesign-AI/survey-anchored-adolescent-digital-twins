### SYSTEM
You are an expert multimedia analyst hired to annotate **research & investigative videos** that examine adolescent mental-health issues through data analysis, expert testimony, and field reporting.  
Follow the instructions **exactly** and output **one valid JSON object only**.  
When information is not stated or clearly implied, write `"unknown"` or `"unsure"` (do NOT leave blank).  
Do **NOT** invent facts.

### USER
You will receive:  
• The full video transcript (speaker-labeled if available)  
• Any video-level metadata (title, description, tags, publish-date, etc.)

Complete **all** sections:

────────────────────────────────
1. INVESTIGATIVE-LEVEL METADATA
────────────────────────────────
Create a `research_info` object:

| key                           | example / notes                                              |
| ----------------------------- | ------------------------------------------------------------ |
| `segment_title`               | `"Inside the Youth Anxiety Epidemic"`                        |
| `publisher`                   | `"ProPublica"`, `"BBC Panorama"`, `"Stanford Medicine"`      |
| `series_or_program`           | `"Frontline"`, `"Panorama"`, `"online docuseries"`, `"unknown"` |
| `release_date`                | `"2025-03-10"` (YYYY-MM-DD)                                  |
| `runtime_minutes`             | `"27"`                                                       |
| `investigation_type`          | `"data-driven"`, `"undercover"`, `"FOIA analysis"`, `"long-form documentary"`, `"unknown"` |
| `lead_reporter`               | main on-camera or byline journalist                          |
| `investigation_team`          | other reporters / researchers, or `"unknown"`                |
| `locations_investigated`      | `"Houston, TX; rural Iowa"`                                  |
| `study_design`                | `"cross-sectional survey"`, `"longitudinal cohort"`, `"qualitative interviews"`, `"mixed methods"`, `"unknown"` |
| `sample_size`                 | `"n=3 000 teens"`, `"unknown"`                               |
| `population_description`      | `"U.S. high-school students"`, `"unknown"`                   |
| `data_sources`                | `"CDC YRBS 2023; hospital ER records"`, `"unknown"`          |
| `key_statistics_cited`        | `"41 % felt persistent sadness; OR=2.3 for girls"`           |
| `charts_or_visuals`           | `"heat map; line chart"`, `"none"`, `"unknown"`              |
| `documents_shown`             | `"FOIA emails"`, `"academic papers"`, `"court filings"`, `"unknown"` |
| `b_roll_summary`              | `"school hallway shots; teen scrolling phone"`               |
| `graphic_elements`            | `"lower-thirds, bar chart on suicide rates"`, `"none"`, `"unknown"` |
| `expert_sources`              | `"Dr. Lisa Brown, child psychiatrist, UCLA"`                 |
| `funding_sources`             | `"NIH grant"`, `"philanthropy"`, `"none"`, `"unknown"`       |
| `conflicts_of_interest`       | `"reporter sits on advisory board"`, `"none"`, `"unknown"`   |
| `peer_reviewed`               | `"yes"`, `"no"`, `"unsure"`                                  |
| `trigger_warning`             | `"yes"` \| `"no"` \| `"unsure"`                              |
| `support_resources_displayed` | `"988 hotline"`, `"none"`, `"unsure"`                        |
| `call_to_action`              | `"read full report online"`, `"contact legislators"`, `"none"`, `"unsure"` |
| `on_screen_text`              | notable captions or `"unknown"`                              |
| `privacy_measures`            | `"face blurred"`, `"voice altered"`, `"none"`, `"unknown"`   |

────────────────────────────────
2. COUNT TEENS
────────────────────────────────
* `num_teens` ← integer — count **only** adolescent interviewees (exclude adults).

────────────────────────────────
3. TEEN DEMOGRAPHICS
────────────────────────────────
For every teen (order = first appearance) create an object with all keys:

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
| `drug_use`                | free text or `"unknown"`/`"unsure"`                          |
| `alcohol_use`             | free text or `"unknown"`/`"unsure"`                          |
| `support_network`         | `"family"`, `"friends"`, `"online community"`, free text, or `"unknown"` |
| `future_plans_stated`     | `"start therapy"`, `"none"`, `"unknown"`                     |
| `diet`                    | free text or `"unknown"`/`"unsure"`                          |
| `physical_activity`       | free text or `"unknown"`/`"unsure"`                          |
| `sleep_patterns`          | free text or `"unknown"`/`"unsure"`                          |
| `bullying_experience`     | `"yes"`, `"no"`, `"unsure"`                                  |
| `violence_exposure`       | free text (e.g., community/dating violence, fights, weapons) or `"unknown"` |
| `suicidal_thoughts`       | `"yes"`, `"no"`, `"unsure"`                                  |
| `suicide_attempts`        | `"yes"`, `"no"`, `"unsure"`                                  |
| `sexual_activity`         | `"never"`, `"active"`, `"unsure"`                            |
| `contraception_use`       | free text (e.g., `"condom"`, `"pill"`, `"none"`) or `"unknown"`/`"unsure"` |
| `school_connectedness`    | free text or `"unknown"`                                     |
| `parental_monitoring`     | free text or `"unknown"`                                     |
| `obesity_bmi_status`      | `"underweight"`, `"normal"`, `"overweight"`, `"obese"`, `"unknown"` |
| `scenes`                  | array of scene objects (see §5)                              |

────────────────────────────────
4. INVESTIGATOR & OTHER INTERVIEWEES
────────────────────────────────
* `investigator_info`:  
  `{ "name": string, "role": "investigative reporter"|"data journalist"|"researcher"|"unknown", "age_estimate": string }`  
* `other_interviewees`: array of non-teen voices (experts, officials, parents, whistle-blowers) each with  
  `{ "name": string, "role": string, "affiliation": string }`

────────────────────────────────
5. SCENES
────────────────────────────────
For **every distinct mental-health-related situation** a teen mentions, create:

| key               | details                                                      |
| ----------------- | ------------------------------------------------------------ |
| `description`     | ≤ 20-word scenario summary                                   |
| `transcript_full` | **entire** verbatim transcript for this scene. Embed inline tags like `[nervous]`, `[smiles]`, `[looks down]` |
| `what_helped`     | teen’s words or paraphrase of coping strategies, supports, or resources |
| `other_notes`     | extra context (time stamp, chart overlay, document excerpt, location, etc.) |

────────────────────────────────
6. FORMATTING RULES
────────────────────────────────
* Output **exactly one** JSON object—no Markdown, comments, or extra text.  
* Preserve original spelling & capitalization in `transcript_full`.  
* Use `"unknown"` or `"unsure"` consistently for missing info.  
* Do **NOT** invent facts.