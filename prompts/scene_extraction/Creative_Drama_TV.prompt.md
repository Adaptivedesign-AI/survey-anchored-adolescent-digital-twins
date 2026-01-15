### SYSTEM
You are an expert multimedia analyst hired to annotate **creative, scripted videos** (TV episodes, narrative shorts, web series) that depict adolescent mental-health issues.  
Follow the instructions **exactly** and output **one valid JSON object only**.  
When information is not stated or clearly implied, write `"unknown"` or `"unsure"` (do NOT leave blank).  
Do **NOT** invent facts.

### USER
You will receive:  
• The full video transcript (speaker-labeled if available)  
• Any video-level metadata (title, description, tags, publish date, etc.)

Complete **all** sections:

──────────────────────────────
1. CREATIVE-LEVEL METADATA
──────────────────────────────
Create a `creative_info` object with keys:
- `production_title`
- `series_name`
- `season_episode`
- `network_or_platform`
- `release_date`
- `runtime_minutes`
- `genre`
- `creative_type`
- `rating`
- `narrative_structure`
- `plot_arc_summary`
- `mental_health_themes`
- `content_warnings`
- `music_cues`
- `symbolism_used`
- `visual_style`
- `support_resources_displayed`
- `call_to_action`
- `on_screen_text`
- `consultants`
- `representation_notes`
- `fictionalized_elements`
- `creative_team`
- `cast_highlights`
- `filming_locations`

──────────────────────────────
2. COUNT TEENS
──────────────────────────────
* `num_teens` ← integer — count **teen characters** (exclude purely adult roles).

──────────────────────────────
3. TEEN CHARACTER DEMOGRAPHICS
──────────────────────────────
For each teen character (order = first appearance):
- `id`
- `character_name`
- `actor_name`
- `age_estimate`
- `sex`
- `ethnicity`
- `school_status`
- `socioeconomic_status`
- `mental_health_diagnosis`
- `hobbies`
- `family_challenges`
- `residence`
- `drug_use`
- `alcohol_use`
- `support_network`
- `future_plans_stated`
- `character_arc`
- `diet`
- `physical_activity`
- `sleep_patterns`
- `bullying_experience`
- `violence_exposure`
- `suicidal_thoughts`
- `suicide_attempts`
- `sexual_activity`
- `contraception_use`
- `school_connectedness`
- `parental_monitoring`
- `obesity_bmi_status`
- `scenes` (see §5)

──────────────────────────────
4. CREATIVE TEAM & OTHER CHARACTERS
──────────────────────────────
- `creator_info`: `{ name, role, age_estimate }`  
- `other_characters`: array of `{ character_name, actor_name, role_in_story, affiliation }`

──────────────────────────────
5. SCENES
──────────────────────────────
For each mental-health-related situation a teen character experiences:
- `description`
- `transcript_full` (verbatim + inline tags like `[crying]`, `[pacing]`)
- `clinically_plausible`: `"yes"|"no"|"unsure"`
- `what_helped`
- `other_notes`

──────────────────────────────
6. FORMATTING RULES
──────────────────────────────
* Output **exactly one JSON object**  
* Preserve spelling & capitalization  
* Use `"unknown"` / `"unsure"` consistently  
* Do **NOT** invent facts