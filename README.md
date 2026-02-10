# Code for survey-anchored adolescent digital twins with RAG memory enrichment and multi-layer validation.

This repository contains the code, prompts, and derived artifacts used in our Nature Human Behaviour submission.  
It is a working research repo (still being cleaned up), but the main pipelines reported in the paper are here.

---

## How this repo maps to the paper

The paper has four main technical pieces, and the repo mirrors them:

1) **Survey-anchored cohort + baseline digital twins**  
2) **Memory enrichment (YouTube micro-scenes + retrieval + profile enrichment)**  
3) **Validation suite (survey fidelity + developmental coherence + human chat study analysis)**  
4) **Virtual RCT replications (10 studies)**

---

## Data (`data/`)

### YRBS anchoring (`data/raw/yrbs/`)
- `Final_Mapping_Table__Q1_Q107_.csv`  
  Mapping from 107 YRBS items to second-person persona statements (used to build DT profiles).
- `questions_107_converted.json`  
  Canonical JSON of the 107 items used throughout prompt generation.
- `stratified_sample_1000_DTs.csv`  
  The stratified N=1000 cohort used for the main experiments.

**Paper:** DT construction + cohort description (Methods; Appendix details).

### YouTube corpus list (`data/raw/youtube/`)
- `video_list.csv`  
  Video list used for transcript/scene extraction and knowledge base construction.

**Paper:** memory enrichment data source (Methods; Appendix).

---

## Prompts (`prompts/`)

### Shared DT rules
- `prompts/dt_shared/shared_prompt.txt`  
  Shared system-level rules used across DT interactions.

### Survey fidelity prompts (1000-DT cohort)
- `prompts/1000 DT baseline persona/internal_validation_prompts.csv`  
  Internal agreement (re-asking YRBS items).
- `prompts/1000 DT baseline persona/external_validation_prompts_*.csv`  
  External holdout sets (selected items removed; file name indicates the holdout).

**Paper:** Layer 1 internal + external agreement (Methods + Results; Appendix holdouts).

### Questionnaire prompts (developmental coherence)
- `prompts/questionnaires/ASRI_36.csv`, `BFI_10.csv`, `SDQ_25.csv`, `Online Victimization.csv`

**Paper:** Layer 2 developmental coherence (Methods + Results; Appendix prompts).

### Scene extraction templates
- `prompts/scene_extraction/*.prompt.md`  
  Genre-specific templates used when extracting micro-scenes.

**Paper:** memory extraction protocol (Appendix).

### RCT replication metadata
- `prompts/rct_replication/study list.csv`  
  List and definitions for the 10 RCT-style replications.

**Paper:** Layer 4 RCT replication (Methods; Appendix study list).

---

## Scripts (`scripts/`)

### 1) Cohort + baseline profiles (`scripts/cohort_generation/`)
- `01_stratified_sampling.py`  
- `02_baseline_profiles.py`  
- `03_dt_chat_interface.py` (utility)

**Paper:** survey-anchored DT construction (Methods; Appendix mapping).

### 2) Memory enrichment (`scripts/enrich_memory/`)
- `01_collect_youtube.py` → `02_extract_scenes.py` → `03_build_knowledge_base.py`  
- `04_retrieve_top2_memories.py` → `05_enrich_profiles_and_conversations.py`

**Paper:** Survey+Memory DT variant (Methods; Appendix).

### 3) Validation suite (`scripts/validation/`)
- `01_run_internal_validation.py`  
- `02_run_external_validation.py`  
- `03_run_psychological_validation.py`  
- `04_run_heterogeneity_analysis.py`  
- `05_analyze_internal_external.py`  
- `06_analyze_psychological.py`  
- `07_analyze_human_experiment.py`

**Paper:** Layer 1–3 validations (Methods + Results; Appendix).

### 4) RCT replications (`scripts/rct/`)
- `01_run_all_simulations.py`  
- `02_process_study_01.py` … `11_process_study_10.py`  
- `12_analyze_all_rct.py`

**Paper:** Layer 4 RCT replications (Methods + Results; Appendix per-study ops).

---

## Results (`results/`)

These folders store the CSV outputs used in the paper figures/tables:

- `results/internal validation/clear_*.csv`  
  Internal agreement outputs across model/DT variants.
- `results/external validation/<model>/*.csv`  
  External holdout agreement outputs.
- `results/psychological validation/`  
  Questionnaire outputs (ASRI/BFI/SDQ, online victimization, risk-taking).
- `results/rct_replication_results/`  
  Raw simulation outputs for Base-DT / Survey-DT / Survey+Memory DT.

---

## Minimal reproduction (high level)

Survey fidelity (internal + external):

    python scripts/validation/01_run_internal_validation.py
    python scripts/validation/02_run_external_validation.py

Developmental coherence (questionnaires):

    python scripts/validation/03_run_psychological_validation.py
    python scripts/validation/06_analyze_psychological.py

Virtual RCT replications (10 studies):

    python scripts/rct/01_run_all_simulations.py
    python scripts/rct/12_analyze_all_rct.py

For model/backbone selection, RAG toggles, and output paths:

    python <script.py> -h

## Notes on data release

- YRBS: this repo includes the mapping table, item JSON, and the stratified cohort file used for the experiments.
- YouTube: this repo includes the video list and the full scene-extraction and knowledge-base pipeline.

## Intended use

This codebase is for methodological evaluation and early-stage hypothesis testing with synthetic, survey-anchored agents. It is not intended for clinical decision-making, individual risk assessment, or estimating real-world prevalence.


