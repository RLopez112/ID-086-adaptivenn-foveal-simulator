# Initiative Intent: ID-086 - AdaptiveNN Foveal Attention Orienting Simulator

## Metadata
- **Idea ID**: ID-086
- **Project Name**: AdaptiveNN Foveal Attention Orienting Simulator
- **Category**: IA / Ciencias Cognitivas
- **Status**: En Progreso
- **Date Created**: 2026-11-04
- **Root Directory**: `F:\PERSONALES\_PROJECTS\ID-086-adaptivenn-foveal-simulator`
- **Tracker Reference**: `F:/PERSONALES/_PROJECTS/ideaTracker/00_REFERENCE/Generated/_artifacts/IdeaTracker.csv`

---

## 1. Problem Statement & Context
Embodied active-vision simulation in NVIDIA Isaac Sim implementing the AdaptiveNN coarse-to-fine sequential perception architecture (from Wang et al., NMI 2025) to model attentional orienting and spatial cueing (Posner paradigm). A lightweight coarse network processes wide-angle low-resolution peripheral views to detect spatial cues and propose fixation targets, while a sequential policy triggers high-resolution foveal patch crops to evaluate attentional capture, invalid cue reorienting latency, and early observation termination.

---

## 2. Primary Objectives & Deliverables
- [ ] Define system architecture and technical scope.
- [ ] Establish initial code / asset prototypes in `50_PROJECT/`.
- [ ] Validate core hypothesis or prototype functionality.
- [ ] Produce final deliverable package in `90_DELIVERY/`.

---

## 3. Scope & Target Stack
- **Domain**: IA / Ciencias Cognitivas
- **Target Platform / Environment**: TBD
- **Pipeline Layout**: Minimal Workspace Layout (`00_REFERENCE`, `50_PROJECT` - expandable on demand)

---

## 4. Immediate Next Steps
- Run `spec-first-sdlc` to formulate High Level Design (HLD) and Executable Implementation Specs (EIS).
- If background investigation is required, run `idea-deep-research`.
