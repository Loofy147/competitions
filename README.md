# Competitor AI Architecture

Sovereign Kaggle competitor architecture synthesizing state-of-the-art logic from top kernels.

## Formal Grounding
The architecture is defined as  = (R, K, M, I, O)$:
- **R (Representation Space)**: Multi-modal support including one-hot grid mapping and kernel manifolds.
- **K (Constraints)**: Resource-aware (4-bit quantization), dimension-constrained (30x30 grids).
- **M (Computational Model)**: Skill-based modular orchestration with manifold surfing and consensus voting.
- **I (Input)**: Unified task interface for ARC, Math, and Tabular domains.
- **O (Output)**: Calibrated predictions and transformation representations.

## Organization
- `src/competition_ai/core`: Main architecture and improvement pipelines.
- `src/competition_ai/skills`: Domain-specific skill modules (ARC, Math, Tabular, Opt, Val).
- `src/competition_ai/utils`: Utility functions like FieldCoordinates.

## Installation
```bash
pip install -e .
```
