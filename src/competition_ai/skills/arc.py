import numpy as np
from typing import List, Dict, Any, Tuple

class ARCSkill:
    """
    Skill: ARC-AGI Specialized logic.
    Extracted from Qwen3-Unsloth and Strafos kernels.
    """
    @staticmethod
    def augment_grid(grid: List[List[int]]) -> List[Tuple[np.ndarray, str]]:
        """Extensive data augmentation for ARC grids."""
        arr = np.array(grid)
        augs = []
        # Rotations
        for k in range(4):
            augs.append((np.rot90(arr, k=k), f"rot90_{k}"))
        # Transpose & Rotations
        transposed = np.transpose(arr)
        for k in range(4):
            augs.append((np.rot90(transposed, k=k), f"transpose_rot90_{k}"))
        return augs

    @staticmethod
    def score_solutions(guesses: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Consensus-based scoring logic from ARC2-Qwen3 kernel.
        Sorts solutions by weight (count + confidence).
        """
        scores = {}
        for g in guesses:
            solution_tuple = tuple(map(tuple, g["solution"]))
            if solution_tuple not in scores:
                scores[solution_tuple] = {"count": 0, "total_score": 0.0, "solution": g["solution"]}
            scores[solution_tuple]["count"] += 1
            scores[solution_tuple]["total_score"] += g.get("confidence", 1.0)

        # Rank by count then total_score
        ranked = sorted(scores.values(), key=lambda x: (x["count"], x["total_score"]), reverse=True)
        return [r["solution"] for r in ranked]
