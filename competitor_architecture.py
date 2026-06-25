import numpy as np
import json
import math
from typing import List, Dict, Any, Tuple

"""
Formal Grounding of the Competitor Architecture:
A = (R, K, M, I, O)

R (Representation Space): One-hot encoded grids (C, H, W) and manifold of transformation kernels.
K (Constraints): Grid dimensions (30x30), discrete color space (10 channels), valid kernels.
M (Computational Model): Search-based manifold navigation (Surfing).
I (Input): Task specification (train/test examples).
O (Output): Predicted transformations/representations.
"""

class FieldCoordinates:
    """
    R: Representation Space Mapping.
    Handles one-hot grid mapping and coordinate transformations.
    """
    CHANNELS = 10
    HEIGHT = 30
    WIDTH = 30

    @staticmethod
    def to_one_hot(grid: List[List[int]]) -> np.ndarray:
        """I -> R: Converts a 2D grid to a one-hot encoded representation space."""
        grid_np = np.array(grid)
        h, w = grid_np.shape
        one_hot = np.zeros((FieldCoordinates.CHANNELS, FieldCoordinates.HEIGHT, FieldCoordinates.WIDTH), dtype=np.float32)

        # K: Constraint Clipping
        h_lim, w_lim = min(h, FieldCoordinates.HEIGHT), min(w, FieldCoordinates.WIDTH)

        for r in range(h_lim):
            for c in range(w_lim):
                color = int(grid_np[r, c])
                if 0 <= color < FieldCoordinates.CHANNELS:
                    one_hot[color, r, c] = 1.0
        return one_hot

    @staticmethod
    def from_one_hot(one_hot: np.ndarray) -> List[List[int]]:
        """R -> O: Converts representation space back to output grid."""
        channels, height, width = one_hot.shape
        grid = []
        for r in range(height):
            row = []
            for c in range(width):
                colors = [ch for ch in range(channels) if one_hot[ch, r, c] >= 0.5]
                if len(colors) == 1:
                    row.append(colors[0])
                elif len(colors) > 1:
                    row.append(11)
                else:
                    row.append(10)

            while row and row[-1] == 10:
                row.pop()
            grid.append(row)

        while grid and not grid[-1]:
            grid.pop()

        return grid

class SurfingSearch:
    """
    M: Computational Model.
    Implements 'Surfing' (Manifold Search) over transformation kernels.
    """
    def __init__(self, kernels=None):
        self.kernels = kernels or self._default_kernels()

    def _default_kernels(self):
        """K: Constrained transformation space."""
        return {
            "identity": lambda x: x,
            "rotate_90": lambda x: np.rot90(x, k=1),
            "rotate_180": lambda x: np.rot90(x, k=2),
            "rotate_270": lambda x: np.rot90(x, k=3),
            "flip_ud": lambda x: np.flipud(x),
            "flip_lr": lambda x: np.fliplr(x),
        }

    def find_best_kernel(self, train_examples: List[Dict[str, Any]]):
        """
        M: Optimization over constraints.
        'Surfs' the manifold of kernels to find one that satisfies all train examples.
        """
        best_kernel_name = None
        for name, kernel_fn in self.kernels.items():
            match = True
            for ex in train_examples:
                inp = np.array(ex['input'])
                out = np.array(ex['output'])
                try:
                    pred = kernel_fn(inp)
                    if not np.array_equal(pred, out):
                        match = False
                        break
                except Exception:
                    match = False
                    break

            if match:
                best_kernel_name = name
                break

        return self.kernels.get(best_kernel_name) if best_kernel_name else None

    def navigate_manifold(self, task_json: Dict[str, Any]):
        """I -> M -> O: Navigates the representation manifold to produce output."""
        best_kernel = self.find_best_kernel(task_json['train'])

        results = []
        for test_ex in task_json['test']:
            if best_kernel:
                inp = np.array(test_ex['input'])
                ans = best_kernel(inp).tolist()
            else:
                ans = test_ex['input']
            results.append(ans)
        return results

class CompetitorArchitecture:
    """
    A = (R, K, M, I, O)
    Synthesized Competitor Architecture.
    """
    def __init__(self):
        self.coords = FieldCoordinates()
        self.search = SurfingSearch()

    def solve(self, task: Dict[str, Any]) -> List[List[List[int]]]:
        """
        A(I) -> O: Constrained computational transformation.
        """
        return self.search.navigate_manifold(task)

if __name__ == "__main__":
    # Integration Test
    arch = CompetitorArchitecture()

    sample_task = {
        "train": [
            {"input": [[1, 0], [0, 0]], "output": [[0, 1], [0, 0]]},
            {"input": [[0, 1], [1, 1]], "output": [[1, 0], [1, 1]]}
        ],
        "test": [
            {"input": [[1, 1], [0, 0]]}
        ]
    }

    result = arch.solve(sample_task)
    print(f"[*] Formal Algorithm A(I) Output: {result[0]}")
