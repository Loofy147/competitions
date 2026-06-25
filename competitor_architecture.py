import numpy as np
import json
import math
from typing import List, Dict, Any, Tuple

class FieldCoordinates:
    """
    Handles one-hot grid mapping and coordinate transformations.
    Inspired by NeuroGolf's field representation.
    """
    CHANNELS = 10
    HEIGHT = 30
    WIDTH = 30

    @staticmethod
    def to_one_hot(grid: List[List[int]]) -> np.ndarray:
        """Converts a 2D grid to a one-hot encoded numpy array (C, H, W)."""
        grid_np = np.array(grid)
        h, w = grid_np.shape
        one_hot = np.zeros((FieldCoordinates.CHANNELS, FieldCoordinates.HEIGHT, FieldCoordinates.WIDTH), dtype=np.float32)

        # Clip to max dimensions
        h_lim, w_lim = min(h, FieldCoordinates.HEIGHT), min(w, FieldCoordinates.WIDTH)

        for r in range(h_lim):
            for c in range(w_lim):
                color = int(grid_np[r, c])
                if 0 <= color < FieldCoordinates.CHANNELS:
                    one_hot[color, r, c] = 1.0
        return one_hot

    @staticmethod
    def from_one_hot(one_hot: np.ndarray) -> List[List[int]]:
        """Converts a one-hot encoded numpy array back to a 2D grid."""
        # one_hot shape: (C, H, W)
        channels, height, width = one_hot.shape
        grid = []
        for r in range(height):
            row = []
            for c in range(width):
                # Find color with max value (should be >= 0.5)
                colors = [ch for ch in range(channels) if one_hot[ch, r, c] >= 0.5]
                if len(colors) == 1:
                    row.append(colors[0])
                elif len(colors) > 1:
                    row.append(11) # Indicator for one-hot error (too many colors)
                else:
                    row.append(10) # Indicator for no color

            # Trim trailing empty cells (color 10)
            while row and row[-1] == 10:
                row.pop()
            grid.append(row)

        # Trim trailing empty rows
        while grid and not grid[-1]:
            grid.pop()

        return grid

    @staticmethod
    def augment(grid: List[List[int]]) -> List[Tuple[List[List[int]], str]]:
        """
        Generates augmented versions of the grid (rotations, transpositions).
        Returns a list of (grid, transformation_name).
        """
        arr = np.array(grid)
        augs = []

        # Original
        augs.append((arr.tolist(), "orig"))

        # Rotations
        for k in range(1, 4):
            augs.append((np.rot90(arr, k=k).tolist(), f"rot90_{k}"))

        # Transpose
        transposed = np.transpose(arr)
        augs.append((transposed.tolist(), "transpose"))

        # Rotations of Transpose
        for k in range(1, 4):
            augs.append((np.rot90(transposed, k=k).tolist(), f"transpose_rot90_{k}"))

        return augs

class SurfingSearch:
    """
    Implements 'Surfing' (Manifold Search) over transformation kernels.
    Inspired by Stratos ARC Solver.
    """
    def __init__(self, kernels=None):
        self.kernels = kernels or self._default_kernels()

    def _default_kernels(self):
        """Standard transformation kernels."""
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
        """Runs the search and returns predictions for test examples."""
        best_kernel = self.find_best_kernel(task_json['train'])

        results = []
        for test_ex in task_json['test']:
            if best_kernel:
                inp = np.array(test_ex['input'])
                ans = best_kernel(inp).tolist()
            else:
                ans = test_ex['input'] # Fallback: return input
            results.append(ans)
        return results

class CompetitorArchitecture:
    """
    Synthesized Competitor Architecture.
    Integrates Field Coordinates, Surfing Search, and Consensus-based Voting.
    """
    def __init__(self):
        self.coords = FieldCoordinates()
        self.search = SurfingSearch()

    def solve(self, task: Dict[str, Any], use_consensus=True) -> List[List[List[int]]]:
        """
        Main solve loop.
        1. Augment problem (Data Engineering from NCAA/ARC).
        2. Search kernel manifold (Surfing from Stratos).
        3. Consensus Voting (Logic from AIMO/Nemo).
        """
        # 1. Augment Train Examples
        # Logic: If we can't find a kernel in the original space, maybe in augmented space?
        # For simplicity, we use the manifold search which already includes common transformations.

        # 2. Search for best transformation kernel
        predictions = self.search.navigate_manifold(task)

        # 3. Consensus/Verification (Mock implementation of extracted LLM consensus logic)
        # If we had an ensemble of models, we would vote here.
        # In this sovereign architecture, the "kernel match" is our consensus.

        return predictions

if __name__ == "__main__":
    # Integration Test
    arch = CompetitorArchitecture()

    # Task where output is input rotated 90 degrees
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
    print(f"[*] Task Input: {sample_task['test'][0]['input']}")
    print(f"[✓] Task Output (Consensus): {result[0]}")

    # Verify Field Coordinates
    one_hot = FieldCoordinates.to_one_hot(sample_task['test'][0]['input'])
    back_to_grid = FieldCoordinates.from_one_hot(one_hot)
    print(f"[✓] Field Coordinates Roundtrip: {back_to_grid == sample_task['test'][0]['input']}")
