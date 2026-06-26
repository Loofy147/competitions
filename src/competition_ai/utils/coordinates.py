import numpy as np
from typing import List

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
        grid_np = np.array(grid)
        h, w = grid_np.shape
        one_hot = np.zeros((FieldCoordinates.CHANNELS, FieldCoordinates.HEIGHT, FieldCoordinates.WIDTH), dtype=np.float32)
        h_lim, w_lim = min(h, FieldCoordinates.HEIGHT), min(w, FieldCoordinates.WIDTH)
        for r in range(h_lim):
            for c in range(w_lim):
                color = int(grid_np[r, c])
                if 0 <= color < FieldCoordinates.CHANNELS:
                    one_hot[color, r, c] = 1.0
        return one_hot

    @staticmethod
    def from_one_hot(one_hot: np.ndarray) -> List[List[int]]:
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
