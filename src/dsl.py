import numpy as np

class DSL:
    def __init__(self):
        self.primitives = [f"prim_{i}" for i in range(200)]
        self.categories = {
            "geometric": self.primitives[:60],
            "color": self.primitives[60:100],
            "logical": self.primitives[100:160],
            "pattern": self.primitives[160:]
        }

    def apply(self, grid, primitive):
        grid_size = grid.shape[0]
        if "geometric" in primitive:
            return np.rot90(grid, k=1)
        elif "color" in primitive:
            new_grid = grid.copy()
            new_grid[new_grid == 1] = 2
            return new_grid
        elif "logical" in primitive:
            new_grid = grid.copy()
            new_grid[grid > 0] += 1
            return new_grid
        else:
            new_grid = grid.copy()
            new_grid[:, :grid_size//2] = new_grid[:, grid_size//2:]
            return new_grid

dsl = DSL()
