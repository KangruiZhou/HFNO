import numpy as np

class KDTreeNode:
    '''
    A Tree Node used by `KDTree`.
    '''
    def __init__(self, depth, points, dim, n_blocks, max_depth, gradient_norm_list, borders):
        self.depth = depth
        self.points = points

        self.points_np = np.array(points)
        self.indices_sd = self.points_np[:, -1].astype(int)
        self.dim = dim
        self.n_blocks = n_blocks
        self.max_depth = max_depth
        # Calculate the bounding box
        self.bbox = [(np.min(self.points_np[:, i]), 
            np.max(self.points_np[:, i])) for i in range(self.dim)]

        self.gradient_norm_list = gradient_norm_list
        self.gradient_max_sd_array = np.array(self.gradient_norm_list)[self.indices_sd]
        self.gradient_max_sd_array_sort = np.array(sorted(self.gradient_max_sd_array))
        self.num = len(self.points)
        self.var = np.var(self.gradient_max_sd_array)
        self.borders = borders
        self.L = self.bbox[0][1] - self.bbox[0][0]
        self.H = self.bbox[1][1] - self.bbox[1][0]

    def split_mcts(self):
        # L, H = self.bbox[0][1] - self.bbox[0][0], self.bbox[1][1] - self.bbox[1][0]
        # nextmove_dim = np.random.choice([0, 1], p=MAXMIN([L, H]))
        if self.L < 0.2:
            nextmove_dim = 1
        elif self.H<0.2:
            nextmove_dim = 0
        else:
            nextmove_dim = np.random.choice([0, 1])
        L1, L2 = self.bbox[nextmove_dim]
        split_val = np.random.uniform(L1, L2)
        self.points.sort(key=lambda x: x[nextmove_dim])
        self.points_np = np.array(self.points)
        split_chosen = np.searchsorted(self.points_np[8:-8, nextmove_dim], split_val) + 8  # len(nodes)>8

        borders_l, borders_r = self.borders[:], self.borders[:]
        borders_l[nextmove_dim] = (
            borders_l[nextmove_dim][0], split_val)
        borders_r[nextmove_dim] = (
            split_val, borders_r[nextmove_dim][1]
        )
        return KDTreeNode(self.depth + 1, self.points[:split_chosen],
                self.dim, self.n_blocks,
                self.max_depth, self.gradient_norm_list, borders_l), \
            KDTreeNode(self.depth + 1, self.points[split_chosen:],
                self.dim, self.n_blocks,
                self.max_depth, self.gradient_norm_list, borders_r), nextmove_dim, split_val

    def get_bounding_box(self):
        return self.bbox

    def get_borders(self):
        x = [self.borders[0][0], self.borders[0][0], self.borders[0][1], self.borders[0][1], self.borders[0][0]]
        y = [self.borders[1][0], self.borders[1][1], self.borders[1][1], self.borders[1][0], self.borders[1][0]]
        return x,y
    def get_box_list_node(self):
        x = [self.bbox[0][0], self.bbox[0][0], self.bbox[0][1], self.bbox[0][1], self.bbox[0][0]]
        y = [self.bbox[1][0], self.bbox[1][1], self.bbox[1][1], self.bbox[1][0], self.bbox[1][0]]
        return x,y

def MAXMIN(lis):
    arr = np.array(lis)
    new_arr = arr / arr.sum()
    return new_arr