import numpy as np
from .node import KDTreeNode



class KDTree:
    def __init__(self, points_indices, dim, gradient_norm_list, n_blocks=8,
        smallest_points=8, overall_borders=None):

        self.dim = dim
        if overall_borders is None:
            points_np = np.array(points_indices)
            overall_borders = [(np.min(points_np[:, i]),
            np.max(points_np[:, i])) for i in range(self.dim)]
        # self.smallest_points = smallest_points
        self.nodes = [KDTreeNode(0, points_indices, dim=dim, n_blocks=n_blocks, max_depth=None,
                          gradient_norm_list=gradient_norm_list, borders=overall_borders)]
        # self.overall_bbox = self.nodes[0].bbox
    
        # self.gradient_span = max(gradient_norm_list) - min(gradient_norm_list)
        self.total_num = len(points_indices)

    def get_var_sum(self):
        var_sum = np.sum([node.var * node.num for node in self.nodes])
        return var_sum

    def get_var_sum_list(self):
        var_sum = [node.var * node.num for node in self.nodes]
        return var_sum

    def get_var_list(self):
        var_list = [node.var for node in self.nodes]
        return var_list

    def get_subdomain_points(self):
        '''
        Remember to call `solve()` before calling this function.
        '''
        if self.return_indices:
            return [node.points_np[:, :-1] for node in self.nodes]
        else:
            return [node.points_np for node in self.nodes]

    def get_subdomain_bounding_boxes(self):
        '''
        Format: [[(np.min(nodes.points_dim_i),
            np.max(nodes.points_dim_i)) for i in range(self.dim)]
            for node in self.nodes]
        Remember to call `solve()` before calling this function.
        '''
        return [node.get_bounding_box() for node in self.nodes]

    def get_subdomain_indices(self):
        '''
        Remember to call `solve()` before calling this function.
        Reminder: if `return_indices == False`, it will return `None`.
        '''
        return [node.points_np[:, -1].astype(int)
            for node in self.nodes]
    def get_subdomain_borders(self):
        # borders = [node.get_borders() for node in self.nodes]
        x = [node.get_borders()[0]
            for node in self.nodes]
        y = [node.get_borders()[1]
            for node in self.nodes]
        return x,y

    def get_bbox_list_tree(self):
        x = [node.get_box_list_node()[0]
            for node in self.nodes]
        y = [node.get_box_list_node()[1]
            for node in self.nodes]
        return x,y

    def split(self):
        # ind = list(range(len(self.nodes)))
        ind = [i for i in range(len(self.nodes))
               if self.nodes[i].L>=0.2 or self.nodes[i].H>=0.2]
        KDode_chosen = np.random.choice(ind)
        if len(self.nodes) == 1:
            KDode_chosen = ind[0]
        else:
            arr = np.array(self.get_var_list())
            var_list = list(arr[ind])
            KDode_chosen = np.random.choice(ind, p=Max_Min(var_list))
            # KDode_chosen = np.random.choice(ind, p=Max_Min(self.get_var_sum_list()))

        KDnode_split = self.nodes[KDode_chosen]
        while KDnode_split.num < 16:
            KDode_chosen = np.random.choice(ind)
            KDnode_split = self.nodes[KDode_chosen]
        son_a, son_b, nextmove_dim, split_val = KDnode_split.split_mcts()
        self.nodes[KDode_chosen] = son_a
        self.nodes.append(son_b)
        return nextmove_dim, KDode_chosen, split_val

    def split_move(self, moves):
        for move in moves:
            nextmove_dim, KDode_chosen, split_val = move
            ind = list(range(len(self)))
            KDnode_split = self.nodes[KDode_chosen]
            KDnode_split.points.sort(key=lambda x: x[nextmove_dim])
            KDnode_split.points_np = np.array(KDnode_split.points)
            split_chosen = np.searchsorted(KDnode_split.points_np[8:-8, nextmove_dim], split_val) + 8  # len(nodes)>8

            borders_l, borders_r = KDnode_split.borders[:], KDnode_split.borders[:]
            borders_l[nextmove_dim] = (
                borders_l[nextmove_dim][0], split_val)
            borders_r[nextmove_dim] = (
                split_val, borders_r[nextmove_dim][1])

            son_a = KDTreeNode(KDnode_split.depth + 1, KDnode_split.points[:split_chosen],
                    KDnode_split.dim, KDnode_split.n_blocks,
                    KDnode_split.max_depth, KDnode_split.gradient_norm_list,borders_l)
            son_b = KDTreeNode(KDnode_split.depth + 1, KDnode_split.points[split_chosen:],
                    KDnode_split.dim, KDnode_split.n_blocks,
                    KDnode_split.max_depth, KDnode_split.gradient_norm_list,borders_r)
            # self.nodes.pop(ind[KDode_chosen])
            # self.nodes.append(son_a)
            # self.nodes.append(son_b)
            self.nodes[KDode_chosen] = son_a
            self.nodes.append(son_b)
        return self

    def __len__(self):
        return len(self.nodes)
    
    def get_subdomain_borders2(self):
        return [node.borders for node in self.nodes]

def Max_Min(array):
    return (array / sum(array))