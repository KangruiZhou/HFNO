import sys
sys.path.append('model')
sys.path.append('.')
import math
from kdtree_mcts import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from util.utilities_HFNO import cal_gradient
import argparse
from model.mcts import *


SCALAR = 1 / (2 * math.sqrt(2.0))     # MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sims', type=int, default = 1000)
    parser.add_argument('--num_dataset', type=int, default=5)
    parser.add_argument('--NUM_TURNS', type=int, default=6)
    parser.add_argument('--PATH', type=str, default='data/data_case1.npy')
    parser.add_argument('--filename', type=str, default='result/saved_KDtree/KDtree.npy')
    parser.add_argument('--variable', type=str, default='sigma_xy_2d')
    args = parser.parse_args()

    data = np.load(args.PATH, allow_pickle=True).item()
    input_s = data[args.variable]
    input_rr = data['T_2d']
    input_x = data['xx_2d']
    input_y = data['yy_2d']
    cell = data['cells']
    xy = np.stack((input_x, input_y), axis=2)
    xy = xy[0]

    num_nodes = xy.shape[0]
    gradient_norm_list = cal_gradient(xy, input_s, num_dataset=args.num_dataset)
    gradient_global_var = np.var(gradient_norm_list)

    n_subdomain = args.NUM_TURNS +1
    point_cloud = xy.tolist()
    points_indices = [list(point_cloud[i]) + [i] for i in range(len(point_cloud))]
    dim = 2

    tree_current = KDTree(points_indices, dim=dim, n_blocks=2 ** dim,
                        gradient_norm_list=gradient_norm_list)
    current_node = Node(State())
    cmap_names = ["viridis", "RdBu", "Set1", "jet"]
    cmap = mpl.cm.get_cmap(cmap_names[0], n_subdomain)
    colors = cmap(np.linspace(0, 1, n_subdomain))

    import time
    time_start = time.time()  # 记录开始时间
    for l in range(args.NUM_TURNS):
        print("============Node: %d==========" % (l + 1))
        current_node = UCTSEARCH(args.num_sims, current_node, points_indices, gradient_norm_list, gradient_global_var, dim, SCALAR)
        print("Num Children: %d" % len(current_node.children))
        for j, c in enumerate(current_node.children):
            print(j, c)
        print("Best Child: %s" % current_node.state)
        print("--------------------------------")

    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)

    tree_current.split_move(current_node.state.moves)
    print("current_node_var_sum: %f" % (tree_current.get_var_sum() / gradient_global_var / tree_current.total_num))

    ## save
    data = dict(Tree=tree_current)
    np.save(args.filename, data)

    ## plot
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    for i, KDnode in enumerate(tree_current.nodes):
        x = KDnode.points_np[:, 0]
        y = KDnode.points_np[:, 1]
        ax.scatter(x, y, s=10, c=np.array(colors[i]).reshape(1,-1))
    # ax.legend()
    ax.set_title('Scatter Plot of Points in Different Regions')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    name_ = args.PATH.split('/')[-1][:-4]
    plt.savefig(f"result/Decomposition_{name_}.png")


