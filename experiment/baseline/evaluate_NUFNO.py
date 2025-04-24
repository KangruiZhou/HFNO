import sys
sys.path.append('model')
sys.path.append('.')

from timeit import default_timer
from util.kdtree_NUFNO.tree import *
from util.utilities_HFNO import *
from scipy.interpolate import LinearNDInterpolator, \
    NearestNDInterpolator, RegularGridInterpolator
import datetime
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

################################################################
# Configs
################################################################
# Data path
PATH = 'data/data_case3.npy'
PATH_mcts='result/saved_KDtree/KDTree_3.npy'
load_name = 'result/saved_model/model_NUNO.pt'
checkpoint = torch.load(load_name)
device = 'cuda'
##
tree = np.load(PATH_mcts, allow_pickle=True).item()['Tree']
bbox_sd = tree.get_subdomain_bounding_boxes()
indices_sd = tree.get_subdomain_indices()

gradient_mean_nodes_array = np.array([np.mean(node.gradient_max_sd_array) for node in tree.nodes])
high_gradient_mask = gradient_mean_nodes_array > np.mean(gradient_mean_nodes_array)
high_gradient_index = np.where(high_gradient_mask)[0]
indices_sd_high = []
bbox_sd_high = []
indices_all_high = np.array([])
for index in high_gradient_index:
    indices_sd_high.append( indices_sd[index] )
    bbox_sd_high.append( bbox_sd[index] )
    indices_all_high = np.concatenate((indices_all_high, indices_sd[index]))

indices_all_list_high = list(indices_all_high.astype(int))

low_gradient_mask = ~high_gradient_mask
low_gradient_index = np.where(low_gradient_mask)[0]
indices_sd_low = []
bbox_sd_low = []
indices_all_low = np.array([])
for index in low_gradient_index:
    indices_sd_low.append( indices_sd[index] )
    bbox_sd_low.append( bbox_sd[index] )
    indices_all_low = np.concatenate((indices_all_low, indices_sd[index]))

indices_all_list_low = list(indices_all_low.astype(int))

# Dataset params
n_train = 5000
n_test = 1000
n_total = n_train + n_test
# The number of points in (output) point cloud

# FNO configs
modes = 8
width = 20

batch_size = 20
# Grid params
oversamp_ratio = 2.0        # used to calculate grid sizes
input_dim = 3               # (x, y, T)
output_dim = 1              # (sigma)

# K-D tree params
n_subdomains = 16

if __name__ == '__main__':
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d %H:%M:%S")
    ################################################################
    # Load data and preprocessing
    ################################################################
    data = np.load(PATH, allow_pickle=True).item()
    sigma = data['sigma_xy_2d']
    T = data['T_2d']
    input_x = data['xx_2d']
    input_y = data['yy_2d']

    input_point_cloud = np.stack((sigma, input_x, input_y, T), axis=2)
    input_point_cloud = input_point_cloud.reshape(input_point_cloud.shape[0],-1,4)
    sigma_mean = np.mean(input_point_cloud[:n_train, :])
    sigma_std = np.std(input_point_cloud[:n_train, :])
    input_point_cloud = input_point_cloud[:n_total]

    xy = np.stack((input_x, input_y), axis=2)
    xy = xy[0]

    print("Start KD-Tree splitting...")
    t1 = default_timer()
    point_cloud = xy.tolist()
    # Use kd-tree to generate subdomain division
    tree= KDTree(
        point_cloud, dim=2, n_subdomains=n_subdomains,
        n_blocks=8, return_indices=True
    )
    tree.solve()

    # Gather subdomain info
    bbox_sd = tree.get_subdomain_bounding_boxes()
    indices_sd = tree.get_subdomain_indices()
    # Pad the point cloud of each subdomain to the same size
    max_n_points_sd = np.max([len(indices_sd[i])
                              for i in range(n_subdomains)])
    xy_sd = np.zeros((1, max_n_points_sd, n_subdomains, 2))
    input_point_cloud_sd = np.zeros((n_total,
                                     max_n_points_sd, input_point_cloud.shape[-1], n_subdomains))
    # Mask is used to ignore padded zeros when calculating errors
    input_u_sd_mask = np.zeros((1, max_n_points_sd, 1, n_subdomains))
    # The maximum grid shape of subdomains
    grid_shape = [-1] * 2
    # (s1, s2, s3)
    # The new coordinate order of each subdomain
    # (after long side alignment)
    order_sd = []
    for i in range(n_subdomains):
        # Normalize to [-1, 1]
        _xy = xy[indices_sd[i], :]
        _min, _max = np.min(_xy, axis=0, keepdims=True), \
            np.max(_xy, axis=0, keepdims=True)
        _xy = (_xy - _min) / (_max - _min) * 2 - 1  # normalization
        # Long side alignment
        bbox = bbox_sd[i]
        scales = [bbox[j][1] - bbox[j][0] for j in range(2)]
        order = np.argsort(scales)
        _xy = _xy[:, order]
        order_sd.append(order.tolist())
        # Calculate the grid shape
        _grid_shape = cal_grid_shape(
            oversamp_ratio * len(indices_sd[i]), scales)
        _grid_shape.sort()
        grid_shape = np.maximum(grid_shape, _grid_shape)
        # Applying
        xy_sd[0, :len(indices_sd[i]), i, :] = _xy
        input_point_cloud_sd[:, :len(indices_sd[i]), :, i] = \
            input_point_cloud[:, indices_sd[i], :]
        input_u_sd_mask[0, :len(indices_sd[i]), 0, i] = 1.
    print(grid_shape)

    grid_shape = np.array(grid_shape)
    t2 = default_timer()
    print("Finish KD-Tree splitting, time elapsed: {:.1f}s".format(t2 - t1))

    # Interpolation from point cloud to uniform grid
    t1 = default_timer()
    print("Start interpolation...")
    input_sd_grid = []
    point_cloud = xy
    point_cloud_val = np.transpose(input_point_cloud, (1, 2, 0))
    interp_linear = LinearNDInterpolator(point_cloud, point_cloud_val)
    interp_nearest = NearestNDInterpolator(point_cloud, point_cloud_val)
    for i in range(n_subdomains):
        bbox = bbox_sd[i]
        _grid_shape = grid_shape[np.argsort(order_sd[i])]
        # Linear interpolation
        grid_x = np.linspace(bbox[0][0], bbox[0][1],
                             num=_grid_shape[0])
        grid_y = np.linspace(bbox[1][0], bbox[1][1],
                             num=_grid_shape[1])
        grid_x, grid_y = np.meshgrid(
            grid_x, grid_y, indexing='ij')
        grid_val = interp_linear(grid_x, grid_y)

        # Fill nan values
        nan_indices = np.isnan(grid_val)[..., 0, 0]

        # Long size alignment
        grid_val = np.transpose(grid_val,
                                order_sd[i] + [2, 3])
        input_sd_grid.append(np.transpose(grid_val, (3, 0, 1, 2)))
    # Convert indexing to 'xy'
    input_sd_grid = np.transpose(
        np.array(input_sd_grid), (1, 3, 2, 4, 0))

    t2 = default_timer()
    print("Finish interpolation, time elapsed: {:.1f}s".format(t2 - t1))

    xy_sd = torch.from_numpy(xy_sd).cuda().float()
    xy_sd = xy_sd.repeat([batch_size, 1, 1, 1]) \
        .permute(0, 2, 1, 3) \
        .reshape(batch_size * n_subdomains, -1, 1, 2)
    # shape: (batch * n_subdomains, n_points_sd_padded, 1, 1, 3)
    input_point_cloud_sd = torch.from_numpy(input_point_cloud_sd).float()
    # shape: (ntotal, n_points_sd_padded, output_dim, n_subdomains)
    input_u_sd_mask = torch.from_numpy(input_u_sd_mask).cuda().float()
    # shape: (1, n_points_sd_padded, 1, n_subdomains)
    input_sd_grid = torch.from_numpy(input_sd_grid).float()
    # shape: (n_total, s2, s1, s3, input_dim + output_dim, n_subdomains)

    train_a_sd_grid = input_sd_grid[:n_train, ..., 1:4, :]. \
        reshape(n_train, grid_shape[1],
                grid_shape[0], -1).cuda()
    test_a_sd_grid = input_sd_grid[-n_test:, ..., 1:4, :]. \
        reshape(n_test, grid_shape[1],
                grid_shape[0], -1).cuda()

    input_sd_grid = input_sd_grid[..., 0:1, :]
    train_u_sd_grid = input_sd_grid[:n_train].cuda()
    test_u_sd_grid = input_sd_grid[-n_test:].cuda()

    input_point_cloud_sd = input_point_cloud_sd[..., 0:1, :]
    train_u_point_cloud = input_point_cloud_sd[:n_train].cuda()
    test_u_point_cloud = input_point_cloud_sd[-n_test:]

    a_normalizer = UnitGaussianNormalizer(train_a_sd_grid)
    train_a_sd_grid = a_normalizer.encode(train_a_sd_grid)
    test_a_sd_grid = a_normalizer.encode(test_a_sd_grid)

    y_normalizer = UnitGaussianNormalizer(train_u_sd_grid)

    ################################################################
    # Re-experiment with different random seeds
    ################################################################
    train_l2_res = []
    test_l2_res = []
    time_res = []
    test_T_l2_res = []

    train_a = train_a_sd_grid; train_u = train_u_sd_grid; train_u_pc = train_u_point_cloud
    test_a = test_a_sd_grid; test_u = test_u_sd_grid; test_u_pc = test_u_point_cloud

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u,
                                       test_u_pc),
        batch_size=batch_size, shuffle=False,
        generator=torch.Generator(device=device)
    )

    new_model = NUFNO2d(modes, modes, width,
                  in_channels=input_dim * n_subdomains,
                  out_channels=output_dim * n_subdomains).cuda()
    print(count_params(new_model))

    myloss = LpLoss(size_average=False)
    regloss = nn.MSELoss()
    y_normalizer.cuda()
    t0 = default_timer()

    new_model.load_state_dict(checkpoint['fno3d'])

    test_l2 = 0.0

    out = new_model(test_a_sd_grid).reshape(n_test,
                                        grid_shape[1], grid_shape[0],
                                        output_dim, n_subdomains)
    out = y_normalizer.decode(out).cpu().detach().numpy()
    out = np.transpose(out, (0, 2, 1, 3, 4))

    pred = np.zeros((n_test, max_n_points_sd,
                     output_dim, n_subdomains))
    for i in range(n_subdomains):
        bbox = bbox_sd[i]
        back_order = np.argsort(order_sd[i])
        _grid_shape = grid_shape[back_order]
        data = np.transpose(out[..., i],
                            (back_order + 1).tolist() + [3, 0])

        grid_x = np.linspace(bbox[0][0], bbox[0][1],
                             num=_grid_shape[0])
        grid_y = np.linspace(bbox[1][0], bbox[1][1],
                             num=_grid_shape[1])
        interp = RegularGridInterpolator(
            (grid_x, grid_y), data)
        data = interp(xy[indices_sd[i], :])
        data = np.transpose(data, (2, 0, 1))
        pred[:, :len(indices_sd[i]), :, i] = data

    pred = torch.tensor(pred).cpu()
    truth = test_u_point_cloud

    test_T_l2 = myloss(pred, truth).item()

    ## plot
    cnt = -1
    error = np.zeros(xy.shape[0])
    list_sd = [len(indices_sd[i]) for i in range(n_subdomains)]
    pred_ = pred[cnt]
    truth_ = truth[cnt]
    pred_list = []
    truth_list = []
    _xy = []
    indices_sd_revert_ = indices_sd
    for i in range(n_subdomains):
        pred_list.append(pred_[:list_sd[i], 0, i])
        truth_list.append(truth_[:list_sd[i], 0, i])
        _xy.append(xy[indices_sd[i], :])

    for i in range(1, n_subdomains):
        pred_list[0] = torch.cat((pred_list[0], pred_list[i]), dim=0)
        truth_list[0] = torch.cat((truth_list[0], truth_list[i]), dim=0)
        indices_sd_revert_[0] = np.concatenate([indices_sd_revert_[0], indices_sd_revert_[i]])

    indices_sd_revert = np.argsort(indices_sd_revert_[0])
    pred__ = pred_list[0][indices_sd_revert]
    truth__ = truth_list[0][indices_sd_revert]
    error = pred__ - truth__

    data = np.load(PATH, allow_pickle=True).item()

    X = xy[:, 0]
    Y = xy[:, 1]
    cell = data['cells']

    fig, ax = plt.subplots(1, 3, figsize=(30, 5))
    ax = ax.ravel()
    fig.set_tight_layout(True)
    triang = Triangulation(X, Y, cell)
    ax[0].tripcolor(triang, truth__, cmap='bwr')
    ax[0].set_title("Truth")
    fig.colorbar(ax[0].collections[0], ax=ax[0])
    ax[1].tripcolor(triang, pred__, cmap='bwr')
    ax[1].set_title("Preds")
    fig.colorbar(ax[1].collections[0], ax=ax[1])
    ax[2].tripcolor(triang, error, cmap='bwr')
    ax[2].set_title("Error")
    fig.colorbar(ax[2].collections[0], ax=ax[2])
    plt.savefig('result/NUFNO.png')

    myloss = MAELoss(size_average=False)
    MAE = []; MSE = []; num_nodes = X.shape[0];

    MAE_NUNO_high = []
    MAE_NUNO_low = []
    error_high = []
    error_low = []

    for cnt in range(n_test):
        error = np.zeros(xy.shape[0])
        pred_ = pred[cnt]
        truth_ = truth[cnt]
        pred_list = []
        truth_list = []
        for i in range(n_subdomains):
            pred_list.append(pred_[:list_sd[i], 0, i])
            truth_list.append(truth_[:list_sd[i], 0, i])
        for i in range(1, n_subdomains):
            pred_list[0] = torch.cat((pred_list[0], pred_list[i]), dim=0)
            truth_list[0] = torch.cat((truth_list[0], truth_list[i]), dim=0)
        pred__ = pred_list[0]
        truth__ = truth_list[0]
        error = pred__ - truth__
        MAE.append(myloss(pred__, truth__))

        truth_array = np.zeros(num_nodes)
        pred_array = np.zeros(num_nodes)
        indices_sd = tree.get_subdomain_indices()
        list_sd = [len(indices_sd[i]) for i in range(n_subdomains)]
        for i in range(n_subdomains):
            num_subnodes = list_sd[i]
            for j in range(num_subnodes):
                truth_array[indices_sd[i][j]] = pred[cnt, j, 0, i]
                pred_array[indices_sd[i][j]] = truth[cnt, j, 0, i]

        pred_array_high = pred_array[indices_all_list_high]
        pred_array_low = pred_array[indices_all_list_low]
        truth_array_high = truth_array[indices_all_list_high]
        truth_array_low = truth_array[indices_all_list_low]

        error_high += abs(pred_array_high-truth_array_high).tolist()
        error_low += abs(pred_array_low-truth_array_low).tolist()

    num_nodes_low = len(indices_all_list_low)
    num_nodes_high = len(indices_all_list_high)

    error_list = error_low + error_high
    print('=================================================')
    print('(high gradient) MAE:%.4f' % np.mean(error_high))
    print('(high gradient) s_MAE:%.4f' % np.sqrt(np.var(error_high)))
    print('=================================================')
    print('(low gradient) MAE:%.4f' % np.mean(error_low))
    print('(low gradient) s_MAE:%.4f' % np.sqrt(np.var(error_low)))
    print('=================================================')
    print('(Total) MAE:%.4f' % np.mean(error_list))
    print('(Total) s_MAE:%.4f' % np.sqrt(np.var(error_list)))
    print('=================================================')
