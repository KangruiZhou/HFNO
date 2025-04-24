
import sys
sys.path.append('model')
from timeit import default_timer
from util.utilities_HFNO import *
from util.Adam import Adam
import datetime
from matplotlib.tri import Triangulation
from scipy.interpolate import LinearNDInterpolator, \
    NearestNDInterpolator, RegularGridInterpolator, RBFInterpolator
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--PATH', type=str, default='data/data_case3.npy')
    parser.add_argument('--PATH_kdtree', type=str, default='result/saved_KDtree/KDTree_3.npy')
    parser.add_argument('--saved_model', type=str, default='result/saved_model/model3.pt')
    parser.add_argument('--variable', type=str, default='sigma_xy_2d')
    args = parser.parse_args()
    ################################################################
    # configs
    ################################################################
    modes_geo = 8
    width_geo = 20

    ## NUNO
    oversamp_ratio = 2.0  # used to calculate grid sizes
    input_dim = 3  # (u, v, w)
    output_dim = 1  # (T)
    reg_lambda = 5e-3
    boundry_lambda = 0.1

    modes_nuno = 8
    width_nuno = 20
    ################################################################
    # load data
    ################################################################
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d %H_%M_%S")

    oversamp_ratio = 2.0
    n_total = 6000
    n_train = 5000
    n_test = 1000
    batch_size = 20
    num_sample_points = 20

    ## decomposition
    tree = np.load(args.PATH_kdtree, allow_pickle=True).item()['Tree']
    bbox_sd = tree.get_subdomain_borders2()
    indices_sd = tree.get_subdomain_indices()

    ################################################################
    # GEO train setting
    ################################################################
    gradient_mean_nodes_array = np.array([np.mean(node.gradient_max_sd_array) for node in tree.nodes])
    high_gradient_mask = gradient_mean_nodes_array > np.mean(gradient_mean_nodes_array)
    high_gradient_index = np.where(high_gradient_mask)[0]

    high_gradient_borders = [bbox_sd[id] for id in high_gradient_index]
    low_gradient_borders = [bbox_sd[id] for id in list(range(len(indices_sd))) if id not in high_gradient_index]

    high_gradient_borders_2 = [tree.get_subdomain_borders2()[id] for id in high_gradient_index]
    low_gradient_borders_2 = [tree.get_subdomain_borders2()[id] for id in list(range(len(indices_sd))) if id not in high_gradient_index]

    # Load data
    data = np.load(args.PATH, allow_pickle=True).item()
    sigma = data['sigma_xy_2d']
    T = data['T_2d']
    input_x = data['xx_2d'][0]
    input_x_mesh = data['xx_2d'][0]
    input_y = data['yy_2d'][0]
    input_y_mesh = data['yy_2d'][0]
    cell = data['cells']
    n_nodes = sigma.shape[1]
    # input_point_cloud_mesh = np.stack((sigma, np.tile(input_x, (n_total, 1)), np.tile(input_y, (n_total, 1)), T), axis=2)
    ## 加入边界的插值数据
    intersections_dict, intersections_all = find_intersection(high_gradient_borders_2, low_gradient_borders_2)
    sample_points_subdomains = []
    n_sample_points_list = []
    sample_points_dict = dict()
    sample_points_all_list = []
    indices_sample_sd = []
    for intersections in intersections_dict:
        sample_points = sample_boundary(intersections_dict[intersections], num_sample_points)
        indices_sample_sd.append(np.array(range(len(sample_points))) + len(sample_points_all_list) + n_nodes)
        sample_points_all_list += sample_points.tolist()
        sample_points_subdomains.append(sample_points)
        n_sample_points_list.append(len(sample_points))
        sample_points_dict[f"{intersections}_sample_points"] = sample_points
        if len(sample_points) !=0:
            input_x = np.concatenate((input_x, sample_points[:, 0]), axis=0)
            input_y = np.concatenate((input_y, sample_points[:, 1]), axis=0)

    n_max_sample_points = max(n_sample_points_list)
    n_sample_points = sum(n_sample_points_list)
    indices_samples = np.array(range(n_sample_points)) + n_nodes
    sample_points_all = np.array(sample_points_all_list)

    input_x = np.tile(input_x, (n_total, 1))
    input_y = np.tile(input_y, (n_total, 1))
    input_xy = np.stack((input_x, input_y), axis=2)
    input_xy_mesh = np.stack((input_x_mesh, input_y_mesh), axis=1)
    T_s_val_mesh = np.stack((sigma, T), axis=2)
    T_s_val_mesh = np.transpose(T_s_val_mesh, (1, 2, 0))
    interp_rbf = RBFInterpolator(input_xy_mesh, T_s_val_mesh, kernel='cubic')
    T_s_val_samples = interp_rbf(sample_points_all)
    sigma_val_sample = T_s_val_samples[:, 0, :].T
    T_val_sample = T_s_val_samples[:, -1, :].T
    T = np.concatenate((T, T_val_sample), axis=1)
    sigma = np.concatenate((sigma, sigma_val_sample), axis=1)

    #
    input_point_cloud = np.stack((sigma, input_x, input_y, T), axis=2)
    input_point_cloud = input_point_cloud.reshape(input_point_cloud.shape[0], -1, 4)
    input_point_cloud = input_point_cloud[:n_total]
    input_point_cloud_mesh = input_point_cloud[:, :n_nodes, :]
    xy = np.stack((input_x, input_y), axis=2)  # with sample points
    xy = xy[0]

    ## 高低梯度区域处理
    indices_sd_Geo = []
    bbox_sd_Geo = []
    indices_all_GEO = np.array([])
    for index in high_gradient_index:
        indices_sd_Geo.append(indices_sd[index])
        bbox_sd_Geo.append(bbox_sd[index])
        indices_all_GEO = np.concatenate((indices_all_GEO, indices_sd[index]))
    indices_all_GEO = np.concatenate((indices_all_GEO, indices_samples))  # samples
    indices_all_list_GEO = list(indices_all_GEO.astype(int))
    input_xy_out = input_xy[:, indices_all_list_GEO, :]
    input_xy = torch.tensor(input_xy, dtype=torch.float)
    input_xy_out = torch.tensor(input_xy_out, dtype=torch.float)
    ################################################################
    # NUNO train setting
    ################################################################
    low_gradient_mask = ~high_gradient_mask
    low_gradient_index = np.where(low_gradient_mask)[0]
    indices_sd_NUNO = []
    bbox_sd_NUNO = []
    i = 0
    for index in low_gradient_index:
        if len(indices_sample_sd[i]) !=0:
            indices_sd_NUNO.append( np.concatenate((indices_sd[index], indices_sample_sd[i]), axis=0))
        else:
            indices_sd_NUNO.append(indices_sd[index])
        bbox_sd_NUNO.append(bbox_sd[index])
        i += 1
    # indices_sd_NUNO = np.concatenate((indices_sd_NUNO, indices_samples))  # samples NUNO
    n_subdomains = sum(low_gradient_mask)
    print("Start KD-Tree splitting...")
    t1 = default_timer()
    max_n_points_sd = np.max([len(indices_sd_NUNO[i])
                            for i in range(n_subdomains)])
    xy_sd = np.zeros((1, max_n_points_sd, n_subdomains, 2))
    input_point_cloud_sd = np.zeros((n_total,
                                    max_n_points_sd, input_point_cloud.shape[-1], n_subdomains))
    input_u_sd_mask = np.zeros((1, max_n_points_sd, 1, n_subdomains))
    grid_shape = [-1] * 2
    order_sd = []
    input_u_sd_mask_mesh = np.zeros((1, max_n_points_sd, 1, n_subdomains))
    sample_points_all = torch.tensor([])
    for i in range(n_subdomains):
        _xy = xy[indices_sd_NUNO[i], :]
        _min, _max = np.min(_xy, axis=0, keepdims=True), \
                    np.max(_xy, axis=0, keepdims=True)
        border = eval(list(intersections_dict.keys())[i])
        _min = np.array([border[0][0], border[1][0]])
        _max = np.array([border[0][1], border[1][1]])
        _xy = (_xy - _min) / (_max - _min) * 2 - 1  # normalization

        if len(sample_points_subdomains[i]) != 0:
            sample_points_subdomains[i] = (sample_points_subdomains[i] - _min) / (_max - _min)
            sample_points_all = torch.cat((sample_points_all,torch.tensor(sample_points_subdomains[i])),dim = 0)

        bbox = bbox_sd_NUNO[i]
        scales = [bbox[j][1] - bbox[j][0] for j in range(2)]
        order = np.argsort(scales)
        _xy = _xy[:, order]
        order_sd.append(order.tolist())
        # Calculate the grid shape
        _grid_shape = cal_grid_shape(
            oversamp_ratio * len(indices_sd_NUNO[i]), scales)
        _grid_shape.sort()
        grid_shape = np.maximum(grid_shape, _grid_shape)
        # Applying
        xy_sd[0, :len(indices_sd_NUNO[i]), i, :] = _xy
        input_point_cloud_sd[:, :len(indices_sd_NUNO[i]), :, i] = \
            input_point_cloud[:, indices_sd_NUNO[i], :]
        input_u_sd_mask[0, :len(indices_sd_NUNO[i]), 0, i] = 1.
        input_u_sd_mask_mesh[0, :(len(indices_sd_NUNO[i]) - len(indices_sample_sd[i])), 0, i] = 1.
    sample_points_all = sample_points_all.type(torch.float32)
    print(grid_shape)
    grid_shape = np.array(grid_shape)
    t2 = default_timer()
    print("Finish KD-Tree splitting, time elapsed: {:.1f}s".format(t2 - t1))

    # Interpolation from point cloud to uniform grid
    t1 = default_timer()
    print("Start interpolation...")
    input_sd_grid = []
    point_cloud = input_xy_mesh
    point_cloud_val = np.transpose(input_point_cloud_mesh, (1, 2, 0))
    interp_linear = LinearNDInterpolator(point_cloud, point_cloud_val)
    interp_nearest = NearestNDInterpolator(point_cloud, point_cloud_val)
    for i in range(n_subdomains):
        bbox = bbox_sd_NUNO[i]
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

    ## divide train and test dataset
    T = torch.tensor(T, dtype=torch.float)
    sigma = torch.tensor(sigma, dtype=torch.float)

    train_rr = T[:n_train]
    test_rr = T[-n_test:]
    train_s = sigma[:n_train]
    test_s = sigma[-n_test:]
    train_s = train_s[:, indices_all_list_GEO]
    test_s = test_s[:, indices_all_list_GEO]

    train_xy = input_xy[:n_train]
    test_xy = input_xy[-n_test:]
    train_xy_out = input_xy_out[:n_train]
    test_xy_out = input_xy_out[-n_test:]
    print(train_rr.shape, train_s.shape, train_xy.shape)

    model_GEO = GeoFNO2d(modes_geo, modes_geo, width_geo, in_channels=2, out_channels=1).cuda()
    model_iphi = IPHI(sigma.shape[1]).cuda()
    print(count_params(model_GEO), count_params(model_iphi))

    params = list(model_GEO.parameters()) + list(model_iphi.parameters())

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

    model_NUNO = NUFNO2d(modes_nuno, modes_nuno, width_nuno,
                        in_channels=input_dim * n_subdomains,
                        out_channels=output_dim * n_subdomains).cuda()

    print(count_params(model_NUNO))

    # normalize
    train_x_max = train_xy[0, :, 0].max()
    train_x_min = train_xy[0, :, 0].min()
    train_y_max = train_xy[0, :, 1].max()
    train_y_min = train_xy[0, :, 1].min()
    array1 = torch.tensor([train_x_min, train_y_min])
    array2 = torch.tensor([train_x_max - train_x_min, train_y_max - train_y_min])

    train_xy = (train_xy - array1) / array2
    test_xy = (test_xy - array1) / array2
    train_xy_out = (train_xy_out - array1) / array2
    test_xy_out = (test_xy_out - array1) / array2

    rr_normalizer = UnitGaussianNormalizer(train_rr)
    train_rr = rr_normalizer.encode(train_rr)
    test_rr = rr_normalizer.encode(test_rr)

    s_normalizer = UnitGaussianNormalizer(train_s)
    ################################################################
    # training and evaluation
    ################################################################
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_rr, train_s, train_xy, train_xy_out, train_a_sd_grid, train_u_sd_grid,
                                    train_u_point_cloud), batch_size=batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_rr, test_s, test_xy, test_xy_out, test_a_sd_grid, test_u_sd_grid,
                                    test_u_point_cloud), batch_size=batch_size,
        shuffle=False)

    checkpoint = torch.load(args.saved_model)
    model_GEO.load_state_dict(checkpoint['fno2d'])
    model_iphi.load_state_dict(checkpoint['IPHI'])
    model_NUNO.load_state_dict(checkpoint['fno3d'])

    ###
    MAE_loss = MAELoss(size_average=False)
    MRE_loss = MRELoss(size_average=False)
    myloss = LpLoss(size_average=False)
    regloss = nn.MSELoss()
    y_normalizer.cuda()
    s_normalizer.cuda()
    t00 = default_timer()
    test_l2_GEO = 0.0
    test_l2_NUNO = 0.0
    test_l2_boundry = 0.
    MAE_GEO = []
    MRE_GEO = []
    MAE_NUNO = []
    MRE_NUNO = []
    MAE_Boundary = []
    MAE_Boundary_interpolate = []
    MRE_Boundary = []
    with torch.no_grad():
        for rr, sigma, mesh, mesh_out, x, y, y_nuno in test_loader:
            rr, sigma, mesh, mesh_out = rr.cuda(), sigma.cuda(), mesh.cuda(), mesh_out.cuda()

            ## GEO
            out_Geo = model_GEO(mesh, code=rr, iphi=model_iphi, x_out=mesh_out)
            out_Geo = s_normalizer.decode(out_Geo.reshape(batch_size, -1))
            out_Geo_mesh = out_Geo[:, :(len(indices_all_GEO)-n_sample_points)]
            sigma_geo_mesh = sigma[:, :(len(indices_all_GEO)-n_sample_points)]
            MAE_GEO += abs(out_Geo.view(-1) - sigma.view(-1)).tolist()
            MRE_GEO += abs((out_Geo.view(-1) - sigma.view(-1)) / sigma.view(-1)).tolist()
            test_l2_GEO += myloss(out_Geo_mesh.view(batch_size, -1), sigma_geo_mesh.view(batch_size, -1)).item()

            # nuno
            out_nuno = model_NUNO(x).reshape(batch_size,
                                            grid_shape[1], grid_shape[0],
                                            output_dim, n_subdomains)
            out_nuno = y_normalizer.decode(out_nuno)
            test_l2_NUNO += myloss(out_nuno, y).item()
            # Boundary
            out_nuno = out_nuno.permute(0, 4, 3, 1, 2) \
                .reshape(-1, output_dim,
                        grid_shape[1], grid_shape[0])

            u = F.grid_sample(input=out_nuno, grid=xy_sd,
                            padding_mode='border', align_corners=False)
            out_nuno = u.squeeze(-1).squeeze(-1).permute(0, 2, 1) \
                .reshape(batch_size, n_subdomains, -1, output_dim) \
                .permute(0, 2, 3, 1)
            out_nuno = out_nuno * input_u_sd_mask
            out_nuno_boundary_ = torch.tensor([]).cuda()
            for i in range(n_subdomains):
                out_nuno_boundary_ = torch.cat(
                    (out_nuno_boundary_, out_nuno[:, (len(indices_sd_NUNO[i]) - len(indices_sample_sd[i])):
                                                    len(indices_sd_NUNO[i]), :, i]), dim=1)
            out_nuno_boundary = out_nuno_boundary_.squeeze(-1)
            ## geo
            out_Geo_boundary = out_Geo[:, -n_sample_points:]
            sigma_boundary = sigma[:, -n_sample_points:]
            MAE_Boundary += abs(out_nuno_boundary- out_Geo_boundary).tolist()
            out_boundary = 0.5 * (out_nuno_boundary + out_Geo_boundary)
            MAE_Boundary_interpolate += abs(out_boundary-sigma_boundary).tolist()
            test_l2_boundry = myloss(out_nuno_boundary, out_Geo_boundary).item()

        ## NUNO
        out_nuno = model_NUNO(test_a_sd_grid).reshape(n_test,
                                                    grid_shape[1], grid_shape[0],
                                                    output_dim, n_subdomains)
        out_nuno = y_normalizer.decode(out_nuno).cpu().numpy()


        out_nuno = np.transpose(out_nuno, (0, 2, 1, 3, 4))

        pred = np.zeros((n_test, max_n_points_sd,
                        output_dim, n_subdomains))
        for i in range(n_subdomains):
            bbox = bbox_sd_NUNO[i]
            back_order = np.argsort(order_sd[i])
            _grid_shape = grid_shape[back_order]
            data = np.transpose(out_nuno[..., i],
                                (back_order + 1).tolist() + [3, 0])
            grid_x = np.linspace(bbox[0][0], bbox[0][1],
                                num=_grid_shape[0])
            grid_y = np.linspace(bbox[1][0], bbox[1][1],
                                num=_grid_shape[1])
            interp = RegularGridInterpolator(
                (grid_x, grid_y), data)
            data = interp(xy[indices_sd_NUNO[i], :])
            data = np.transpose(data, (2, 0, 1))
            pred[:, :len(indices_sd_NUNO[i]), :, i] = data
        pred = torch.tensor(pred).cpu()
        truth = test_u_point_cloud

        list_sd = [len(indices_sd_NUNO[i]-len(indices_sample_sd[i])) for i in range(n_subdomains)]  # sample
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
            # error_NUNO += abs(pred__ - truth__).tolist()
            MAE_NUNO += abs(pred__ - truth__).tolist()
            MRE_NUNO += abs((pred__.view(-1) - truth__.view(-1)) / truth__.view(-1)).tolist()

            # else:
            #     t_Boundry = 0
    test_l2_GEO = test_l2_GEO / n_test
    test_l2_NUNO = test_l2_NUNO / n_test
    test_l2_boundry = test_l2_boundry / n_test

    MAE_total = MAE_GEO + MAE_NUNO
    MRE_total = MRE_GEO + MRE_NUNO
    print('=================================================')
    print('(GEO) MAE:%.4f' % np.mean(MAE_GEO))
    print('(GEO) test_l2_GEO:%.4f' % test_l2_GEO)
    print('(GEO) Max absolute error:%.4f' % np.max(MAE_GEO))
    print('(GEO) MRE:%.8f' % np.mean(MRE_GEO))
    print('(GEO) s_MAE:%.4f' % np.sqrt(np.var(MAE_GEO)))
    print('=================================================')
    print('(NUNO) MAE:%.4f' % np.mean(MAE_NUNO))
    print('(NUNO) test_l2_NUNO:%.4f' % test_l2_NUNO)
    print('(NUNO) MRE:%.4f' % np.mean(MRE_NUNO))
    print('(NUNO) Max absolute error:%.4f' % np.max(MAE_NUNO))
    print('(NUNO) s_MAE:%.4f' % np.sqrt(np.var(MAE_NUNO)))
    print('=================================================')
    print('(Boundary_interpolation) MAE:%.4f' % np.mean(MAE_Boundary_interpolate))
    print('(Boundary_interpolation) Max absolute error:%.4f' % np.max(MAE_Boundary_interpolate))
    print('(Boundary_interpolation) s_MAE:%.4f' % np.sqrt(np.var(MAE_Boundary_interpolate)))
    print('=================================================')
    print('(Boundary) MAE:%.4f' % np.mean(MAE_Boundary))
    print('(Boundary) test_l2_boundry:%.4f' % test_l2_boundry)
    print('(Boundary) Max absolute error:%.4f' % np.max(MAE_Boundary))
    print('(Boundary) s_MAE:%.4f' % np.sqrt(np.var(MAE_Boundary)))
    print('=================================================')
    print('(Total) MAE:%.4f' % np.mean(MAE_total))
    print('(Total) MRE:%.4f' % np.mean(MRE_total))
    print('(Total) Max absolute error:%.4f' % np.max(MAE_total))
    print('(Total) s_MAE:%.4f' % np.sqrt(np.var(MAE_total)))
    print('=================================================')


    ## plot

    num_nodes_Total = n_nodes
    Truth = np.zeros(num_nodes_Total)
    Pred  = np.zeros(num_nodes_Total)
    X = np.zeros(num_nodes_Total)
    Y = np.zeros(num_nodes_Total)
    cnt= -1

    list_sd = [(len(indices_sd_NUNO[i])-len(indices_sample_sd[i])) for i in range(n_subdomains)]
    for i in range(n_subdomains):
        num_subnodes = list_sd[i]
        for j in range(num_subnodes):
            Pred[indices_sd_NUNO[i][j]] = pred[cnt, j, 0, i]
            Truth[indices_sd_NUNO[i][j]] = truth[cnt, j, 0, i]

    truth = sigma[cnt].squeeze().detach().cpu().numpy()
    pred = out_Geo[cnt].squeeze().detach().cpu().numpy()
    for i in range(len(indices_all_list_GEO)-n_sample_points):
        Truth[indices_all_list_GEO[i]] = truth[i]
        Pred[indices_all_list_GEO[i]] = pred[i]
    t1 = default_timer()

    data = np.load(args.PATH, allow_pickle=True).item()
    X = data['xx_2d'][0]
    Y = data['yy_2d'][0]

    fig, ax = plt.subplots(1, 3, figsize=(25, 5))
    triang = Triangulation(X, Y, cell)
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[2].set_aspect('equal')

    map_min = Truth.min()
    map_max = Truth.max()

    ax[0].tripcolor(triang, Truth, cmap='bwr', vmin=map_min, vmax=map_max)
    ax[1].tripcolor(triang, Pred, cmap='bwr', vmin=map_min, vmax=map_max)
    im = ax[2].tripcolor(triang, Truth - Pred, cmap='bwr', vmin=map_min, vmax=map_max)

    ax[0].set_title("Ground Truth", fontsize=18)
    ax[1].set_title("Prediction", fontsize=18)
    ax[2].set_title("Error", fontsize=18)

    cax = add_right_cax(ax[2], pad=0.01, width=0.01)
    cbar = fig.colorbar(im, cax=cax)

    fig.savefig("result/" + args.saved_model.split('/')[-1][:-3] + ".png")


