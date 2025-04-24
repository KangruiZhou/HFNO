import sys 
sys.path.append('.')
from timeit import default_timer
import torch.nn.functional as F
from torch.optim import Adam
from util.kdtree_NUFNO.tree import *
from util.utilities_NUFNO import *
from scipy.interpolate import LinearNDInterpolator, \
    NearestNDInterpolator, RegularGridInterpolator
import datetime

import wandb
################################################################
# Configs
################################################################
seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
# Dataset params
n_train = 5000
n_test = 1000
n_total = n_train + n_test

# FNO configs
modes = 8
width = 20

# Training params
batch_size = 20
learning_rate = 0.001
weight_decay = 1e-4
epochs = 501
reg_lambda = 5e-3
step_size = 100
gamma = 0.5

# Grid params
oversamp_ratio = 2.0  # used to calculate grid sizes
input_dim = 3  # (u, v, w)
output_dim = 1  # (T)

# K-D tree params
n_subdomains = 16
device = 'cuda'
################################################################
# Training and evaluation
################################################################
def main(train_a, train_u, train_u_pc, test_a, test_u, test_u_pc):
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u,
                                       train_u_pc),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator(device=device)
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u,
                                       test_u_pc),
        batch_size=batch_size, shuffle=False,
        generator=torch.Generator(device=device)
    )

    model = NUFNO2d(modes, modes, width,
                  in_channels=input_dim * n_subdomains,
                  out_channels=output_dim * n_subdomains).cuda()
    print(count_params(model))
    optimizer = Adam(model.parameters(),
                     lr=learning_rate, weight_decay=weight_decay,capturable=True)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma)

    myloss = LpLoss(size_average=False)
    regloss = nn.MSELoss()
    y_normalizer.cuda()
    t00 = default_timer()

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_tot_loss = 0.0
        for x, y, y_pc in train_loader:
            optimizer.zero_grad()
            out = model(x).reshape(batch_size,
                                   grid_shape[1], grid_shape[0],
                                   output_dim, n_subdomains)
            out = y_normalizer.decode(out)
            loss1 = myloss(out, y)

            # Interpolation (from grids to point cloud)
            out = out.permute(0, 4, 3, 1, 2) \
                .reshape(-1, output_dim,
                         grid_shape[1], grid_shape[0])
            u = F.grid_sample(input=out, grid=xy_sd,
                              padding_mode='border', align_corners=False)
            # Output shape: (batch * n_subdomains, output_dim,
            #   n_points_sd_padded, 1, 1)
            out = u.squeeze(-1).squeeze(-1).permute(0, 2, 1) \
                .reshape(batch_size, n_subdomains, -1, output_dim) \
                .permute(0, 2, 3, 1)
            # Output shape: (batch_size, n_points_sd_padded,
            #   output_dim, n_subdomains)

            out = out * input_u_sd_mask
            loss = loss1 + reg_lambda * regloss(out, y_pc)
            loss.backward()

            optimizer.step()
            train_l2 += loss1.item()
            train_tot_loss += loss.item()

        scheduler.step()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y, _ in test_loader:
                out = model(x).reshape(batch_size,
                                       grid_shape[1], grid_shape[0],
                                       output_dim, n_subdomains)
                out = y_normalizer.decode(out)

                test_l2 += myloss(out, y).item()

        train_l2 /= n_train
        test_l2 /= n_test

        t2 = default_timer()
        print("[Epoch {}] NUNO: Time: {:.1f}s L2: {:>4e} Test_L2: {:>4e}"
              .format(ep, t2 - t1, train_l2, test_l2))
        wandb.log({'train_l2_NUNO': train_l2, 'test_l2_NUNO': test_l2,'cost_time': t2 - t00})
    
        if ep%100==0 and ep!=0:
            with torch.no_grad():
                out = model(test_a_sd_grid).reshape(n_test,
                                                    grid_shape[1], grid_shape[0],
                                                    output_dim, n_subdomains)
                out = y_normalizer.decode(out).cpu().numpy()
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


                Error_NUFNO=[]
                for num in range(n_test):
                    for i in range(len(indices_sd)):
                        for j in range(len(indices_sd[i])):
                            Error_NUFNO.append(abs(pred[num,j,0,i]-truth[num,j,0,i]))
                print('MAE:%.4f' % np.mean(Error_NUFNO))
                print('s_E:%.4f' % np.var(Error_NUFNO))
                state = {'fno3d': model.state_dict()
                         }
                torch.save(state, filename+f'_{ep}'+'.pt')

    t01 = default_timer()
    print('Total train time:%.1f s'%(t01-t00))

now = datetime.datetime.now()
date = now.strftime("%Y_%m_%d %H_%M_%S")

PATH = 'data/data_case3.npy'
PATH_mcts='result/saved_KDtree/KDTree_3.npy'
name_ = PATH.split('/')[-1][:-4]
filename = f'result/saved_model/model_NUNO.pt'

wandb.init(project='HFNO',
           id=f"NUNO_{n_total}_{name_}_{modes}_{width}_{batch_size}_{epochs}_{learning_rate}_{step_size}_{gamma}_{weight_decay}",
           config={
    "modes_geo": modes,
    "width_geo": width
})

################################################################
# Load data and preprocessing
################################################################
data = np.load(PATH, allow_pickle=True).item()
sigma = data['sigma_xy_2d']
T = data['T_2d']
input_x = data['xx_2d']
input_y = data['yy_2d']

input_point_cloud = np.stack((sigma, input_x, input_y, T), axis=2)
input_point_cloud = input_point_cloud.reshape(input_point_cloud.shape[0], -1, 4)
sigma_mean = np.mean(input_point_cloud[:n_train, :])
sigma_std = np.std(input_point_cloud[:n_train, :])
input_point_cloud = input_point_cloud[:n_total]

xy = np.stack((input_x, input_y), axis=2)
xy = xy[0]

num_nodes = xy.shape[0]

print("Start KD-Tree splitting...")
t1 = default_timer()
point_cloud = xy.tolist()
# Use kd-tree to generate subdomain division
tree = KDTree(
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
# point_cloud_val = point_cloud_val.reshape(point_cloud_val.shape[0], -1)
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
if __name__ == "__main__":
    main(train_a_sd_grid, train_u_sd_grid,
        train_u_point_cloud, test_a_sd_grid,
        test_u_sd_grid, test_u_point_cloud)


