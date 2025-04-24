import sys 
sys.path.append('')
sys.path.append('model')
from util.utilities_GeoFNO import *
from util.Adam import Adam
import datetime
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
from util.utilities_HFNO import UnitGaussianNormalizer, cal_metric
import torch.nn.functional as F
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

################################################################
# configs
################################################################
Ntotal = 6000
ntrain = 5000
ntest = 1000

batch_size = 20
learning_rate_fno = 0.001
learning_rate_iphi = 0.0001

epochs = 501

modes = 12
width = 22

################################################################
# load data and data normalization
################################################################
now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d %H:%M:%S")

PATH = 'data/data_case3.npy'
PATH_mcts='result/saved_KDtree/KDTree_3.npy'
load_name = 'result/saved_model/model_GeoFNO.pt'

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
# Load data
data = np.load(PATH, allow_pickle=True).item()
input_s = data['sigma_xy_2d']
input_rr = data['T_2d']
input_x = data['xx_2d']
input_x_mesh = data['xx_2d'][0]
input_y = data['yy_2d']
input_y_mesh = data['yy_2d'][0]
cell = data['cells']
n_nodes = input_s.shape[1]
input_xy = np.stack((input_x, input_y), axis=2)
input_rr = torch.tensor(input_rr, dtype=torch.float)
input_s = torch.tensor(input_s, dtype=torch.float)
input_xy = torch.tensor(input_xy, dtype=torch.float)

train_rr = input_rr[:ntrain]
test_rr = input_rr[-ntest:]
train_s = input_s[:ntrain]
test_s = input_s[-ntest:]
train_xy = input_xy[:ntrain]
test_xy = input_xy[-ntest:]

rr_normalizer = UnitGaussianNormalizer(train_rr)
train_rr = rr_normalizer.encode(train_rr)
test_rr = rr_normalizer.encode(test_rr)

s_normalizer = UnitGaussianNormalizer(train_s)

print(train_rr.shape, train_s.shape, train_xy.shape)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_rr, train_s, train_xy), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_rr, test_s, test_xy), batch_size=batch_size,
                                          shuffle=False)

################################################################
# training and evaluation
################################################################
model = GeoFNO2d(modes, modes, width, in_channels=2, out_channels=1).cuda()
model_iphi = IPHI(input_s.shape[1]).cuda()
print(count_params(model)+count_params(model_iphi))
checkpoint = torch.load(load_name)
model.load_state_dict(checkpoint['fno2d'])
model_iphi.load_state_dict(checkpoint['IPHI'])

params = list(model.parameters()) + list(model_iphi.parameters())

MAE_loss = MAELoss(size_average=False)
MRE_loss = MRELoss(size_average=False)
myloss = LpLoss(size_average=False)

num_nodes = train_rr.shape[1]
s_normalizer.cuda()
test_l2 = 0.0
regloss = nn.MSELoss()
MAE_GEO = []; MRE_GEO = [];

MAE_GEO_high = []
MAE_GEO_low = []
MRE_GEO_high = []
MRE_GEO_low = []
error_high = []
error_low = []
num_MRE_GEO = 0
test_l2 = 0
with torch.no_grad():
    for rr, sigma, mesh in test_loader:
        mesh_high = mesh[:, indices_all_list_high]
        mesh_low = mesh[:, indices_all_list_low]
        sigma_high = sigma[:, indices_all_list_high]
        sigma_low = sigma[:, indices_all_list_low]

        rr, sigma, mesh = rr.cuda(), sigma.cuda(), mesh.cuda()
        mesh_low, mesh_high = mesh_low.cuda(), mesh_high.cuda()
        sigma_low, sigma_high = sigma_low.cuda(), sigma_high.cuda()

        index = abs(sigma) > 0
        index_high = abs(sigma_high) > 0
        index_low = abs(sigma_low) > 0

        ## GEO
        out_Geo = model(mesh, code=rr, iphi=model_iphi)
        out_Geo = s_normalizer.decode(out_Geo.reshape(batch_size,-1))
        test_l2 += myloss(out_Geo.view(batch_size, -1), sigma.view(batch_size, -1))
        # MAE_GEO.append(MAE_loss(out_Geo[index], sigma[index]))
        # MRE_GEO.append(MRE_loss(out_Geo[index], sigma[index]))
        MAE_GEO += abs(out_Geo.view(-1)-sigma.view(-1)).tolist()
        MRE_GEO += abs((out_Geo[index].view(-1)-sigma[index].view(-1))/sigma[index].view(-1)).tolist()
        num_MRE_GEO += torch.sum(index)


        ## high gradient
        out_high = model(mesh, code=rr, iphi=model_iphi, x_out=mesh_high)
        out_high = out_high.reshape(batch_size,-1)
        out_high = F.pad(out_high, (0, out_Geo.shape[1]-out_high.shape[1]))
        out_high = s_normalizer.decode(out_high)
        out_high = out_high[:,:mesh_high.shape[1]]

        out_high = out_Geo[:, indices_all_list_high]
        MAE_GEO_high.append(MAE_loss(out_high[index_high], sigma_high[index_high]))
        MRE_GEO_high.append(MRE_loss(out_high[index_high], sigma_high[index_high]))
        error_high += abs(out_high.reshape(-1) - sigma_high.reshape(-1)).tolist()
        ## low gradient
        # out_low = model(mesh, code=rr, iphi=model_iphi, x_out=mesh_low)
        # out_low = out_low.reshape(batch_size,-1)
        # out_low = F.pad(out_low, (0, out_Geo.shape[1]-out_low.shape[1]))
        # out_low = s_normalizer.decode(out_low)
        # out_low = out_low[:,:mesh_low.shape[1]]
        out_low = out_Geo[:, indices_all_list_low]
        MAE_GEO_low.append(MAE_loss(out_low[index_low], sigma_low[index_low]))
        MRE_GEO_low.append(MRE_loss(out_low[index_low], sigma_low[index_low]))
        error_low += abs(out_low.reshape(-1) - sigma_low.reshape(-1)).tolist()
num_nodes_low = len(indices_all_list_low)
num_nodes_high = len(indices_all_list_high)
test_l2 = test_l2 / ntest
print(f"test_l2:{test_l2}")
#
# for i in range(len(indices_all_list_low)):
#     MAE_GEO_low.append(MAE_GEO[indices_all_list_low[i]])
#     MRE_GEO_low.append(MRE_GEO[indices_all_list_low[i]])
# for i in range(len(indices_all_list_high)):
#     MAE_GEO_high.append(MAE_GEO[indices_all_list_high[i]])
#     MRE_GEO_high.append(MRE_GEO[indices_all_list_high[i]])
error_list = error_low + error_high
print('=================================================')
print('(GEO) MAE:%.4f' % np.mean(error_high))
print('(GEO) s_MAE:%.4f' % np.sqrt(np.var(error_high)))
print('=================================================')
print('(NUNO) MAE:%.4f' % np.mean(error_low))
print('(NUNO) s_MAE:%.4f' % np.sqrt(np.var(error_low)))
print('=================================================')
print('(Total) MAE:%.4f' % np.mean(error_list))
print('(Total) s_MAE:%.4f' % np.sqrt(np.var(error_list)))
print('=================================================')


# print('=================================================')
# print('(GEO) MAE:%.4f' % np.mean(MAE_GEO))
# print('(GEO) s_MAE:%.4f' % np.var(MAE_GEO))
# print('(GEO) MRE:%.8f' % np.mean(MRE_GEO))
# print('(GEO) s_MRE:%.4f' % np.var(MRE_GEO))
# print('=================================================')
meanMAE_GEO, meanMRE_GEO, sigma_MAE_GEO, sigma_MRE_GEO = \
    cal_metric(MAE_GEO, MRE_GEO, num_nodes, ntest)
meanMAE_high, meanMRE_high, sigma_MAE_high, sigma_MRE_high = \
    cal_metric(MAE_GEO_high, MRE_GEO_high, num_nodes_high, ntest)
meanMAE_low, meanMRE_low, sigma_MAE_low, sigma_MRE_low = \
    cal_metric(MAE_GEO_low, MRE_GEO_low, num_nodes_low, ntest)
print('=======================Total==========================')
print('(Total) MAE:%.4f' % meanMAE_GEO)
print('(Total) MRE:%.4f' % meanMRE_GEO)
# print('(Total) MRE_np:%.4f' % np.mean(MRE))
print('(Total) sigma_MAE:%.4f' % sigma_MAE_GEO)
print('(Total) sigma_MRE:%.4f' % sigma_MRE_GEO)
print('=================================================')
print('================high gradient==========================')
print('(high gradient) MAE:%.4f' % meanMAE_high)
print('(high gradient) MRE:%.4f' % meanMRE_high)
print('(high gradient) MRE_np:%.4f' % meanMRE_high)
print('(high gradient) sigma_MAE:%.4f' % sigma_MAE_high)
print('(high gradient) sigma_MRE:%.4f' % sigma_MRE_high)
print('=================================================')
print('================low gradient==========================')
print('(low gradient) MAE:%.4f' % meanMAE_low)
print('(low gradient) MRE:%.4f' % meanMRE_low)
print('(low gradient) sigma_MAE:%.4f' % sigma_MAE_low)
print('(low gradient) sigma_MRE:%.4f' % sigma_MRE_low)
print('=================================================')

XY = mesh[-1].squeeze().detach().cpu().numpy()
truth = sigma[-1].squeeze().detach().cpu().numpy()
pred = out_Geo[-1].squeeze().detach().cpu().numpy()

cell = data['cells']


fig, ax = plt.subplots(1, 3, figsize=(30, 5))
ax = ax.ravel()
fig.set_tight_layout(True)

X = XY[:, 0]
Y = XY[:, 1]
triang = Triangulation(X, Y, cell)

# filename = '/mnt/jfs/zkr/HFNO/baseline/GEO_error_145degree.npy'
# data = dict(triang=triang,Truth=truth, Pred=pred)
# np.save(filename, data)

ax[0].tripcolor(triang, truth, cmap='bwr')
# ax[0].set_title("Truth Y displacement")
# ax[0].set_title("Truth $\sigma_{yy}$")
fig.colorbar(ax[0].collections[0], ax=ax[0])
# ax[1].tripcolor(triang, array_preds, cmap='bwr', vmin=vvmin, vmax=vvmax)
ax[1].tripcolor(triang, pred,  cmap='bwr')
# ax[1].set_title("Preds Y displacement")
# ax[1].set_title("Preds $\sigma_{yy}$")
fig.colorbar(ax[1].collections[0], ax=ax[1])
ax[2].tripcolor(triang, truth - pred, cmap='bwr')
# ax[2].tripcolor(triang, Error_, cmap='bwr', vmin=vvmin, vmax=vvmax)
# ax[2].set_title("Error $\sigma_{yy}$")
fig.colorbar(ax[2].collections[0], ax=ax[2])
fig.savefig(f"result/GEOFNO.png")
# fig.show()
