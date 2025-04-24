import sys 
sys.path.append('.')
from timeit import default_timer
from util.utilities_GeoFNO import *
from util.Adam import Adam
import datetime
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
from util.utilities_HFNO import UnitGaussianNormalizer
import wandb

seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
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

weight_decay = 1e-4
step_size = 100
gamma = 0.5


################################################################
# load data and data normalization
################################################################
now = datetime.datetime.now()
date = now.strftime("%Y_%m_%d %H_%M_%S")

PATH = 'data/data_case3.npy'
PATH_mcts='result/saved_KDtree/KDTree_3.npy'
name_ = PATH.split('/')[-1][:-4]
filename = f'result/saved_model/model_GeoFNO.pt'

wandb.init(project='HFNO',
           id=f"GEO_{Ntotal}_{name_}_{batch_size}_{epochs}_{learning_rate_fno}_{learning_rate_iphi}_{step_size}_{gamma}_{weight_decay}",
           config={
    "modes_geo": modes,
    "width_geo": width
})

# Load data
data = np.load(PATH, allow_pickle=True).item()
input_s = data['sigma_xy_2d']
input_rr = data['T_2d']
input_x = data['xx_2d']
input_y = data['yy_2d']
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
print(count_params(model), count_params(model_iphi))

params = list(model.parameters()) + list(model_iphi.parameters())
optimizer_fno = Adam(model.parameters(), lr=learning_rate_fno, weight_decay=weight_decay)
scheduler_fno = torch.optim.lr_scheduler.StepLR(optimizer_fno, step_size=step_size, gamma=gamma)

optimizer_iphi = Adam(model_iphi.parameters(), lr=learning_rate_iphi, weight_decay=weight_decay)
scheduler_iphi = torch.optim.lr_scheduler.StepLR(optimizer_iphi, step_size=step_size, gamma=gamma)

MAE_loss = MAELoss(size_average=False)
MRE_loss = MRELoss(size_average=False)
myloss = LpLoss(size_average=False)
t00 = default_timer()
num_nodes = train_rr.shape[1]
s_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0.
    test_l2 = 0.
    for rr, sigma, mesh in train_loader:
        rr, sigma, mesh = rr.cuda(), sigma.cuda(), mesh.cuda()
        optimizer_fno.zero_grad()
        optimizer_iphi.zero_grad()
        out = model(mesh, code=rr, iphi=model_iphi)
        out = s_normalizer.decode(out.reshape(batch_size,-1))
        loss = myloss(out.view(batch_size, -1), sigma.view(batch_size, -1))
        loss.backward()
        optimizer_fno.step()
        optimizer_iphi.step()
        train_l2 += loss.item()

    scheduler_fno.step()
    scheduler_iphi.step()

    model.eval()

    with torch.no_grad():
        for rr, sigma, mesh in test_loader:
            rr, sigma, mesh = rr.cuda(), sigma.cuda(), mesh.cuda()
            optimizer_fno.zero_grad()
            optimizer_iphi.zero_grad()

            out_Geo = model(mesh, code=rr, iphi=model_iphi)
            out_Geo = s_normalizer.decode(out_Geo.reshape(batch_size,-1))
            test_l2 += myloss(out_Geo.view(batch_size, -1), sigma.view(batch_size, -1))

    train_l2 = train_l2 / ntrain
    test_l2 = test_l2 / ntest

    t2 = default_timer()
    print("[Epoch {}] GEO: time:{:.1f}s L2: {:>4e} Test_L2: {:>4e}"
          .format(ep, t2 - t1, train_l2, test_l2))
    wandb.log({"train_l2_GEO": train_l2, "test_l2_GEO": test_l2, 'cost_time': t2 - t00})
    
    if ep%100==0 and ep !=0 :
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

        ax[0].tripcolor(triang, truth, cmap='bwr')
        ax[0].set_title("Truth $\sigma_{yy}$")
        fig.colorbar(ax[0].collections[0], ax=ax[0])
        ax[1].tripcolor(triang, pred, cmap='bwr')
        ax[1].set_title("Preds $\sigma_{yy}$")
        fig.colorbar(ax[1].collections[0], ax=ax[1])
        ax[2].tripcolor(triang, truth - pred, cmap='bwr')
        ax[2].set_title("Error $\sigma_{yy}$")
        fig.colorbar(ax[2].collections[0], ax=ax[2])
        # fig.show()
        state = {'fno2d': model.state_dict(),
                 'IPHI': model_iphi.state_dict()}
        torch.save(state, filename)
t01 = default_timer()
print('Total train time:%.1f s'%(t01-t00))


Error_GEO = []
with torch.no_grad():
    for rr, sigma, mesh in test_loader:
        rr, sigma, mesh= rr.cuda(), sigma.cuda(), mesh.cuda()
        optimizer_fno.zero_grad()
        optimizer_iphi.zero_grad()

        out_Geo = model(mesh, code=rr, iphi=model_iphi)
        for i in range(batch_size*num_nodes):
            Error_GEO+= [abs(out_Geo.view(-1)[i]-sigma.view(-1)[i]).tolist()]
print('=================================================')
print('(GEO) MAE:%.4f' % np.mean(Error_GEO))
print('(GEO) sigma_Error:%.4f' % np.var(Error_GEO))
print('=================================================')
