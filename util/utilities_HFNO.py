
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import operator
from functools import reduce
from scipy.interpolate import RBFInterpolator
from scipy.misc import derivative
import matplotlib as mpl
from tqdm import tqdm

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
################################################################
# fourier layer
################################################################
class GeoSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, s1=32, s2=32):
        super(GeoSpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.s1 = s1
        self.s2 = s2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, u, x_in=None, x_out=None, iphi=None, code=None):
        batchsize = u.shape[0]

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        if x_in == None:
            u_ft = torch.fft.rfft2(u)
            s1 = u.size(-2)
            s2 = u.size(-1)
        else:
            u_ft = self.fft2d(u, x_in, iphi, code)
            s1 = self.s1
            s2 = self.s2

        # Multiply relevant Fourier modes
        # print(u.shape, u_ft.shape)
        factor1 = self.compl_mul2d(u_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        factor2 = self.compl_mul2d(u_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        if x_out == None:
            out_ft = torch.zeros(batchsize, self.out_channels, s1, s2 // 2 + 1, dtype=torch.cfloat, device=u.device)
            out_ft[:, :, :self.modes1, :self.modes2] = factor1
            out_ft[:, :, -self.modes1:, :self.modes2] = factor2
            u = torch.fft.irfft2(out_ft, s=(s1, s2))
        else:
            out_ft = torch.cat([factor1, factor2], dim=-2)
            u = self.ifft2d(out_ft, x_out, iphi, code)

        return u

    def fft2d(self, u, x_in, iphi=None, code=None):
        # u (batch, channels, n)
        # x_in (batch, n, 2) locations in [0,1]*[0,1]
        # iphi: function: x_in -> x_c

        batchsize = x_in.shape[0]
        N = x_in.shape[1]
        device = x_in.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=self.modes2, step=1), \
                            torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        # print(x_in.shape)
        if iphi == None:
            x = x_in
        else:
            x = iphi(x_in, code)

        # print(x.shape)
        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[...,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[...,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(-1j * 2 * np.pi * K).to(device)

        # Y (batch, channels, N)
        u = u + 0j
        Y = torch.einsum("bcn,bnxy->bcxy", u, basis)
        return Y

    def ifft2d(self, u_ft, x_out, iphi=None, code=None):
        batchsize = x_out.shape[0]
        N = x_out.shape[1]
        device = x_out.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=self.modes2, step=1), \
                            torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        if iphi == None:
            x = x_out
        else:
            x = iphi(x_out, code)

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[:,:,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[:,:,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(1j * 2 * np.pi * K).to(device)

        # coeff (batch, channels, m1, m2)
        u_ft2 = u_ft[..., 1:].flip(-1, -2).conj()
        u_ft = torch.cat([u_ft, u_ft2], dim=-1)

        # Y (batch, channels, N)
        Y = torch.einsum("bcxy,bnxy->bcn", u_ft, basis)
        Y = Y.real
        return Y

class GeoFNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels, out_channels, is_mesh=True, s1=40, s2=40):
        super(GeoFNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.is_mesh = is_mesh
        self.s1 = s1
        self.s2 = s2

        self.fc0 = nn.Linear(in_channels, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = GeoSpectralConv2d(self.width, self.width, self.modes1, self.modes2, s1, s2)
        self.conv1 = GeoSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = GeoSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = GeoSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = GeoSpectralConv2d(self.width, self.width, self.modes1, self.modes2, s1, s2)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.b0 = nn.Conv2d(2, self.width, 1)
        self.b1 = nn.Conv2d(2, self.width, 1)
        self.b2 = nn.Conv2d(2, self.width, 1)
        self.b3 = nn.Conv2d(2, self.width, 1)
        self.b4 = nn.Conv1d(2, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, u, code=None, x_in=None, x_out=None, iphi=None):
        # u (batch, Nx, d) the input value
        # code (batch, Nx, d) the input features
        # x_in (batch, Nx, 2) the input mesh (sampling mesh)
        # xi (batch, xi1, xi2, 2) the computational mesh (uniform)
        # x_in (batch, Nx, 2) the input mesh (query mesh)

        if self.is_mesh and x_in == None:
            x_in = u # [20,1732,2]
        if self.is_mesh and x_out == None:
            x_out = u
        grid = self.get_grid([u.shape[0], self.s1, self.s2], u.device).permute(0, 3, 1, 2) # [20,2,40,40]

        u = self.fc0(u) # [20,1732,32]
        u = u.permute(0, 2, 1) # [20,32,1732]

        uc1 = self.conv0(u, x_in=x_in, iphi=iphi, code=code)
        uc3 = self.b0(grid)
        uc = uc1 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv1(uc)
        uc2 = self.w1(uc)
        uc3 = self.b1(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv2(uc)
        uc2 = self.w2(uc)
        uc3 = self.b2(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv3(uc)
        uc2 = self.w3(uc)
        uc3 = self.b3(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        u = self.conv4(uc, x_out=x_out, iphi=iphi, code=code) # x_out [20,1732,2]
        u3 = self.b4(x_out.permute(0, 2, 1))
        u = u + u3

        u = u.permute(0, 2, 1)
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        return u

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class IPHI(nn.Module):
    def __init__(self, num_nodes, width=32):
        super(IPHI, self).__init__()

        """
        inverse phi: x -> xi
        """
        self.width = width
        self.fc0 = nn.Linear(4, self.width)
        self.fc_code = nn.Linear(num_nodes, self.width)
        self.fc_no_code = nn.Linear(3*self.width, 4*self.width)
        self.fc1 = nn.Linear(4*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 4*self.width)
        self.fc3 = nn.Linear(4*self.width, 4*self.width)
        self.fc4 = nn.Linear(4*self.width, 2)
        self.activation = torch.tanh
        self.center = torch.tensor([0.0001,0.0001], device="cuda").reshape(1,1,2)

        self.B = np.pi*torch.pow(2, torch.arange(0, self.width//4, dtype=torch.float, device="cuda")).reshape(1,1,1,self.width//4)


    def forward(self, x, code=None):
        # x (batch, N_grid, 2)
        # code (batch, N_features)

        # some feature engineering
        angle = torch.atan2(x[:,:,1] - self.center[:,:, 1], x[:,:,0] - self.center[:,:, 0])
        radius = torch.norm(x - self.center, dim=-1, p=2)
        xd = torch.stack([x[:,:,0], x[:,:,1], angle, radius], dim=-1)

        # sin features from NeRF
        b, n, d = xd.shape[0], xd.shape[1], xd.shape[2]
        x_sin = torch.sin(self.B * xd.view(b,n,d,1)).view(b,n,d*self.width//4)
        x_cos = torch.cos(self.B * xd.view(b,n,d,1)).view(b,n,d*self.width//4)
        xd = self.fc0(xd)
        xd = torch.cat([xd, x_sin, x_cos], dim=-1).reshape(b,n,3*self.width)

        if code!= None:
            cd = self.fc_code(code)
            cd = cd.unsqueeze(1).repeat(1,xd.shape[1],1)
            xd = torch.cat([cd,xd],dim=-1)
        else:
            xd = self.fc_no_code(xd)

        xd = self.fc1(xd)
        xd = self.activation(xd)
        xd = self.fc2(xd)
        xd = self.activation(xd)
        xd = self.fc3(xd)
        xd = self.activation(xd)
        xd = self.fc4(xd)
        return x + x * xd


class NUSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(NUSpectralConv2d, self).__init__()

        """
        3D Fourier layer. It does; FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        # self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        # x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])
        x_ft = torch.fft.rfft2(x, dim=[-2, -1])
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, -self.modes2:], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, -self.modes2:], self.weights4)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class NUFNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels=3, out_channels=3):
        super(NUFNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 6  # pad the domain if input is non-periodic

        self.p = nn.Linear(in_channels + 2,
                           self.width)  # input channel is 6: (x_velocity, y_velocity, z_velocity) + 3 locations (u, v, w, x, y, z)
        self.conv0 = NUSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = NUSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = NUSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = NUSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP(self.width, out_channels, self.width * 4)  # output channel is 3: (u, v, w)

    def forward(self, x):  # x [20,13,10,48]
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class MAELoss(nn.Module):
    def __init__(self, size_average=True):
        super(MAELoss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        if self.size_average:
            return torch.mean(torch.abs(input - target))
        else:
            return torch.sum(torch.abs(input - target))

class MRELoss(nn.Module):
    def __init__(self, size_average=True):
        super(MRELoss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        if self.size_average:
            return torch.mean(torch.abs((input - target) / target))
        else:
            return torch.sum(torch.abs((input - target) / target))

# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

def cal_grid_shape(tot_size, aspect_ratios: list[float]):
    '''
    Given the total size and (geometric) aspect ratio,
    output the grid shape.
    '''
    dim = len(aspect_ratios)
    shape = [None] * dim
    shape[0] = tot_size * np.prod([aspect_ratios[0] / \
        aspect_ratios[j] for j in range(1, dim)])
    shape[0] = shape[0] ** (1 / dim)
    for j in range(1, dim):
        shape[j] = aspect_ratios[j] / \
            aspect_ratios[0] * shape[0]
    shape = [max(int(np.round(l)), 2) for l in shape]
    return shape

# Normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class Max_Min_Normalizer(object):
    def __init__(self, x, eps=0.00001):
        super(Max_Min_Normalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.min = torch.min(x, 0)[0]
        self.max = torch.max(x, 0)[0]
        self.eps = eps

    def encode(self, x):
        x = (x - self.min ) / (self.max - self.min + self.eps) *2-1
        return x

    def decode(self, x, sample_idx=None):
        # if sample_idx is None:
        #     std = self.std + self.eps # n
        #     mean = self.mean
        # else:
        #     if len(self.mean.shape) == len(sample_idx[0].shape):
        #         std = self.std[sample_idx] + self.eps  # batch*n
        #         mean = self.mean[sample_idx]
        #     if len(self.mean.shape) > len(sample_idx[0].shape):
        #         std = self.std[:,sample_idx]+ self.eps # T*batch*n
        #         mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = ((x+1)/2 * (self.max - self.min + self.eps)) + self.min 
        return x

    def cuda(self):
        self.min = self.min.cuda()
        self.max = self.max.cuda()

    def cpu(self):
        self.min = self.min.cpu()
        self.max = self.max.cpu()

def cal_gradient(xobs, input_s, num_dataset=1):
    eps = 1e-6  # 数值微分步长
    yflat_ = np.zeros([xobs.shape[0]])
    for i in tqdm(range(num_dataset), desc="Processing samples's gradient"):
        yobs = input_s[i]
        f = RBFInterpolator(xobs, yobs)
        x = np.array(xobs)
        yflat_dx = derivative(f, x, dx=eps, n=0)
        yflat_dy = derivative(f, x, dx=eps, n=1)
        yflat = np.sqrt(yflat_dx ** 2 + yflat_dy ** 2)
        yflat_ += yflat
    yflat_ = yflat_ / num_dataset
    return yflat_

def add_right_cax(ax, pad, width):
    '''
    在一个ax右边追加与之等高的cax.
    pad是cax与ax的间距.
    width是cax的宽度.
    '''
    axpos = ax.get_position()
    caxpos = mpl.transforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0,
        axpos.x1 + pad + width,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)

    return cax


def find_intersection(high_gradient_borders, low_gradient_borders):
    intersections_dict = dict()
    intersections_all = []
    for low_border in low_gradient_borders:
        intersections = []
        xl1, xl2 = low_border[0][0], low_border[0][1]
        yl1, yl2 = low_border[1][0], low_border[1][1]
        for high_border in high_gradient_borders:
            xh1, xh2 = high_border[0][0], high_border[0][1]
            yh1, yh2 = high_border[1][0], high_border[1][1]
            # 计算交集
            x_left = max(xh1, xl1)
            x_right = min(xh2, xl2)
            y_bottom = max(yh1, yl1)
            y_top = min(yh2, yl2)

            # 检查是否存在交集
            if x_left <= x_right and y_bottom <= y_top:
                intersection = [
                    [x_left, x_right],
                    [y_bottom, y_top]
                ]
                intersections.append(intersection)
                intersections_all.append(intersection)
        intersections_dict[str(low_border)] = intersections
    return intersections_dict, intersections_all

def sample_boundary(boundary, num_samples_per_edge):
    # sampled_points = []

    for i in range(len(boundary)):
        x_left, x_right = boundary[i][0]
        y_bottom, y_top = boundary[i][1]
        
        # x方向边界
        if y_bottom == y_top:
            x = np.linspace(x_left, x_right, num_samples_per_edge)
            y = np.ones([num_samples_per_edge]) * y_bottom
        elif x_left == x_right:
            y = np.linspace(y_bottom, y_top, num_samples_per_edge)
            x = np.ones([num_samples_per_edge]) * x_left
        else:
            print("Error!")
            return None


        if i == 0:
            points = np.stack([x,y],axis=1)
        else:
            points = np.concatenate((points, np.stack([x,y],axis=1)))

    if len(boundary) == 0:
        points = np.array([])
    # sampled_points.append(points)
    return points

def cal_metric(MAE, MRE, num_nodes, n_test):
    meanMAE = sum(MAE) / n_test / num_nodes
    meanMRE = sum(MRE) / n_test / num_nodes
    ss = 0
    sss = 0
    for i in range(len(MAE)):
        ss += (MAE[i] / num_nodes - meanMAE) ** 2
        sss += (MRE[i] / num_nodes - meanMRE) ** 2
    sigma_MAE = (ss / n_test) ** 0.5
    sigma_MRE = (sss / n_test) ** 0.5
    return meanMAE, meanMRE, sigma_MAE, sigma_MRE

def torch_diff(u, xt, order = 1, dim=None):
    grad = torch.autograd.grad(outputs=u,inputs = xt,
                                grad_outputs = torch.ones_like(u),
                                 create_graph=True)[0]
    if dim is not None:
        grad = grad[:,:,dim:dim+1]
    for _ in range(order-1):
        grad = torch.autograd.grad(outputs=grad,inputs = xt,
                                grad_outputs = torch.ones_like(u),
                                 create_graph=True)[0]
        if dim is not None:
            grad = grad[:,:,dim:dim+1]
    return grad