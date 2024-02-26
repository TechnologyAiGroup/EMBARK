import numpy as np
from math import sqrt, log2, ceil

class MPUFModel():
    def __init__(self, Sweight, Dweight, noise=0.0):
        self.Sweight = Sweight
        self.Dweight = Dweight
        self.noise = noise
        self.PUF_length = Sweight.shape[-1] - 1
    
    def getSelect(self, phi):
        select = 0
        Sweight = self.Sweight + np.random.normal(0, self.noise, size=self.Sweight.shape)
        Souts = np.sum(phi * Sweight, axis=-1, keepdims=False)
        for i in range(Souts.shape[0]):
            select = select * 2 + int(Souts[i] >= 0)
        return select

    def getResponse(self, phi):
        select = self.getSelect(phi)
        Dweight = self.Dweight[select] + np.random.normal(0, self.noise, size=self.Dweight[select].shape)
        delta = np.sum(phi * Dweight, axis=-1, keepdims=False)
        return int(delta >= 0)

    def getInnerPearson(self, op='S'):
        weights = self.Sweight
        if op == 'SD':
            weights = np.concatenate((self.Sweight, self.Dweight), 0)

        x = torch.from_numpy(weights)
        vx = x - torch.mean(x, axis=-1).unsqueeze(-1)
        std_x = torch.rsqrt(torch.sum(vx ** 2, axis=-1)).unsqueeze(1)
        pearson_coef = 0.0
        for current_index in range(x.shape[0]-1):
            cost = (vx[current_index] @ vx[current_index+1:].T) * std_x[current_index] * std_x[current_index+1:].T
            pearson_coef += torch.sum(torch.abs(cost))
        return pearson_coef

    def randomSample(Snum, length=32, alpha=0.05, noise=0.0):
        Sweight = np.random.normal(0, alpha, size=(Snum, length + 1))
        Dweight = np.random.normal(0, alpha, size=(2 ** Snum, length + 1))
        return MPUFModel(Sweight, Dweight, alpha * noise)
    
class cMPUFModel(MPUFModel):
    def getResponse(self, phi):
        select = self.getSelect(phi)
        select, neg = select >> 1, select & 1
        Dweight = self.Dweight[select] + np.random.normal(0, self.noise, size=self.Dweight[select].shape)
        delta = np.sum(phi * Dweight, axis=-1, keepdims=False)
        return int(delta >= 0) ^ neg

    def randomSample(Snum, length=32, alpha=0.05, noise=0.0):
        Sweight = np.random.normal(0, alpha, size=(Snum, length + 1))
        Dweight = np.random.normal(0, alpha, size=(2 ** (Snum - 1), length + 1))
        return cMPUFModel(Sweight, Dweight, alpha * noise)
    
class rMPUFModel(MPUFModel):
    def getSelect(self, phi):
        select = 0
        Sweight = self.Sweight + np.random.normal(0, self.noise, size=self.Sweight.shape)
        Souts = np.sum(phi * Sweight, axis=-1, keepdims=False)
        plc = 1
        while plc <= self.Sweight.shape[0]:
            response = int(Souts[self.Sweight.shape[0] - plc] >= 0)
            select = select * 2 + response
            plc = plc * 2 + 1 - response
        return select

    def randomSample(Snum, length=32, alpha=0.05, noise=0):
        Sweight = np.random.normal(0, alpha, size=(2 ** Snum - 1, length + 1))
        Dweight = np.random.normal(0, alpha, size=(2 ** Snum, length + 1))
        return rMPUFModel(Sweight, Dweight, alpha * noise)
   
import torch
import torch.nn as nn

class embarkMPUF(nn.Module):
    def __init__(self, Snum, PUF_length, device='cuda'):
        super().__init__()
        
        self.Snum = Snum
        self.device = device
        self.P_S_lins = nn.Sequential(
            nn.Linear(PUF_length + 1, Snum),
            nn.Sigmoid()
        )
        self.P_D_lins = nn.Sequential(
            nn.Linear(PUF_length + 1, (2 **Snum)),
            nn.Sigmoid()
        )

        self.W, self.B = [], []
        tmpX = np.array([[1], [1]])
        for i in range(self.Snum):
            w = np.array([[-1], [1]])
            b = np.array([[1], [0]])
            for j in range(i + 1, self.Snum):
                w = np.kron(tmpX, w)
                b = np.kron(tmpX, b)
            for j in range(i):
                w = np.kron(w, tmpX)
                b = np.kron(b, tmpX)
            self.W.append(torch.tensor(w.T).to(self.device, torch.float32))
            self.B.append(torch.tensor(b.T).to(self.device, torch.float32))

    def forward(self, phi):
        Souts = self.P_S_lins(phi)
        Douts = self.P_D_lins(phi)

        M = torch.ones_like(Douts)
        for i in range(self.Snum):
            m = torch.mm(Souts[:, i:i+1], self.W[i]) + self.B[i]
            M = M * m
        ans = torch.sum(Douts * M * 0.99999, dim=-1, keepdim=True)
        # ans = torch.minimum(ans, torch.ones_like(ans))
        return ans

class embarkcMPUF(nn.Module):
    def __init__(self, Snum, PUF_length, device='cuda'):
        super().__init__()

        self.Snum = Snum
        self.device = device
        self.P_S_lins = nn.Sequential(
            nn.Linear(PUF_length + 1, Snum),
            nn.Sigmoid()
        )
        self.P_D_lins = nn.Sequential(
            nn.Linear(PUF_length + 1, 2 ** (Snum - 1)),
            nn.Sigmoid()
        )

        self.W, self.B = [], []
        tmpX = np.array([[1], [1]])
        for i in range(self.Snum):
            w = np.array([[-1], [1]])
            b = np.array([[1], [0]])
            for _ in range(i + 1, self.Snum):
                w = np.kron(tmpX, w)
                b = np.kron(tmpX, b)
            for _ in range(i):
                w = np.kron(w, tmpX)
                b = np.kron(b, tmpX)
            self.W.append(torch.tensor(w.T).to(self.device, torch.float32))
            self.B.append(torch.tensor(b.T).to(self.device, torch.float32))
        
        self.W0 = [[0] * (2 ** self.Snum) for _ in range(2 ** (self.Snum - 1))]
        self.B0 = [0] * (2 ** self.Snum)
        for i in range(2 ** (self.Snum - 1)):
            self.W0[i][i * 2], self.W0[i][i * 2 + 1] = 1, -1
            self.B0[i * 2], self.B0[i * 2 + 1] = 0, 1
        self.W0 = torch.tensor(self.W0).to(self.device, torch.float32)
        self.B0 = torch.tensor(self.B0).to(self.device, torch.float32)
    
    def forward(self, phi):
        Souts = self.P_S_lins(phi)
        Douts = self.P_D_lins(phi)
        Douts = torch.mm(Douts, self.W0) + self.B0

        M = torch.ones_like(Douts)
        for i in range(self.Snum):
            m = torch.mm(Souts[:, i:i+1], self.W[i]) + self.B[i]
            M = M * m
        ans = torch.sum(Douts * M * 0.99999, dim=-1, keepdim=True)
        return ans

class embarkrMPUF(nn.Module):
    def __init__(self, Snum, PUF_length, emb_dim=1, device='cuda'):
        super().__init__()
        
        self.Snum = Snum
        self.device = device
        self.P_S_lins = nn.Sequential(
            nn.Linear(PUF_length + 1, 2 ** Snum - 1),
            nn.Sigmoid()
        )
        self.P_D_lins = nn.Sequential(
            nn.Linear(PUF_length + 1, 2 ** Snum),
            nn.Sigmoid()
        )

        self.W, self.B = [], []        
        tmpX1 = np.array([[1, 0], [0, 1]])
        tmpX2 = np.array([[1], [1]])
        for i in range(self.Snum):
            w = np.array([[-1], [1]])
            b = np.array([[1], [0]])
            for _ in range(i + 1, self.Snum):
                w = np.kron(tmpX1, w)
                b = np.kron(tmpX2, b)
            for _ in range(i):
                w = np.kron(w, tmpX2)
                b = np.kron(b, tmpX2)
            self.W.append(torch.tensor(w.T).to(self.device, torch.float32))
            self.B.append(torch.tensor(b.T).to(self.device, torch.float32))

    def forward(self, phi):
        Souts = self.P_S_lins(phi)
        Douts = self.P_D_lins(phi)

        M = torch.ones_like(Douts)
        l, r = 0, 0
        for i in range(self.Snum):
            l, r = r, r + 2 ** (self.Snum - i - 1)
            m = torch.mm(Souts[:, l:r], self.W[i]) + self.B[i]
            M = M * m
        ans = torch.sum(Douts * M * 0.99999, dim=-1, keepdim=True)
        return ans
        
class combineMPUF(nn.Module):
    def __init__(self, Snum, PUF_length, device='cuda'):
        super().__init__()
        
        self.Snum = Snum
        self.device = device
        self.P_S_lins = nn.Linear(PUF_length + 1, Snum)
        self.P_D_lins = nn.Linear(PUF_length + 1, (2 ** Snum))

        self.W, self.B = [], []
        tmpX = np.array([[1], [1]])
        for i in range(self.Snum):
            w = np.array([[-1], [1]])
            b = np.array([[1], [0]])
            for _ in range(i + 1, self.Snum):
                w = np.kron(tmpX, w)
                b = np.kron(tmpX, b)
            for _ in range(i):
                w = np.kron(w, tmpX)
                b = np.kron(b, tmpX)
            self.W.append(torch.tensor(w.T).to(self.device, torch.float32))
            self.B.append(torch.tensor(b.T).to(self.device, torch.float32))

        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def calc_pearson_coef(x, y):
        vx = x - torch.mean(x, axis=0).unsqueeze(axis=0)
        vy = y - torch.mean(y, axis=0).unsqueeze(axis=0)
        pearson_coef = vx * vy * (torch.rsqrt(torch.sum(vx ** 2, axis=0)) * torch.rsqrt(torch.sum(vy ** 2, axis=0))).unsqueeze(axis=0)
        pearson_coef = torch.sum(pearson_coef, axis=0)
        return torch.sum(pearson_coef)

    def inner_pearson_coef(self):
        x = self.P_S_lins.weight

        vx = x - torch.mean(x, axis=-1).unsqueeze(-1)
        std_x = torch.rsqrt(torch.sum(vx ** 2, axis=-1)).unsqueeze(1)
        pearson_coef = 0.0
        for current_index in range(x.shape[0]-1):
            cost = (vx[current_index] @ vx[current_index+1:].T) * std_x[current_index] * std_x[current_index+1:].T
            pearson_coef += torch.sum(torch.abs(cost))
        return pearson_coef
        
    def forward(self, phi):
        Sdelta = self.P_S_lins(phi)
        Ddelta = self.P_D_lins(phi)

        Souts = self.sigmoid(Sdelta)
        Douts = self.sigmoid(Ddelta)
        M = torch.ones_like(Douts)
        for i in range(self.Snum):
            m = torch.mm(Souts[:, i:i+1], self.W[i]) + self.B[i]
            M = M * m
        ans = torch.sum(Douts * M * 0.99999, dim=-1, keepdim=True)

        reliability_output = torch.abs(Sdelta)
        return (ans, reliability_output)
    
class combinecMPUF(nn.Module):
    def __init__(self, Snum, PUF_length, device='cuda'):
        super().__init__()

        self.Snum = Snum
        self.device = device
        self.P_S_lins = nn.Linear(PUF_length + 1, Snum)
        self.P_D_lins = nn.Linear(PUF_length + 1, 2 ** (Snum - 1))
        
        self.W, self.B = [], []
        tmpX = np.array([[1], [1]])
        for i in range(self.Snum):
            w = np.array([[-1], [1]])
            b = np.array([[1], [0]])
            for _ in range(i + 1, self.Snum):
                w = np.kron(tmpX, w)
                b = np.kron(tmpX, b)
            for _ in range(i):
                w = np.kron(w, tmpX)
                b = np.kron(b, tmpX)
            self.W.append(torch.tensor(w.T).to(self.device, torch.float32))
            self.B.append(torch.tensor(b.T).to(self.device, torch.float32))
        
        self.W0 = [[0] * (2 ** self.Snum) for _ in range(2 ** (self.Snum - 1))]
        self.B0 = [0] * (2 ** self.Snum)
        for i in range(2 ** (self.Snum - 1)):
            self.W0[i][i * 2], self.W0[i][i * 2 + 1] = 1, -1
            self.B0[i * 2], self.B0[i * 2 + 1] = 0, 1
        self.W0 = torch.tensor(self.W0).to(self.device, torch.float32)
        self.B0 = torch.tensor(self.B0).to(self.device, torch.float32)
   
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def calc_pearson_coef(x, y):
        vx = x - torch.mean(x, axis=0).unsqueeze(axis=0)
        vy = y - torch.mean(y, axis=0).unsqueeze(axis=0)
        pearson_coef = vx * vy * (torch.rsqrt(torch.sum(vx ** 2, axis=0)) * torch.rsqrt(torch.sum(vy ** 2, axis=0))).unsqueeze(axis=0)
        pearson_coef = torch.sum(pearson_coef, axis=0)

        #reweight
        pearson_coef[0] = pearson_coef[0] * 2
        return torch.sum(pearson_coef)

    def inner_pearson_coef(self):
        x = self.P_S_lins.weight

        vx = x - torch.mean(x, axis=-1).unsqueeze(-1)
        std_x = torch.rsqrt(torch.sum(vx ** 2, axis=-1)).unsqueeze(1)
        pearson_coef = 0.0
        for current_index in range(x.shape[0]-1):
            cost = (vx[current_index] @ vx[current_index+1:].T) * std_x[current_index] * std_x[current_index+1:].T
            pearson_coef += torch.sum(torch.abs(cost))
        return pearson_coef

    def forward(self, phi):
        Sdelta = self.P_S_lins(phi)
        Ddelta = self.P_D_lins(phi)

        Souts = self.sigmoid(Sdelta)
        Douts = self.sigmoid(Ddelta)
        Douts = torch.mm(Douts, self.W0) + self.B0
        M = torch.ones_like(Douts)
        for i in range(self.Snum):
            m = torch.mm(Souts[:, i:i+1], self.W[i]) + self.B[i]
            M = M * m
        ans = torch.sum(Douts * M * 0.99999, dim=-1, keepdim=True)

        reliability_output = torch.abs(Sdelta)
        return (ans, reliability_output)

class combinerMPUF(nn.Module):
    def __init__(self, Snum, PUF_length, device='cuda'):
        super().__init__()

        self.Snum = Snum
        self.device = device
        self.P_S_lins = nn.Linear(PUF_length + 1, 2 ** Snum - 1)
        self.P_D_lins = nn.Linear(PUF_length + 1, 2 ** Snum)

        self.W, self.B = [], []        
        tmpX1 = np.array([[1, 0], [0, 1]])
        tmpX2 = np.array([[1], [1]])
        for i in range(self.Snum):
            w = np.array([[-1], [1]])
            b = np.array([[1], [0]])
            for _ in range(i + 1, self.Snum):
                w = np.kron(tmpX1, w)
                b = np.kron(tmpX2, b)
            for _ in range(i):
                w = np.kron(w, tmpX2)
                b = np.kron(b, tmpX2)
            self.W.append(torch.tensor(w.T).to(self.device, torch.float32))
            self.B.append(torch.tensor(b.T).to(self.device, torch.float32))

        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def calc_pearson_coef(x, y):
        vx = x - torch.mean(x, axis=0).unsqueeze(axis=0)
        vy = y - torch.mean(y, axis=0).unsqueeze(axis=0)
        pearson_coef = vx * vy * (torch.rsqrt(torch.sum(vx ** 2, axis=0)) * torch.rsqrt(torch.sum(vy ** 2, axis=0))).unsqueeze(axis=0)
        pearson_coef = torch.sum(pearson_coef, axis=0)
        
        #reweight
        reweight = 1.0
        length, cnt = 1, 0
        for i in range(pearson_coef.shape[0] - 1, 0, -1):
            pearson_coef[i] = pearson_coef[i] * reweight
            cnt += 1
            if cnt == length:
                cnt, length = 0, length * 2
                reweight = reweight / 2
        return torch.sum(pearson_coef)

    def inner_pearson_coef(self):
        x = self.P_S_lins.weight

        vx = x - torch.mean(x, axis=-1).unsqueeze(-1)
        std_x = torch.rsqrt(torch.sum(vx ** 2, axis=-1)).unsqueeze(1)
        pearson_coef = 0.0
        for current_index in range(x.shape[0]-1):
            cost = (vx[current_index] @ vx[current_index+1:].T) * std_x[current_index] * std_x[current_index+1:].T
            pearson_coef += torch.sum(torch.abs(cost))
        return pearson_coef
    
    def forward(self, phi):
        Sdelta = self.P_S_lins(phi)
        Ddelta = self.P_D_lins(phi)

        Souts = self.sigmoid(Sdelta)
        Douts = self.sigmoid(Ddelta)
        M = torch.ones_like(Douts)
        l, r = 0, 0
        for i in range(self.Snum):
            l, r = r, r + 2 ** (self.Snum - i - 1)
            m = torch.mm(Souts[:, l:r], self.W[i]) + self.B[i]
            M = M * m
        ans = torch.sum(Douts * M * 0.99999, dim=-1, keepdim=True)

        reliability_output = torch.abs(Sdelta)
        return (ans, reliability_output)
