import torch
import torch.nn.functional as F

class Config():
    def __init__(self, datafile, makedata=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # data config
        self.datafile = datafile
        self.batch_size = 4096 # 512 1024 2048

        import os
        self.makedata = not os.path.exists(datafile) if makedata is None else makedata

        import re
        args = re.split(r"[_]", datafile)
        args[0] = re.split(r"[\/]", args[0])[-1]
        args[-1] = args[-1][:-4]
        self.datasize = float(args[-1][:-1]) * (1e6 if args[-1][-1] == 'm' else 1e3)
        self.datasize = int(self.datasize)
        self.noise = 0.0
        if args[-2].isdigit():
            self.noise = float(1.0 / int(args[-2]))

        # MPUF config
        self.PUF_length = int(args[1])
        self.PUF_type = args[2]
        if self.PUF_type == 'MPUF':
            self.Snum = int(args[0])
        elif self.PUF_type == 'cMPUF':
            self.Snum = int(args[0])
        elif self.PUF_type == 'rMPUF':
            self.Snum = int(args[0])
        elif self.PUF_type == 'XORPUF':
            self.Xnum = int(args[0])
        else:
            raise TypeError

        # learning config
        self.lr = 0.01
        self.epochs = 500
        if self.PUF_type == 'rMPUF':
            self.epochs = 1000
        self.loss_function = F.binary_cross_entropy

        
        self.predict_loss_coef = 5
        self.pearson_correlation_coef = -0.5
        self.inner_constrain_coef = 1
