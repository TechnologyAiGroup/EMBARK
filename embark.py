import torch
import torch.nn as nn

from utils.config import Config
from utils.model import MPUFModel, cMPUFModel, rMPUFModel
from utils.model import embarkMPUF, embarkcMPUF, embarkrMPUF
from utils.data import makeData, loadData, loadTrainData, loadTestData

if __name__ == "__main__":
    config = Config("./data/5_64_MPUF_50k.csv", makedata=True)

    if config.PUF_type == 'MPUF':
        model = embarkMPUF(config.Snum, config.PUF_length, device=config.device).to(config.device)
    elif config.PUF_type == 'cMPUF':
        model = embarkcMPUF(config.Snum, config.PUF_length, device=config.device).to(config.device)
        # model = embarkMPUF(config.Snum, config.PUF_length, device=config.device).to(config.device)
    elif config.PUF_type == 'rMPUF':
        model = embarkrMPUF(config.Snum, config.PUF_length, device=config.device).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = config.loss_function

    if config.makedata is True:
        print("Making Data...")
        if config.PUF_type == 'MPUF':
            PUFSample = MPUFModel.randomSample(config.Snum, config.PUF_length, noise=config.noise)
        elif config.PUF_type[0] == 'c':
            PUFSample = cMPUFModel.randomSample(config.Snum, config.PUF_length, noise=config.noise)
        elif config.PUF_type[0] == 'r':
            PUFSample = rMPUFModel.randomSample(config.Snum, config.PUF_length, noise=config.noise)
        makeData(config.datafile, config.datasize, PUFSample)
    train_loader, valid_loader, test_loader = loadData(config.datafile, config.batch_size, config.device)

    # train_loader = loadTrainData("./data/5_64_MPUF_50k_train.csv", config.batch_size, config.device)
    # valid_loader, test_loader = loadTestData("./data/5_64_MPUF_50k_test.csv", config.batch_size, config.device)
    
    print("Start Training...")
    from time import time
    st = time()
    best_valid_acc = 0
    for epoch in range(config.epochs):
        #train model
        model.train()
        tot_loss = 0

        from tqdm import tqdm
        for (phi, R) in train_loader:
            predict = model(phi)
            loss = criterion(predict, R)
            tot_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #valid model
        model.eval()
        acc_count = 0
        for (phi, R) in valid_loader:
            predict = model(phi).round()
            acc_count += (predict == R).sum().item()
        valid_acc = acc_count / valid_loader.size
        best_valid_acc = max(valid_acc, best_valid_acc)
        print("Epoch %d: Loss = %.2f, Valid acc = %.2f%%" % \
              (epoch, tot_loss, valid_acc * 100))
    
    #testmodel
    model.eval()
    acc_count = 0
    for (phi, R) in test_loader:
        predict = model(phi).round()
        acc_count += (predict == R).sum().item()
    test_acc = acc_count / test_loader.size

    ed = time()
    print("Best valid Acc = %.2f%%, Test Acc = %.2f%%, Training time cost = %.2f" 
          % (best_valid_acc * 100, test_acc * 100, (ed - st) / 60))
    
    if config.makedata is True:
        #PUFSamlpe eval
        PUFSample.noise = 0.0
        acc_count = 0
        for (phi, R) in test_loader:
            for i in range(phi.shape[0]):
                eachphi, eachR = phi[i].cpu().numpy(), R[i]
                response = PUFSample.getResponse(eachphi)
                acc_count += 1 if response == eachR else 0
        PUF_acc = acc_count / test_loader.size
        print("PUF Acc = %.2f%%" % (PUF_acc * 100))
    
