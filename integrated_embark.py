import torch
import torch.nn as nn

from utils.config import Config
from utils.model import MPUFModel, cMPUFModel, rMPUFModel
from utils.model import combineMPUF, combinerMPUF
from utils.data import makeData, loadData

if __name__ == "__main__":
    config = Config("./data/5_64_MPUF_10_30k.csv", makedata=True)

    if config.PUF_type == 'MPUF':
        model = combineMPUF(config.Snum, config.PUF_length, device=config.device).to(config.device)
    elif config.PUF_type == 'cMPUF':
        #model = globalckMPUFLR(config.Snum, config.PUF_length, k=1).to(config.device)
        model = combineMPUF(config.Snum, config.PUF_length, device=config.device).to(config.device)
    elif config.PUF_type == 'rMPUF':
        model = combinerMPUF(config.Snum, config.PUF_length, device=config.device).to(config.device)
    optimizer = torch.optim.Adadelta(model.parameters())
    criterion = config.loss_function

    num_target_apuf = config.Snum
    std_inner_pearson = 1.2 * num_target_apuf * (num_target_apuf - 1) / 20

    if config.makedata is True:
        print("Making Data...")
        if config.PUF_type == 'MPUF':
            PUFSample = MPUFModel.randomSample(config.Snum, config.PUF_length, noise=config.noise)
        elif config.PUF_type[0] == 'c':
            PUFSample = cMPUFModel.randomSample(config.Snum, config.PUF_length, noise=config.noise)
        elif config.PUF_type[0] == 'r':
            PUFSample = rMPUFModel.randomSample(config.Snum, config.PUF_length, noise=config.noise)
        makeData(config.datafile, config.datasize, PUFSample, with_reliability=True)

    train_loader, valid_loader, test_loader = loadData(config.datafile, config.batch_size, config.device, with_reliability=True)
    # train_loader = loadTrainData(config.datafile, config.batch_size, config.device, with_reliability=True)
    # valid_loader, test_loader = loadTestData(config.datafile, config.batch_size, config.device, with_reliability=True)
    
    print("Start Training...")
    from time import time
    st = time()
    best_valid_acc = 0
    predict_loss_coef = config.predict_loss_coef
    pearson_correlation_coef = config.pearson_correlation_coef
    inner_constrain_coef = config.inner_constrain_coef
    for epoch in range(config.epochs):
        #train model
        model.train()
        tot_loss = 0
        
        for (phi, Rr) in train_loader:
            R, r = Rr[:, 0].unsqueeze(-1), Rr[:, 1].unsqueeze(-1)
            (predict, reliability) = model(phi)
            loss = predict_loss_coef * criterion(predict, R)
            loss += pearson_correlation_coef * model.calc_pearson_coef(reliability, r)
            loss += inner_constrain_coef * max(model.inner_pearson_coef() - std_inner_pearson * 1.2, 0)
            tot_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #valid model
        model.eval()
        acc_count = 0
        for (phi, Rr) in valid_loader:
            R, r = Rr[:, 0].unsqueeze(-1), Rr[:, 1].unsqueeze(-1)
            (predict, reliability) = model(phi)
            predict = predict.round()
            acc_count += (predict == R).sum().item()
        valid_acc = acc_count / valid_loader.size
        best_valid_acc = max(valid_acc, best_valid_acc)
        print("Epoch %d: Loss = %.2f, Valid acc = %.2f%%" % \
              (epoch, tot_loss, valid_acc * 100))
    
    #testmodel
    model.eval()
    acc_count = 0
    for (phi, Rr) in test_loader:
        R, r = Rr[:, 0].unsqueeze(-1), Rr[:, 1].unsqueeze(-1)
        (predict, reliability) = model(phi)
        predict = predict.round()
        acc_count += (predict == R).sum().item()
    test_acc = acc_count / test_loader.size

    ed = time()
    print("Best valid Acc = %.2f%%, Test Acc = %.2f%%, Training time cost = %.2f" 
          % (best_valid_acc * 100, test_acc * 100, (ed - st) / 60))
    
    if config.makedata is True:
        #PUFSamlpe eval
        PUFSample.noise = 0.0
        acc_count = 0
        for (phi, Rr) in test_loader:
            for i in range(phi.shape[0]):
                eachphi, eachR = phi[i].cpu().numpy(), Rr[i][0]
                response = PUFSample.getResponse(eachphi)
                acc_count += 1 if response == eachR else 0
        PUF_acc = acc_count / test_loader.size
        print("PUF Acc = %.2f%%" % (PUF_acc * 100))
    
