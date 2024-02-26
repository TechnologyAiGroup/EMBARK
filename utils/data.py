import torch
import random
import numpy as np
from math import sqrt

def simpleSplit(data, nums):
    plc = 0
    splitResult = []
    for i in range(len(nums)):
        splitResult.append(data[plc:plc+nums[i]])
        plc += nums[i]
    return splitResult

class simpleDataLoader:
    def __init__(self, Xlabel, Ylabel, batch_size, shuffle=True, drop_last=False, device=torch.device('cuda')):
        assert Xlabel.shape[0] == Ylabel.shape[0], "Data Shape Error"
        self.Xlabel = torch.from_numpy(Xlabel).to(device, torch.float32)
        self.Ylabel = torch.from_numpy(Ylabel).to(device, torch.float32)
        self.size = self.Xlabel.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index = [(x, x + batch_size) for x in range(0, self.size, self.batch_size)]
        if self.index[-1][1] > self.size:
            if drop_last is True:
                self.index = self.index[:-1]
                self.size = self.index[-1][1]
            else:
                self.index[-1] = (self.index[-1][0], self.size)
    
    def __iter__(self):
        self.pos = -1
        if self.shuffle is True:
            random.shuffle(self.index)
        return self

    def __next__(self):
        self.pos += 1
        if self.pos >= len(self.index):
            raise StopIteration
        (l, r) = self.index[self.pos]
        return self.Xlabel[l:r], self.Ylabel[l:r]
    
    def __len__(self):
        return len(self.index)
    
def makeData(filename, data_size, PUFSample, with_reliability=False, repetition=11):
    dataset = []
    length = PUFSample.PUF_length + 1
    for _ in range(data_size):
        phi = np.asarray([random.randint(0, 1) * 2 - 1 for _ in range(length)])
        if with_reliability is True:
            r = 0
            for _ in range(repetition):
                r += PUFSample.getResponse(phi)
            if r >= 6:
                R = 1
                r = r * 2.0 / 11 - 1
            else:
                R = 0
                r = (11 - r) * 2.0 / 11 - 1
            dataline = np.hstack((phi, R, r)).tolist()
        else:
            R = PUFSample.getResponse(phi)
            dataline = np.hstack((phi, R)).tolist()
        dataset.append(dataline)
    dataset = np.asarray(dataset)
    if with_reliability is True:
        np.savetxt(filename, dataset, fmt='%.2f', delimiter=',')
    else:
        np.savetxt(filename, dataset, fmt='%d', delimiter=',')

def loadData(filename, batch_size, device=torch.device('cuda'), with_reliability=False):
    dataSet = np.loadtxt(filename, delimiter=',')
    np.random.shuffle(dataSet)

    dataSize = dataSet.shape[0]
    trainSize = int(dataSize * 0.9)
    validSize = int(dataSize * 0.01)
    testSize = dataSize - trainSize - validSize

    if with_reliability is True:
        Xlabel, Ylabel = dataSet[:,:-2], dataSet[:,-2:]
    else:
        Xlabel, Ylabel = dataSet[:,:-1], dataSet[:,-1:]
    [trainSetX, validSetX, testSetX] = simpleSplit(Xlabel, [trainSize, validSize, testSize])
    [trainSetY, validSetY, testSetY] = simpleSplit(Ylabel, [trainSize, validSize, testSize])

    trainLoader = simpleDataLoader(trainSetX, trainSetY, batch_size=batch_size, shuffle=True, device=device)
    validLoader = simpleDataLoader(validSetX, validSetY, batch_size=batch_size, shuffle=True, device=device)
    testLoader = simpleDataLoader(testSetX, testSetY, batch_size=batch_size, shuffle=True, device=device)
    return trainLoader, validLoader, testLoader

def loadTrainData(filename, batch_size, device=torch.device('cuda'), with_reliability=False):
    dataSet = np.loadtxt(filename, delimiter=',')
    np.random.shuffle(dataSet)

    dataSize = dataSet.shape[0]
    trainSize = dataSize

    if with_reliability is True:
        Xlabel, Ylabel = dataSet[:,:-2], dataSet[:,-2:]
    else:
        Xlabel, Ylabel = dataSet[:,:-1], dataSet[:,-1:]
    trainSetX, trainSetY = Xlabel, Ylabel

    trainLoader = simpleDataLoader(trainSetX, trainSetY, batch_size=batch_size, shuffle=True, device=device)
    return trainLoader

def loadTestData(filename, batch_size, device=torch.device('cuda'), with_reliability=False):
    dataSet = np.loadtxt(filename, delimiter=',')
    np.random.shuffle(dataSet)

    dataSize = dataSet.shape[0]
    validSize = int(dataSize * 0.1)
    testSize = dataSize - validSize

    if with_reliability is True:
        Xlabel, Ylabel = dataSet[:,:-2], dataSet[:,-2:]
    else:
        Xlabel, Ylabel = dataSet[:,:-1], dataSet[:,-1:]
    [validSetX, testSetX] = simpleSplit(Xlabel, [validSize, testSize])
    [validSetY, testSetY] = simpleSplit(Ylabel, [validSize, testSize])

    validLoader = simpleDataLoader(validSetX, validSetY, batch_size=batch_size, shuffle=True, device=device)
    testLoader = simpleDataLoader(testSetX, testSetY, batch_size=batch_size, shuffle=True, device=device)
    return validLoader, testLoader