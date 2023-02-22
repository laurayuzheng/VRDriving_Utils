import torch
from torch import Tensor
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.nn import Linear, MSELoss, L1Loss, functional as F

from sklearn.model_selection import KFold

from torch.utils.tensorboard import SummaryWriter
import numpy as np

import random

from run_stats import * 
from config import * 

# SCENARIO_MAX_STEPS = {
#     0: 700, 
#     1: 600, 
#     2: 750, 
#     3: 1450
# }

class VRDrivingDataset(Dataset):
    def __init__(self, scenario, shuffle=True):
        self.stats = StatsManager(DATADIR, EXCLUSIONS)
        self.personality_data = self.stats.get_personality_data()
        self.simdata = self.stats.get_sim_data(scenario)

        if shuffle: 
            zipped = list(zip(list(self.personality_data), list(self.simdata)))
            random.shuffle(zipped)
            self.personality_data, self.simdata = zip(*zipped)

    def __len__(self):
        return len(self.personality_data)

    def __getitem__(self, idx):
        return self.simdata[idx].T, self.personality_data[idx]

class SimpleNet(torch.nn.Module):
    ''' 
    Simple linear regression model 
    '''
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = Linear(MAX_STEPS*2, 100)
        self.fc2 = Linear(100, 8)
        # self.fc3 = Linear(100, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.fc3(x)
        return x

class Trainer:

    def __init__(self, scenario, 
                 optimizer="adam", 
                 criterion="mse", 
                 n_epochs=1000, 
                 lr=0.01, 
                 batch_size=1, 
                 num_splits=2):        

        self.scenario = scenario
        self.num_splits = num_splits

        self.init_model(criterion, 
                    n_epochs, 
                    lr, 
                    batch_size)

    def init_model(self, 
                criterion, 
                n_epochs, 
                lr, 
                batch_size):
        
        self.net = SimpleNet() 
        self.dataset = VRDrivingDataset(self.scenario)
        # self.dataloader = DataLoader(self.dataset, batch_size=batch_size)
        self.splits = KFold(n_splits=self.num_splits,shuffle=True,random_state=42)

        print("Size of dataset: ", len(self.dataset))

        self.lr = lr 
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.criterion = MSELoss() if criterion=="mse" else L1Loss()
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.writer = SummaryWriter("runs/simple_model_scenario_%d" % (self.scenario))
    
    def train_epoch(self, model, device, train_loader, criterion, optimizer):
        epoch_loss = 0

        for i, data in enumerate(train_loader, 0):
            X, y = data 
            self.optimizer.zero_grad()

            X = Tensor(X).float().flatten()
            y = Tensor(y).float()

            y_pred = model(X)

            loss = criterion(y_pred, y)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        return epoch_loss 

    def test_epoch(self, model, device, test_loader, criterion):
        epoch_loss = 0
        model.eval()

        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                X, y = data 
                self.optimizer.zero_grad()

                X = Tensor(X).float().flatten()
                y = Tensor(y).float()

                y_pred = model(X)

                loss = criterion(y_pred, y)

                epoch_loss += loss.item()


        return epoch_loss 

    def fit(self):

        history = {'train_loss': [], 'test_loss': []}

        for fold, (train_idx,val_idx) in enumerate(self.splits.split(np.arange(len(self.dataset)))):
            print('Fold {}'.format(fold + 1))
            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)
            
            dataloader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=train_sampler)
            test_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=test_sampler)

            self.net = SimpleNet()
            self.optimizer =  Adam(self.net.parameters(), lr=self.lr) 

            for epoch in range(self.n_epochs):
                train_loss = self.train_epoch(self.net, "cpu", dataloader, self.criterion, self.optimizer)
                test_loss = self.test_epoch(self.net, "cpu", test_loader, self.criterion)

                train_loss = train_loss / len(dataloader.sampler)
                test_loss = test_loss / len(test_loader.sampler)

                print("Epoch: {}; Train Loss: {}; Test Loss: {}".format(epoch, train_loss, test_loss))

                if fold >= 0 and fold < 3:
                    self.writer.add_scalars('split{}'.format(fold), {'Loss/train': train_loss,
                        'Loss/validation': test_loss}, epoch)

                history['train_loss'].append(train_loss)
                history['test_loss'].append(test_loss)

        
        avg_train_loss = np.mean(history['train_loss'])
        avg_test_loss = np.mean(history['test_loss'])
        print("Average Training Loss: {:.4f} \t Average Test Loss: {:.4f} ".format(avg_train_loss,avg_test_loss))  
        print("Finished training. ")


if __name__ == "__main__":
    
    for scenario in range(4):
        trainer = Trainer(scenario=scenario, num_splits=8, n_epochs=500) 
        trainer.fit()
