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

from data_processing import * 
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

        # print(len(self.simdata), len(self.personality_data))
        # print(self.personality_data)
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
    Simple 2 layer NN
    '''
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.fc1 = Linear(MAX_STEPS*2, 100)
        self.fc2 = Linear(100, 8)

        # torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Trainer:

    def __init__(self, scenario, 
                 optimizer="adam",
                 exp_name="simple", 
                 criterion="mse", 
                 n_epochs=1000, 
                 lr=0.01, 
                 batch_size=1, 
                 num_splits=2, 
                 load=None):        

        self.exp_name = exp_name
        self.scenario = scenario
        self.num_splits = num_splits

        self.init_model(criterion, 
                    n_epochs, 
                    lr, 
                    batch_size)
        
        if load: 
            self.net.load_state_dict(torch.load(os.path.join("./models/%s/model.pt" % (load))))
            print("Loaded ./models/%s/model.pt" % (load))

    def init_model(self, 
                criterion, 
                n_epochs, 
                lr, 
                batch_size):
        
        self.net = SimpleNet() 
        self.dataset = VRDrivingDataset(self.scenario)
        # self.dataloader = DataLoader(self.dataset, batch_size=batch_size)
        self.splits = KFold(n_splits=self.num_splits,shuffle=True,random_state=42)

        # print("Size of dataset: ", len(self.dataset))

        self.lr = lr 
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.criterion = MSELoss() if criterion=="mse" else L1Loss()
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.writer = SummaryWriter("runs/%s_scenario_%d" % (self.exp_name, self.scenario))

    
    def train_epoch(self, model, device, train_loader, criterion, optimizer):
        model.train()

        epoch_loss = 0

        for i, data in enumerate(train_loader, 0):
            X, y = data 
            self.optimizer.zero_grad()

            X = Tensor(X).float().view(self.batch_size, -1)
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

                X = Tensor(X).float().view(self.batch_size, -1)
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

    def save(self):
        os.makedirs("./models/%s" % (self.exp_name), exist_ok=True)
        torch.save(self.net.state_dict(), "./models/%s/%s" % (self.exp_name, "model.pt"))

    def get_grad_normalized(self):

        self.net.eval() 
        
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size)

        grads = [] 

        for i, data in enumerate(dataloader, 0):
            X, y = data 

            # X shape: (b=1, 2, 1500) --> (1, 3000)
            X = X.float().view(self.batch_size, -1)

            # y shape: (b=1, 8) 
            y = y.float()

            X.requires_grad = True 
            y.requires_grad = True 

            self.optimizer.zero_grad()

            y_pred = self.net(X)
            loss = self.criterion(y_pred, y)

            loss.backward()
            
            grad = X.grad.data.view(2, -1).mean(axis=0)

            # this nonzero count is always 0
            # print(np.count_nonzero(np.array(grad)))
            # print(np.array(grad))
            grad = np.array(grad)
            grad -= np.amin(grad)
            grad = grad**2
            grad /= (np.amax(grad) - np.amin(grad))
            grad[grad<0.6] = 0.1
            grads.append(grad)
        
        grads = np.array(grads)

        # # now grads is weighted from 0 to 1, we can use this for visualizations.
        return grads

    def predict(self, user, scenario):
        
        X = self.dataset.stats.get_sim_data(scenario, user=user)

        X = X[0].T

        self.net.eval() 

        X = torch.Tensor(X).unsqueeze(0).float().reshape(1, -1)
        
        with torch.no_grad():
            prediction = self.net(X)

        # print(X)
        print(prediction)



if __name__ == "__main__":
    
    stats = StatsManager(DATADIR, EXCLUSIONS)
    stats.save_to_csv()

    learning_rates = [0.001, 0.001, 0.001, 0.0001]
    scenarios = [0,1,2,3]

    for scenario in scenarios:
        exp = "scenario_%d" % (scenario)
        trainer = Trainer(exp_name=exp, scenario=scenario, num_splits=8, n_epochs=100, lr=learning_rates[scenario], load=exp) 
        # trainer.fit()
        # trainer.save() 


        # trainer.predict(user=-1, scenario=scenario)

        # stats.plot_personality_tsne(dimension=2)
        grads = trainer.get_grad_normalized()
        stats.trajectory_heatmap(grads, scenario=scenario)
