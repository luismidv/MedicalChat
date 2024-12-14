import torch.nn as nn
import torch
from tqdm import tqdm

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.sigmoid(out)
        return out


def model_training(model, dataloader, loss_fn, optimizer, device = torch.device("cpu")):
    for (x,y) in tqdm(dataloader, desc = "TRAINING", leave = False):
        print(len(x))
        x = x.to(device)
        y = y.to(device)

        model.train()
        optimizer.zero_grad()

        predict = model(x)

        
        loss = loss_fn(predict, y)
        loss.backward()
        optimizer.step()

    return predict

