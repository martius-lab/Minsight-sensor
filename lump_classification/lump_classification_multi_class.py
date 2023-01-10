import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix

from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from tqdm import tqdm


class DataSet(Dataset):

  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    if len(self.X) != len(self.Y):
      raise Exception("The length of X does not match the length of Y")

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):

    _x = self.X[index]
    _y = self.Y[index]

    return _x, _y

def get_dataloaders(result_path, mode="forcemaps"):
    file_list_train = glob.glob(data_path +"train/"+"*forcevecs.npy")
    file_list_test = glob.glob(data_path +"test/"+"*forcevecs.npy")

    print("Train set")
    train_loader = get_dataloader(file_list_train, result_path + "train/", 32,  mode=mode)
    print("Test set")
    test_loader = get_dataloader(file_list_test, result_path + "test/", 1, mode=mode)

    return train_loader, test_loader

def get_dataloader(file_list, result_path, batch_size, mode="forcemaps"):
    
    os.makedirs(result_path, exist_ok=True)
    
    X = []
    y = []

    maps_nolump = []
    maps_small_lump = []
    maps_medium_lump = []
    maps_big_lump = []

    for file in file_list:
        prefix = file[:file.rfind("forcevecs")]
        print(prefix)

        if mode == "forcemaps":
            forcemaps_file = prefix + "forcemaps.npy"
            forcemaps = np.load(forcemaps_file)
        elif mode == "images":
            forcemaps = np.load(prefix + "images.npy")
            forcemaps = np.moveaxis(forcemaps, -1, 1)
            print(len(forcemaps))
        else:
            raise Exception("Wrong mode for network input")

        forcevec = np.load(file)
        forcevec_norm = np.linalg.norm(forcevec[:,3:], axis=1)

        # A : Use every single image of a palpation as input
        for i in range(50, len(forcevec_norm)):
            if forcevec_norm[i] >= 0.00:
                X.append(forcemaps[i]*1000)
                #print(np.max(forcemaps[i]))
                if "nolump" in file:
                    y.append([0, forcevec_norm[i]])
                    maps_nolump.append(forcemaps[i])              
                elif "lump6.5" in file:
                    y.append([1, forcevec_norm[i]])
                    maps_small_lump.append(forcemaps[i])
                elif "lump9.5" in file:
                    y.append([2, forcevec_norm[i]])
                    maps_medium_lump.append(forcemaps[i])
                elif "lump12.5" in file:
                    y.append([3, forcevec_norm[i]])
                    maps_big_lump.append(forcemaps[i])
            
    print(" The dataset contains %s - %s - %s samples with s-m-b lumps and %s samples without lump" %(len(maps_small_lump), len(maps_medium_lump), len(maps_big_lump), len(maps_nolump)))

    data_loader = DataLoader(DataSet(X, y), batch_size=batch_size, shuffle=True)

    return data_loader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(784, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():

    train_loader, test_loader = get_dataloaders(result_path, mode="forcemaps")

    model = Net()
    summary(model, (3,40,40), device = 'cpu')

    criterion=nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    loss_store = []
    for epoch in range(15):

        running_loss = 0.0
        
        for i, data in tqdm(enumerate(iter(train_loader), 0)):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = torch.squeeze(model(inputs))

            loss = criterion(outputs, labels[0])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 1:
                loss_store.append(loss.item())
                running_loss = 0.0

        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')

    print('Finished Training')
    torch.save(model, result_path + "model.pt")
    plt.plot(loss_store)
    plt.ylabel("Loss")
    plt.show()
    plt.savefig(result_path + "loss_during_training.png")

    
    #Calculate score
    correct = 0
    total = 0

    y_pred = []
    y_true = []

    with torch.no_grad():
        for data in iter(test_loader):
            images, labels = data

            outputs = model(images)
            predicted = torch.max(outputs,1).indices
            
            total += labels[0].size(0)
            correct += (predicted == labels[0])

            y_true.append(labels[0].numpy())
            y_pred.append(predicted.cpu().numpy())


    print("Confusion matrix on test set")
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(conf_matrix)

    df_cm = pd.DataFrame(conf_matrix, index = [i for i in ["no lump", "6.5", "9.5", "12.5"]],
                  columns = [i for i in ["no lump", "6.5", "9.5", "12.5"]])
    plt.figure(figsize = (10,7))
    sns.set(font_scale=2)
    sns.heatmap(df_cm, annot=True, fmt="d", cmap = sns.color_palette("rocket_r", as_cmap=True))
    plt.savefig(result_path + "confusion_matrix.png", dpi=600)


    print(f'Accuracy of the network on the test set: {100 * correct.item() // total} %')

if __name__ == "__main__":

    data_path = "../../Data/lump_detection_data/"
    result_path = "results/"
    main()