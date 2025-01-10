import torch
# from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tkinter as tk
from tkinter import messagebox

# Define a simple neural network model
class NeuralNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # input layer (28x28 images) -> hidden layer (128 units)
        self.fc2 = nn.Linear(128, 10)  # hidden layer (128 units) -> output layer (10 units)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # activation function for hidden layer
        x = self.fc2(x)
        return x

# Define a custom dataset class for our data
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# Load MNIST dataset (you can replace this with your own dataset)
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = DataLoader(MyDataset(trainset.data, trainset.targets), batch_size=64, shuffle=True)

# Set up the model, loss function, and optimizer
model = NeuralNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Define a function to train the model
def train_model():
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.view(-1, 784))  # flatten the input data
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch %d, loss = %.3f' % (epoch+1, running_loss/(i+1)))

# Define a function to save the model
def save_model():
    torch.save(model.state_dict(), 'model.pth')
    messagebox.showinfo("Model Saved", "Model saved to model.pth")

# Define a function to load the model
def load_model():
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    messagebox.showinfo("Model Loaded", "Model loaded from model.pth")

# Create the Tkinter interface
root = tk.Tk()
root.title("Neural Network")

# Create buttons to train, save, and load the model
train_button = tk.Button(root, text="Train Model", command=train_model)
train_button.pack()

save_button = tk.Button(root, text="Save Model", command=save_model)
save_button.pack()

load_button = tk.Button(root, text="Load Model", command=load_model)
load_button.pack()

root.mainloop()