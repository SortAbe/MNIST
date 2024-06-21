import math
import torch
import torchvision

from torch import nn
from torchvision import transforms
from matplotlib import pyplot

#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameter
input_size = 28*28
hidden_size = 100
classes = 10
epochs = 5
batch_size = 60
learning_rate = 0.001

#MNIST
traninig_dataset: torchvision.datasets.mnist.MNIST = torchvision.datasets.MNIST(root='./data', train=True,
    transform=transforms.ToTensor(), download=True)
test_dataset: torchvision.datasets.mnist.MNIST = torchvision.datasets.MNIST(root='./data', train=False,
    transform=transforms.ToTensor(), download=True)

train_loader: torch.utils.data.dataloader.DataLoader = torch.utils.data.DataLoader(dataset=traninig_dataset, batch_size=batch_size, shuffle=True)
test_loader: torch.utils.data.dataloader.DataLoader = torch.utils.data.DataLoader(dataset=traninig_dataset, batch_size=batch_size, shuffle=False)

class NeuralNet(nn.Module):

    def __init__(self, input_size=input_size, hidden_size=hidden_size, classes=classes):
        super(NeuralNet, self).__init__()

        self.linear = nn.Linear(input_size, hidden_size, device=device)
        self.linear_hidden = nn.Linear(hidden_size, hidden_size, device=device)
        self.linear_hidden2 = nn.Linear(hidden_size, hidden_size, device=device)
        self.linear_out = nn.Linear(hidden_size, classes, device=device)

        self.relu = nn.GELU()

    def forward(self, x):
        out = self.relu(self.linear(x))
        out = self.relu(self.linear_hidden(out))
        out = self.relu(self.linear_hidden2(out))
        out = self.relu(self.linear_out(out))
        return out

    def count(self):
        size = 0
        print(self.state_dict().keys())
        for key in self.state_dict().keys():
            size += math.prod(list(self.state_dict()[key].size()))
        print(size)

    def image_save(self):
        for key, value in self.state_dict().items():
            value = value.detach().numpy()
            if len(value.shape) == 2:
                if value.shape[0] > value.shape[1]:
                    height = 40
                    width = (value.shape[1] / value.shape[0]) * 40
                else:
                    height = (value.shape[0] / value.shape[1]) * 40
                    width = 40
                pyplot.figure(figsize=(width,height))
                pyplot.imshow(value, cmap='seismic')
                pyplot.title(key)
                pyplot.ylabel('output')
                pyplot.xlabel('input')
                pyplot.savefig(key.replace('.','_') + '.png')

    def first(self):
        count = 1
        #print(len(self.state_dict()['linear.weight']))
        for neuron in self.state_dict()['linear.weight']:
            neuron = torch.reshape(neuron, [28,28]).detach().numpy()
            pyplot.subplot(5, 5, count)
            pyplot.imshow(neuron, cmap='seismic')
            pyplot.title(count)
            count += 1
            if count == 26:
                pyplot.show()
                count = 1

    def display(self):
        for key, value in self.state_dict().items():
                    value = value.detach().numpy()
                    if len(value.shape) == 2:
                        pyplot.imshow(value, cmap='seismic')
                        pyplot.title(key)
                        pyplot.ylabel('output')
                        pyplot.xlabel('input')
                        pyplot.show()


    def save(self):
        torch.save(self.state_dict(), 'parameters')

    def load(self):
        self.load_state_dict(torch.load('parameters'))


model = NeuralNet(input_size, hidden_size, classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Training loop
n_total_steps = len(train_loader)
def train():
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)

            #Forward pass
            prediction = model.forward(images)
            loss = criterion(prediction, labels)

            #Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i) % 100 == 0:
                print(f'Epoch: {epoch + 1}/{epochs}, step: {i}/{n_total_steps}, loss: {loss.item():.6f}')

train()

#Testing loop
def test():
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        printed = True
        subplot = 1
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model.forward(images)

            #value, index
            _, prediction = torch.max(outputs, 1)
            if printed:
                for i in range(len(prediction)):
                    if labels[i] == prediction[i] and subplot < 26:
                        image = images[i].reshape([28,28]).cpu().detach().numpy()
                        pyplot.subplot(5, 5, subplot)
                        pyplot.imshow(image, cmap='gray')
                        pyplot.title(f'Label: {labels[i].item()}, Prediction: {prediction[i].item()}')
                        pyplot.axis('off')
                        subplot += 1
                        printed = False
                pyplot.show()

            n_samples += labels.shape[0]
            n_correct += (prediction == labels).sum().item()

        accuracy = 100.0 * n_correct / n_samples
        print(accuracy)
test()
model.save()
model.first()
model.display()
