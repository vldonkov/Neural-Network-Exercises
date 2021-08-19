# from six.moves import urllib
# opener = urllib.request.build_opener()
# opener.addheaders = [('User-agent', 'Mozilla/5.0')]
# urllib.request.install_opener(opener)

import torch
import torchvision

use_convolutional_network = True
K = 100
device = torch.device("cuda")


class FullyConnected(torch.nn.Module):
    def __init__(self):
        # call base class constructor
        super(FullyConnected, self).__init__()

        # initialize components
        self.fc1 = torch.nn.Linear(28 * 28, K, bias=True)
        self.activation = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(K, 10)
        print(f"Initialized a fully-connected network with {K} hidden units")

    def forward(self, x):
        # forward through layers
        return self.fc2(self.activation(self.fc1(x)))


class Convolutional(torch.nn.Module):
    def __init__(self):
        # call base class constructor
        super(Convolutional, self).__init__()

        # define network structure
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=64, kernel_size=(5, 5), stride=1,
                                     padding=2)
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.activation = torch.nn.Sigmoid()
        self.fc1 = torch.nn.Linear(7 * 7 * 64, 10, bias=True)
        print("Initialized a convolutional network")

    def forward(self, x):
        # forward through layers
        a = self.activation(self.pool(self.conv1(x)))
        a = self.activation(self.pool(self.conv2(a)))
        a = torch.flatten(a, 1)
        return self.fc1(a)


# define transform for different networks
transform = torchvision.transforms.ToTensor() \
    if use_convolutional_network else \
    torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torch.flatten
    ])

# training set: MNIST
train_set = torchvision.datasets.MNIST(
    root="C:/Users/nightmare/Documents/UZH/Semester 3/03_Deep Learning/02_Exercises/Python/Ex6",
    train=True,
    download=True,
    transform=transform
)
# here batch size does not matter, we can use as much as fits on the GPU (or CPU)
# data loader with batch size 32
train_loader = torch.utils.data.DataLoader(
    train_set,
    shuffle=True,
    batch_size=32
)

# test set: MNIST
test_set = torchvision.datasets.MNIST(
    root="C:/Users/nightmare/Documents/UZH/Semester 3/03_Deep Learning/02_Exercises/Python/Ex6",
    train=False,
    download=True,
    transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    shuffle=False,
    batch_size=32
)

# create network
network = Convolutional().to(device) if use_convolutional_network else FullyConnected().to(device)
# Cross-entropy loss with softmax activation
loss = torch.nn.CrossEntropyLoss()
# stochastic gradient descent
optimizer = torch.optim.SGD(
    params=network.parameters(),
    lr=0.01, momentum=0.9
)

# train network for a few epochs
for epoch in range(10):
    # iterate over training batches
    for x, t in train_loader:
        optimizer.zero_grad()
        # forward batch through network and obtain logits
        z = network(x.to(device))
        # compute the loss
        J = loss(z, t.to(device))
        # compute the gradient via automatic differentiation
        J.backward()
        # perform weight update
        optimizer.step()

    # compute test accuracy
    correct = 0
    with torch.no_grad():
        for x, t in test_loader:
            # compute logits for batch
            z = network(x.to(device))
            # compute the index of the largest logits per sample
            _, y = torch.max(z.data, 1)
            # compute how often the correct index was predicted
            correct += (y == t.to(device)).sum().item()

    # print epoch and accuracy
    print(F"Epoch {epoch + 1}; test accuracy: {correct / len(test_set) * 100.:1.2f} %")

print()
