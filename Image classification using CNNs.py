# *** Pakages ***
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split
import torch
import torch.nn as nn
import math
import torch.optim as optim
import torch.nn.functional as F


def out_dimensions(conv_layer, h_in, w_in):
    '''
    This function computes the output dimension of each convolutional layers in the most general way.
    '''
    h_out = math.floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) /
                  conv_layer.stride[0] + 1)
    w_out = math.floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) /
                  conv_layer.stride[1] + 1)
    return h_out, w_out


class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        # First Convolution Block with Batch Normalization
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second Convolution Block with Batch Normalization
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Fully Connected Layer with Dropout
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First Convolution Block
        x = F.gelu(self.conv1_bn(self.conv1(x)))
        x = F.gelu(self.conv2_bn(self.conv2(x)))
        x = self.pool1(x)

        # Second Convolution Block
        x = F.gelu(self.conv3_bn(self.conv3(x)))
        x = F.gelu(self.conv4_bn(self.conv4(x)))
        x = self.pool2(x)

        # Flatten
        x = x.view(-1, 256 * 8 * 8)

        # Fully Connected Layers with dropout
        x = F.gelu(self.fc1_bn(self.fc1(x)))
        x=self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Set the seed for reproducibility
    manual_seed = 42
    torch.manual_seed(manual_seed)

    # Loading and Inspect the Data with some augmentation to have better generalization
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.RandomCrop(32, padding=2),
        transforms.ToTensor(),  # Converting to tensor after applying augmentations
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # Showing one image per class
    def imshow(img):
        img = img * torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)  # Unnormalizing for image visualization
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    def show_one_image_per_class():
        seen_classes = set()
        fig = plt.figure(figsize=(12, 8))
        for images, labels in trainloader:
            for i in range(len(labels)):
                label = labels[i].item()
                if label not in seen_classes:
                    seen_classes.add(label)
                    img = images[i]
                    plt.subplot(2, 5, label + 1)
                    plt.axis('off')
                    plt.title(classes[label])
                    imshow(img)
                if len(seen_classes) == len(classes):
                    return


    show_one_image_per_class()


    # Plotting the histogram of the dataset distribution
    def plot_dataset_distribution(dataset, dataset_type):
        labels = [sample[1] for sample in dataset]
        plt.hist(labels, bins=np.arange(len(classes) + 1) - 0.5, rwidth=0.8, alpha=0.7, color='b')
        plt.xlabel('Classes')
        plt.ylabel('Frequency')
        plt.title(f'{dataset_type} Set Class Distribution')
        plt.xticks(np.arange(len(classes)), classes, rotation=45)
        plt.grid(axis='y', linestyle='--')
        plt.show()


    # Plotting the training set distribution
    plot_dataset_distribution(trainset, 'Training')

    # Plotting the test set distribution
    plot_dataset_distribution(testset, 'Test')

    # Split test set into validation and test sets
    val_size = len(testset) // 2
    test_size = len(testset) - val_size
    valset, testset = random_split(testset, [val_size, test_size])

    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)


    # Model, Loss, and Optimizer
    net = ImprovedCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)

    # Training Loop
    epochs = 6
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        running_loss = 0.0
        correct_train, total_train = 0, 0

        # Training Phase
        net.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Record statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(trainloader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.3f}, Train Accuracy: {train_acc:.2f}%')

        # Validation Phase
        net.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(valloader)
        val_acc = 100 * correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f'Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_acc:.2f}%')

    print('Finished Training')

    # Plot Train and Validation Losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Test Phase
    net.eval()
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()


    # Lists to store losses
    train_losses = []
    val_losses = []

    num_epochs=6
    for epoch in range(num_epochs):
        # Training phase
        net.train()
        running_train_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(trainloader)
        train_losses.append(avg_train_loss)

        # Validation phase
        net.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Test Phase
        net.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_acc = 100 * correct_test / total_test
        print(f'Test Accuracy: {test_acc:.2f}%')

    for seed in range(5,10):
        torch.manual_seed(seed)
        print("Seed equal to ", torch.random.initial_seed())
    




