# -*- coding: UTF-8 -*-
"""
@Project ：base 
@File ：main.py
@Author ：AnthonyZ
@Date ：2022/6/2 14:58
"""

import argparse
import os

from tqdm import tqdm

from data import *
from utils import *
from ResNet import *


def train(args, epoch):
    running_loss = 0.0
    for index, (inputs, labels) in tqdm(enumerate(train_loader, 0), desc="Epoch " + str(epoch)):
        # get the inputs; data is a list of [inputs, labels]
        # inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.to(args.device))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if index % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {index + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0


def validate(args, loss_vector, accuracy_vector):
    net.eval()
    val_loss, correct = 0, 0
    for _, (data, target) in enumerate(test_loader):
        data = data.to(args.device)
        target = target.to(args.device)
        output = net(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(test_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(test_loader.dataset)
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(test_loader.dataset), accuracy))


def main(args, loss_vector, accuracy_vector):
    for epoch in range(args.epochs):
        train(args, epoch)
        PATH = os.path.join(args.data_path, 'cifar_net.pth')
        torch.save(net.state_dict(), PATH)
        validate(args, loss_vector, accuracy_vector)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../data", type=str, help="The input data dir")
    parser.add_argument("--batch_size", default=4, type=int, help="The batch size of training")
    parser.add_argument("--device", default='cpu', type=str, help="The training device")
    parser.add_argument("--learning_rate", default=0.0004, type=int, help="learning rate")
    parser.add_argument("--epochs", default=20, type=int, help="Training epoch")

    args = parser.parse_args()

    train_loader, test_loader, classes = cifar100_dataset(args)

    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    #
    # imshow(torchvision.utils.make_grid(images))

    net = ResNet().to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    lossv, accv = [], []
    main(args, lossv, accv)

    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, args.epochs + 1), lossv)
    plt.title('validation loss')

    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, args.epochs + 1), accv)
    plt.title('validation accuracy')




