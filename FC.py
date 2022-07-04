import os

from torch import nn
from torch.autograd import Variable

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torchvision
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot  as plt
batch_size=32
learning_rate=0.005


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])
trainset = torchvision.datasets.ImageFolder('./train',transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.ImageFolder('./test',transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
valset = torchvision.datasets.ImageFolder('./validation',transform=transform)
valloader = torch.utils.data.DataLoader(valset,batch_size=batch_size,shuffle=True)
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 设计模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(32*32*3, 2048)
        self.l2 = torch.nn.Linear(2048, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # print(x.shape)
        x = x.view(-1, 32*32*3)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)




if torch.cuda.is_available():
    model = Net().cuda()

# 构建损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate,weight_decay=0.001, momentum=0.9)
traincost =[]
valcost =[]
acc =[]



# 定义训练函数
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(trainloader,0):
        inputs, label = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            label = label.cuda()
        # 前馈+反馈+更新
        outputs = model(inputs)
        loss = criterion(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 300 == 299:
            print('[%d,%5d] Traning loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))

            running_loss = 0.0
    traincost.append(running_loss/300)

# 定义测试函数
def test():
    running_loss = 0.0
    correct = 0
    total = 0
    listacc = []
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # for label, prediction in zip(labels, predicted):
            #     if label == prediction:
            #         correct_pred[classes[label]] += 1
            #     total_pred[classes[label]] += 1
            # # print accuracy for each class
            # for classname, correct_count in correct_pred.items():
            #     accuracy = 100 * float(correct_count) / total_pred[classname]
            #     listacc.append(accuarcy)
            #     print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        print('loss on testset:%.3f' % ( running_loss / 300))
        valcost.append(running_loss/300)
    print('Accuracy on test set:%d %%' % (100 * correct / total))
    acc.append(100 * correct / total)

# 实例化训练和测试
if __name__ == '__main__':
    for epoch in range(50):
        train(epoch)
        # test(valloader)
        test()

    print(traincost)
    print(valcost)
    print(acc)
    plt.plot(traincost, color='g', label="traincost")
    plt.plot(valcost, color='b', label="valcost")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(' Net')
    plt.legend()
    plt.show()

    plt.plot(acc)
    plt.title(' accaurcy')
    plt.show()

    running_loss = 0.0
    correct = 0
    total = 0
    listacc = []
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
            # print accuracy for each class
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                listacc.append(accuracy)
                print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        print('loss on testset:%.3f' % (running_loss / 300))
    print('Accuracy on test set:%d %%' % (100 * correct / total))

    print(traincost)
    print(valcost)
    print(acc)
    plt.plot(traincost, color='g', label="traincost")
    plt.plot(valcost, color='b', label="valcost")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(' Net')
    plt.legend()
    plt.show()

    plt.plot(acc)
    plt.title(' accaurcy')
    plt.show()






