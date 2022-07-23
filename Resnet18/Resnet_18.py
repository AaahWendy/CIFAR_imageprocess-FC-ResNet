import csv

from Resnet18.resnet import ResNet18
from Resnet18.plain import Plain18
# Use the ResNet18 on Cifar-10
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')  # 输出结果保存路径
args = parser.parse_args()

# set hyperparameter
EPOCH = 20
pre_epoch = 0
BATCH_SIZE = 128
LR = 0.01

# prepare dataset and preprocessing
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.ImageFolder('../train',transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE, shuffle=True)
testset = torchvision.datasets.ImageFolder('../test',transform=transform_test)
testloader = torch.utils.data.DataLoader(testset,batch_size=100, shuffle=False)
valset = torchvision.datasets.ImageFolder('../validation',transform=transform_test)
valloader = torch.utils.data.DataLoader(valset,batch_size=100, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# define ResNet18 模型
net = ResNet18().to(device)
# net = Plain18().to(device)

# define loss funtion & optimizer # 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

# train
if __name__ == '__main__':
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    best_acc = 85  # 2 初始化best test accuracy
    print("Start Training, Resnet-18!")  #
    with open("acc.txt", "w") as f:
        with open("type.csv","w") as ff:
            with open("log.txt", "w") as f2:
                for epoch in range(pre_epoch, EPOCH):  # 循环训练回合，每回合会以批量为单位训练完整个训练集，一共训练EPOCH个回合
                    print('\nEpoch: %d' % (epoch + 1))
                    net.train()
                    sum_loss = 0.0
                    correct = 0.0
                    total = 0.0
                    for i, data in enumerate(trainloader, 0):
                        # prepare dataset
                        length = len(trainloader)
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()
                        # forward & backward
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        sum_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        correct += predicted.eq(labels.data).cpu().sum()
                        total += labels.size(0)
                        Train_epoch_loss = sum_loss / (i + 1)
                        Train_epoch_acc = 100. * correct / total
                        print('epoch:%d | iter:%d | Loss: %.03f | top1Acc: %.3f%% '
                              % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total
                                 ))
                        f2.write('epoch:%03d | iter:%05d | Loss: %.03f | top1Acc: %.3f%% '
                                 % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total
                                    ))
                        f2.write('\n')
                        f2.flush()
                    print('Waiting Test...')
                    with torch.no_grad():
                        correct = 0.0
                        total = 0.0
                        class_correct = list(0. for i in range(10))
                        class_total = list(0. for i in range(10))
                        typeacc=[]
                        for data in valloader:
                            net.eval()
                            images, labels = data
                            images, labels = images.to(device), labels.to(device)
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum()
                            c = (predicted == labels).squeeze()
                            for i in range(len(images)):
                                label = labels[i]  # 对各个类的进行各自累加
                                class_correct[label] += c[i]
                                class_total[label] += 1

                        print('test acc: %.3f%%' % (100 * correct / total))
                        top1Acc = 100. * correct / total
                        for i in range(10):
                            print('Accuracy of %5s : %.2f %%' % (
                                classes[i], 100 * class_correct[i] / class_total[i]))
                            # typeacc.append((100 * class_correct[i] / class_total[i]).cuda().cpu().numpy())
                            ff.write("%.3f%%," %(100 * class_correct[i] / class_total[i]))
                        ff.write('\n')
                        ff.flush()
                        print('Saving model......')
                        torch.save({'state_dict': net.state_dict()}, 'Resnet18.pth.tar')
                        f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, top1Acc))
                        f.write('\n')
                        f.flush()



            print("Training Finished, TotalEPOCH=%d" % EPOCH)
