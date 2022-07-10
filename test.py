import torch
import FC

model = FC.Net().cuda()
checkpoint = torch.load('net.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
model.eval()
with torch.no_grad():
    for data in FC.testloader:
        images, labels = data
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        c = (predicted == labels).squeeze()
        for i in range(len(images)):  # 因为每个batch都有4张图片，所以还需要一个4的小循环
            label = labels[i]  # 对各个类的进行各自累加
            class_correct[label] += c[i]
            class_total[label] += 1
print('Accuracy on test set:%d %%' % (100 * correct / total))
for i in range(10):
    print('Accuracy of %5s : %.2f %%' % (FC.classes[i], 100 * class_correct[i] / class_total[i]))
