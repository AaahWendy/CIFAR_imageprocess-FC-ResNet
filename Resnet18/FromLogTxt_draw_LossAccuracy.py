import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator

x = []
loss_y = []
acc_y = []
Top3acc_y = []

with open("log.txt") as f:
    for line in f:
        line = line.strip()  # str
        # print(line)
        if len(line.split(" | ")) == 4:
            x.append(float(line.split(" | ")[1].split(':')[1]))  # iter
            loss_y.append(float(line.split(" | ")[2].split(':')[1]))  # Loss
            acc_y.append(float(line.split(" | ")[3].split(':')[1].split('%')[0]))  # acc

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, loss_y, '-', label='loss', color='g')
ax2 = ax.twinx()  # 使用twinx()，得到与ax1 对称的ax2,共用一个x轴，y轴对称（坐标不对称）
ax2.plot(x, acc_y, '-', label='acc', color='r')
ax.legend(loc='upper left')
ax.grid()
plt.title('loss_acc')
ax.set_xlabel("iter")
ax.set_ylabel("loss")
ax2.set_ylabel("acc")
ax.set_ylim(0, 2.5)
ax2.set_ylim(0, 100)
ax2.legend(loc='upper right')
plt.savefig('loss_acc.png')
plt.show()

epoch = []
acc = []

with open("acc.txt") as f:
    for line in f:
        line = line.strip()  # str
        # print(line)
        epoch.append(float(line.split("=")[1].split(',')[0]))  # epoch
        acc.append(float(line.split("=", 2)[2].split('%')[0]))  # acc

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
plt.title('test_acc')
plt.xlabel('epoch')
plt.ylabel('acc')

plt.plot(epoch, acc, label="acc")
plt.legend(loc='upper right')
plt.savefig('test_acc.png')
plt.show()