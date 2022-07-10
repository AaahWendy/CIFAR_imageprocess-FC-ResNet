import os
import shutil
import time

import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
from shutil import copy2
from PIL import Image
from skimage import util
import matplotlib.pyplot as plt

import time
from PIL import Image
from multiprocessing import Pool

if not os.path.exists('./data'):
    os.mkdir('./data')
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=None)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=None)
classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 转成图片
def ToImage(path):
    if not os.path.exists('./Image'):
        os.mkdir('./Image')
        for i in range(1, 6):
            data_name = path + '/' + 'data_batch_' + str(i)
            data_dict = unpickle(data_name)
            print(data_name + ' is processing')
            for j in range(10000):
                img = np.reshape(data_dict[b'data'][j], (3, 32, 32))
                img = np.transpose(img, (1, 2, 0))
                # 通道顺序为RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 要改成不同的形式的文件只需要将文件后缀修改即可
                img_name = './Image/' + str(data_dict[b'labels'][j]) + str((i) * 10000 + j) + '.jpg'
                cv2.imwrite(img_name, img)
            print(data_name + ' is done')
        test_data_name = path + '/test_batch'
        print(test_data_name + ' is processing')
        test_dict = unpickle(test_data_name)

        for m in range(10000):
            img = np.reshape(test_dict[b'data'][m], (3, 32, 32))
            img = np.transpose(img, (1, 2, 0))
            # 通道顺序为RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 要改成不同的形式的文件只需要将文件后缀修改即可
            img_name = './Image/' + str(test_dict[b'labels'][m]) + str(10000 + m) + '.jpg'
            cv2.imwrite(img_name, img)
        print(test_data_name + ' is done')
        print('Finish transforming to image')

# 分为三个文件夹
def divided():
    datadir_normal = "./Image"
    all_data = os.listdir(datadir_normal)
    num_all_data = len(all_data)
    print("num_all_data: " + str(num_all_data))
    index_list = list(range(num_all_data))
    # print(index_list)
    random.shuffle(index_list)
    num = 0

    trainDir = './train'
    if not os.path.exists(trainDir):
        os.mkdir(trainDir)

    validDir = './validation'
    if not os.path.exists(validDir):
        os.mkdir(validDir)

    testDir = './test'
    if not os.path.exists(testDir):
        os.mkdir(testDir)

    for i in index_list:
        fileName = os.path.join(datadir_normal, all_data[i])
        if num < num_all_data * 0.8:
            # print(str(fileName))
            copy2(fileName, trainDir)
        elif num > num_all_data * 0.8 and num < num_all_data * 0.9:
            # print(str(fileName))
            copy2(fileName, validDir)
        else:
            copy2(fileName, testDir)
        num += 1
    shutil.rmtree("./Image")



#查找文件夹中所有文件
def findAllFile(base):
    img_path = []
    for i in os.listdir(base):
        path = base + '/' + i
        img_path.append(path)
    return img_path

# 图片处理
def Imageprocess(path):
    # for i in os.listdir(base):
    #     path = base+'/'+i
        # 反转
        original = Image.open(path)
        im_inverted = original.point(lambda _: 255 - _)

        # 高斯模糊
        img_gu = cv2.GaussianBlur(np.array(im_inverted), (5 ,5),0)

        #高斯噪声


        # sig = random(0, 50)
        # np.random.shuffle(sig)
        noise = np.random.normal(0,0, img_gu.shape)
        Gimg = img_gu.astype(np.float)
        Gimg = Gimg + noise
        Gimg = np.clip(Gimg, 0, 255)
        GuassImg = Gimg.astype(np.uint8)

        #x和y梯度
        sobelx = cv2.Sobel(GuassImg, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.Sobel(GuassImg, cv2.CV_64F, 0, 1, ksize=3)
        sobely = cv2.convertScaleAbs(sobely)
        sobelxy2 = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

        plt.axis("off")
        plt.imshow(sobelxy2[:, :, ::-1])
        # plt.savefig(path, bbox_inches='tight',pad_inches=0)



        # img = cv2.bitwise_not(sobelxy2)
        cv2.imwrite(path,sobelxy2 )
        print(path)

        # plt.subplot(2, 3, 1), plt.title('original')
        # plt.imshow(original)
        # plt.subplot(2, 3, 2), plt.title('invert')
        # plt.imshow(im_inverted)
        # plt.subplot(2, 3, 3), plt.title('Guass')
        # plt.imshow(img_gu)
        # plt.subplot(2, 3, 4), plt.title('adGuassnoise')
        # plt.imshow(GuassImg)
        # plt.subplot(2, 3, 5), plt.title('sobel')
        # plt.imshow(sobelxy2)
        #
        # plt.tight_layout()
        # plt.savefig(path)
        # plt.show()
def  classify(path):
    for i in os.listdir(path):
        lable = i[0:1]
        if not os.path.exists(path + '/'+lable):
            os.mkdir(path + '/'+lable)
        src = path + '/' + i
        dst = path +'/'+lable +'/' + i
        shutil.move(src, dst)
        print("success move" + src)


if __name__ == '__main__':
    start = time.time()
    # data to image

    # ToImage('./data/cifar-10-batches-py')
    # print("success to image")
    #
    # divided();
    # print("Finish divide")

    # path = './ceshi'
    # img_path = findAllFile(path)
    # pool = Pool(6)
    # pool.map(Imageprocess, img_path)
    # pool.close()
    # pool.join()
    path = './train'
    img_path = findAllFile(path)
    pool = Pool(6)
    pool.map(Imageprocess, img_path)
    pool.close()
    pool.join()
    #
    pathtest = './test'
    img_path1 = findAllFile(pathtest)
    pool = Pool(6)
    pool.map(Imageprocess, img_path1)
    pool.close()
    pool.join()

    pathval = './validation'
    img_path2 = findAllFile(pathval)
    pool = Pool(6)
    pool.map(Imageprocess, img_path2)
    pool.close()
    pool.join()

    classify('./test')
    classify('./train')
    classify('./validation')

    end = time.time()

    print(end - start)



