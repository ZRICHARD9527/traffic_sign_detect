import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator
from detect.util import preprocessing

path = "../data/myData"  # 数据集文件夹
labelFile = '../data/labels.csv'  # 所有类别
batch_size_val = 50  # 每次处理数据量
steps_per_epoch_val = 446  # 每个周期迭代次数 x_train[0]/batch_size_val
epochs_val = 10  # 整个训练集训练次数
image_dimensions = (224, 224, 3)  # 图片为32*32的彩色图
testRatio = 0.2  # 测试集占比
validationRatio = 0.2  # 验证集占比
WEIGHT_DECAY = 0.005  # 权重衰减
###################################################


# 加载图像与标签
count = 0
images = []  # 图片
classNo = []  # 标签
myList = os.listdir(path)  # 得到路径下所有文件
class_num = len(myList)  # 总类别数
print("Total Classes Detected:", class_num)
print("Importing Classes.....")

# 遍历所有种类
for x in range(0, len(myList)):
    myPicList = os.listdir(path + "/" + str(count))  # 种类下所有文件的路径
    # 遍历所有图片
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)
        curImg = cv2.resize(curImg, (224, 224), interpolation=cv2.INTER_LINEAR)
        # 要缩小图像，通常使用 INTER_AREA 插值看起来效果最好，而使用 .放大图像，通常使用 INTER_CUBIC（慢）或 INTER_LINEAR 效果最好。
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")
    count += 1
print(" ")
# 保存对应的图片和标签
images = np.array(images)
classNo = np.array(classNo)

print("****************************************************8")

# 分割测试集和验证集
# X_train 训练的图像 y_train 种类标签
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
# 从训练集中分出验证集
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

print("Data Shapes")
print("Train", end="")
print(X_train.shape, y_train.shape)
print("Validation", end="")
print(X_validation.shape, y_validation.shape)
print("Test", end="")
print(X_test.shape, y_test.shape)
assert (X_train.shape[0] == y_train.shape[
    0]), "The number of images is not equal to the number of labels in training set"
assert (X_validation.shape[0] == y_validation.shape[
    0]), "The number of images is not equal to the number of labels in validation set"
assert (X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels in test set"
print(X_train.shape[1:])
assert (X_train.shape[1:] == image_dimensions), " The dimensions of the Training images are wrong "
assert (X_validation.shape[1:] == image_dimensions), " The dimensions of the Validation images are wrong "
assert (X_test.shape[1:] == image_dimensions), " The dimensions of the Test images are wrong"

# 读取标签文件
data = pd.read_csv(labelFile)
print("data shape ", data.shape, type(data))

# 可视化部分图标及类别
sample_num = []
cols = 5
num_classes = class_num
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["Name"])
            sample_num.append(len(x_selected))

# 对类别分布做一个统计图
print(sample_num)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), sample_num)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

# 对所有数据进行预处理
X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

# 增加一维 因为使用图像增强后fit函数的参数为四维数据（）
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# 数据增强 扩充数据集大小，增强模型的泛化能力
dataGen = ImageDataGenerator(width_shift_range=0.1,  # 随机宽度偏移量
                             height_shift_range=0.1,  # 随机高度偏移量
                             zoom_range=0.2,  # 随机缩放范围
                             shear_range=0.1,  # 剪切-剪切角度-逆时针剪切
                             rotation_range=10)  # 随机旋转的角度范围
dataGen.fit(X_train)

batches = dataGen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)  # 迭代

# 展示强化后的图像案例
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(image_dimensions[0], image_dimensions[1]))
    axs[i].axis('off')
plt.show()

# 将类别向量转换为二进制（one-hot）形式
y_train = to_categorical(y_train, class_num)
y_validation = to_categorical(y_validation, class_num)
y_test = to_categorical(y_test, class_num)


# vgg16 模型
def vgg_16(weight_decay=WEIGHT_DECAY):
    model = Sequential()  # 顺序模型

    """layer1 224*224*3"""
    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=(224, 224, 3), kernel_regularizer=regularizers.l2(weight_decay)))
    # kernel_regularizer设置权值的正则化方法 这里使用l2即权重衰减法
    model.add(Activation('relu'))
    # 机器学习的一个重要假设就是训练数据和测试数据时满足相同分布的；批标准化使得训练过程中每一层神经网络的输入保持相同分布；
    # 使得输入值落入对激活函数比较敏感的区域，避免梯度消失，加速收敛
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    """layer2 224*224*64"""
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # 最大池化层
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    """layer3  112*112*64"""
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    """layer4 112*112*128"""
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    """layer5 56*56*128"""
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    """layer6 56*56*256"""
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    """layer7 56*56*256"""
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    """layer8 28*28*256"""
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    """layer9 28*28*512"""
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    """layer10 28*28*512"""
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    """layer11 14*14*512"""
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    """layer12 14*14*512"""
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    """layer13 14*14*512"""
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    """layer14 7*7*512"""
    model.add(Flatten())  # 将矩阵转换为1维数组
    model.add(Dense(4096, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    """layer15 4096"""
    model.add(Dense(4096, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    """layer16 4096"""
    model.add(Dropout(0.5))
    model.add(Dense(1000))
    model.add(Activation('softmax'))

    model.summary()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# 调用模型
model = vgg_16()
print(model.summary())

# 开始训练
history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                    steps_per_epoch=steps_per_epoch_val, epochs=epochs_val,
                    validation_data=(X_validation, y_validation), shuffle=1)

# 画图
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
# 开始评估模型
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])  # 损失值
print('Test Accuracy:', score[1])  # 精确度

# 保存模型
model.save('vgg16.h5')
