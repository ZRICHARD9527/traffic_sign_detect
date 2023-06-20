import numpy as np
import cv2
import tensorflow as tf

from detect.util import preprocessing, getCalssName

'''
This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions 
in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

解决方法：
AVX（Advanced Vector Extensions-Intel® AVX) 是intel 优化CPU用于浮点计算的技术,如果有GPU了，
其实不用考虑该警告讯息。 不过， 不管怎么说， 如果不愿意看到该警告讯息， 可以加上如下2行代码：
'''
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

threshold = 0.75  # 概率阈值
font = cv2.FONT_HERSHEY_SIMPLEX

# 导入训练好的模型参数
model = tf.keras.models.load_model('traffic.h5')


# 图片预处理
def pres(img_origin):
    img = np.asarray(img_origin)

    # 网络输入图片指定32*32
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)

    # 预测
    predictions = model.predict(img)
    class_index = np.argmax(predictions, axis=-1)  # 概率最大的下标

    probability_value = predictions[0][class_index]  # 预测概率

    print("this picture has  a ", probability_value, " probability of belonging to class",
          getCalssName(class_index))
    # 概率大于阈值才判断有效检测
    if probability_value > threshold:
        print("this picture belongs to ", getCalssName(class_index))
    else:
        print("this picture does not belong to any category")
        return "No"


if __name__ == '__main__':
    # 读取图片
    img_origin = cv2.imread('../example/img3.png')
    out = pres(img_origin)
