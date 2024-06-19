#### 本次建模使用到的库 ####
import numpy as np  # 导入NumPy库用于数值计算
import pandas as pd  # 导入Pandas库用于数据处理
import seaborn as sns  # 导入Seaborn库用于数据可视化
import matplotlib.pyplot as plt  # 导入Matplotlib库用于绘图
import os  # 导入OS库用于文件和目录操作
from glob import glob  # 导入Glob模块用于文件路径匹配
import cv2  # 导入OpenCV库用于图像处理，安装时要使用 pip install opencv
from keras.models import Sequential  # 从Keras导入Sequential模型
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense  # 从Keras导入各类层
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  # 从Keras导入图像预处理工具
import warnings  # 导入警告模块
warnings.filterwarnings('ignore')  # 忽略警告信息

# 给图片添加标注，肺炎的标为1，正常的标为0。同时重新设置图片的大小。
def picture_separation(folder):
    y = []  # 用于存储标签
    x = []  # 用于存储图像数据
    image_list = []  # 用于存储图像文件名

    for foldername in os.listdir(folder):  # 遍历文件夹中的每个子文件夹
        if not foldername.startswith('.'):  # 排除隐藏文件夹
            if foldername == "NORMAL":
                label = 0  # 正常的标为0
            elif foldername == "PNEUMONIA":
                label = 1  # 肺炎的标为1
            else:
                label = 2  # 其他类别标为2

            for image_filename in os.listdir(folder + "/" + foldername):  # 遍历每个子文件夹中的图像文件
                img_file = cv2.imread(folder + "/" + foldername + '/' + image_filename, 0)  # 读取图像文件，灰度模式

                if img_file is not None:
                    img = cv2.resize(img_file, (64, 64))  # 调整图像大小为64x64
                    img_arr = img_to_array(img) / 255  # 转换图像为数组并归一化
                    x.append(img_arr)  # 添加图像数据到列表
                    y.append(label)  # 添加标签到列表
                    image_list.append(foldername + '/' + image_filename)  # 添加图像文件名到列表

    X = np.asarray(x)  # 转换列表为NumPy数组
    y = np.asarray(y)  # 转换列表为NumPy数组

    return X, y, image_list  # 返回图像数据、标签和图像文件名列表


# 输入本地图片路径
train_dir = 'D:/PycharmProjects/archive/chest_xray/train'
test_dir = 'D:/PycharmProjects/archive/chest_xray/test'
val_dir = 'D:/PycharmProjects/archive/chest_xray/val'

# 生成训练集、测试集、验证集
X_train, y_train, img_train = picture_separation(train_dir)
train_df = pd.DataFrame(img_train, columns=["images"])  # 创建DataFrame存储图像文件名
train_df["target"] = y_train  # 添加标签

X_val, y_val, img_val = picture_separation(val_dir)
val_df = pd.DataFrame(img_val, columns=["images"])  # 创建DataFrame存储图像文件名
val_df["target"] = y_val  # 添加标签

X_test, y_test, img_test = picture_separation(test_dir)
test_df = pd.DataFrame(img_test, columns=["images"])  # 创建DataFrame存储图像文件名
test_df["target"] = y_test  # 添加标签

full_data = pd.concat([train_df, test_df, val_df], axis=0, ignore_index=True)  # 合并所有数据

# 样本可视化
plt.figure(figsize=(10, 7))
img = load_img(train_dir + "/" + full_data["images"][2])  # 加载样本图像
plt.imshow(img)  # 显示图像
plt.title("NORMAL", color="green", size=14)  # 设置标题
plt.grid(color='#CCCCCC', linestyle='--')  # 设置网格线
plt.show()

plt.figure(figsize=(10, 7))
img = load_img(train_dir + "/" + full_data["images"][3010])  # 加载样本图像
plt.imshow(img)  # 显示图像
plt.title("PNEUMONIA", color="green", size=14)  # 设置标题
plt.grid(color='#CCCCCC', linestyle='--')  # 设置网格线
plt.show()

# ImageDataGenerator()是keras.preprocessing.image模块中的图片生成器，
# 可以每一次给模型“喂”一个batch_size大小的样本数据，同时也可以在每一个批次中对这batch_size个样本数据进行增强，
# 扩充数据集大小，增强模型的泛化能力。比如进行旋转，变形，归一化等等。
batch_size = 32
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 值将在执行其他处理前乘到整个图像上，我们的图像在RGB通道都是0~255的整数，
    # 这样的操作可能使图像的值过高或过低，所以我们将这个值定为0~1之间的数。
    shear_range=0.3,  # 剪切变换强度
    horizontal_flip=True,  # 水平翻转
    zoom_range=0.3  # 随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]。用来进行随机的放大。
)
test_datagen = ImageDataGenerator(rescale=1./255)  # 测试集不做数据增强，只做归一化
val_datagen = ImageDataGenerator(rescale=1./255)  # 验证集不做数据增强，只做归一化

# 从路径中获取、调整图像
train_generator = train_datagen.flow_from_directory(
    train_dir,  # 图片路径
    target_size=(64, 64),  # 图像大小 64 * 64
    batch_size=batch_size,
    color_mode="grayscale",  # 灰度图像
    class_mode="binary"  # 二分类
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="binary"
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(64, 64),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="binary"
)

# 建立模型
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:]))  # 第一层卷积层
model.add(Activation("relu"))  # ReLU激活函数
model.add(MaxPooling2D())  # 池化层

model.add(Conv2D(32, (3, 3)))  # 第二层卷积层
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3, 3)))  # 第三层卷积层
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())  # Flatten层用来将输入“压平”，即把多维的输入一维化
model.add(Dense(1024))  # 全连接层
model.add(Activation("relu"))
model.add(Dropout(0.4))  # 防止过拟合
model.add(Dense(1))  # 输出层
model.add(Activation("sigmoid"))  # 二分类

model.compile(loss="binary_crossentropy",  # 编译模型，指定损失函数和优化器
              optimizer="rmsprop",        # RMSprop作为优化器
              metrics=["accuracy"])       # 准确率作为评估指标

from tensorflow.keras.callbacks import EarlyStopping  # 导入早停回调函数
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)  # 定义早停回调

history = model.fit(
    train_generator,  # 训练数据生成器
    steps_per_epoch=5216 // batch_size,  # 每个epoch训练步数
    epochs=20,  # 训练轮数
    validation_data=val_generator,  # 验证数据生成器
    validation_steps=624 // batch_size,  # 每个epoch验证步数
    callbacks=[early_stopping]  # 早停回调
)

#### 训练集 ####
print("Accuracy of the model is - ", model.evaluate(train_generator)[1] * 100, "%")  # 打印训练集准确率
print("Loss of the model is - ", model.evaluate(train_generator)[0])  # 打印训练集损失

#### 验证集效果 ####
print("Accuracy of the model is - ", model.evaluate(val_generator)[1] * 100, "%")  # 打印验证集准确率
print("Loss of the model is - ", model.evaluate(val_generator)[0])  # 打印验证集损失

#### 测试集效果 ####
print("Accuracy of the model is - ", model.evaluate(test_generator)[1] * 100, "%")  # 打印测试集准确率
print("Loss of the model is - ", model.evaluate(test_generator)[0])  # 打印测试集损失

model.save('pneumonia_detection_model.h5')  # 保存模型
