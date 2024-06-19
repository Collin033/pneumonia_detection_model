from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2

# 加载已经训练好的模型
model = load_model('pneumonia_detection_model.h5')


# 定义函数来进行肺炎诊断
def pneumonia_diagnosis(image_path):
    # 加载待诊断的图像并进行预处理
    img = cv2.imread(image_path, 0)  # 灰度读取图像
    img = cv2.resize(img, (64, 64))  # 调整大小
    img_array = img_to_array(img) / 255.0  # 归一化
    img_array = np.expand_dims(img_array, axis=0)  # 增加一个维度以符合模型输入要求

    # 使用模型进行预测
    prediction = model.predict(img_array)

    # 解释预测结果
    if prediction > 0.5:
        return "该图像患有肺炎。"
    else:
        return "该图像正常。"


# 指定待诊断的图像路径
image_path = 'person3_virus_16.jpeg'

# 进行肺炎诊断
result = pneumonia_diagnosis(image_path)
print(result)
