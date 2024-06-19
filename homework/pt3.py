import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2

# 加载已经训练好的模型
model = load_model('pneumonia_detection_model.h5')


# 定义函数来进行肺炎诊断
def pneumonia_diagnosis(image_path):
    img = cv2.imread(image_path, 0)  # 灰度读取图像
    img = cv2.resize(img, (64, 64))  # 调整大小
    img_array = img_to_array(img) / 255.0  # 归一化
    img_array = np.expand_dims(img_array, axis=0)  # 增加一个维度以符合模型输入要求

    prediction = model.predict(img_array)

    if prediction > 0.5:
        return "该图像患有肺炎。"
    else:
        return "该图像正常。"


# 定义图形化界面
class PneumoniaDiagnosisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("肺炎诊断")

        self.label = tk.Label(root, text="请上传胸部X光片")
        self.label.pack(pady=10)

        self.upload_button = tk.Button(root, text="上传图片", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.display_image(file_path)
            result = pneumonia_diagnosis(file_path)
            self.result_label.config(text=result)

    def display_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((200, 200), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)

        self.image_label.config(image=img)
        self.image_label.image = img


# 创建主窗口
root = tk.Tk()
app = PneumoniaDiagnosisApp(root)
root.mainloop()
