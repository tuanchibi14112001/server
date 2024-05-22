import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
print(tf.__version__)
model = tf.keras.models.load_model("my_model_2.h5")

with open("label.txt") as f:
    content = f.readlines()
label = []
for i in content:
    label.append(i[:-1])


def classify_image(imageFile):
    x = []
    img = Image.open(imageFile)
    img.load()
    img = img.resize((224, 224), Image.Resampling.LANCZOS)

    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    print(x.shape)
    pred = model.predict(x)
    print(pred.round(3))
    index = np.argmax(pred)
    class_name = label[index]
    confidence_score = pred[0][index]
    print(class_name, confidence_score)


img_path = '../DATN/TestImage/bee.jpg'
classify_image(img_path)
