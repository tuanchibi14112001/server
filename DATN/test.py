import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
print(tf.__version__)
model = tf.keras.models.load_model("mobilenet_v3_best_model.h5")
new_model = tf.keras.models.load_model("mobilenet_v3_2_best_model.h5")

with open("label.txt") as f:
    content = f.readlines()
label = []
for i in content:
    label.append(i[:-1])

with open("new_label.txt") as f:
    content = f.readlines()
new_label = []
for i in content:
    new_label.append(i[:-1])


def classify_image(imageFile):
    x = []
    img = Image.open(imageFile)
    img.load()
    img = img.resize((224, 224), Image.Resampling.LANCZOS)

    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    print(x.shape)
    pred = model.predict(x)
    print(pred.round(2))
    index = np.argmax(pred)
    class_name = label[index]
    confidence_score = pred[0][index]
    print(class_name, confidence_score)

def new_classify_image(imageFile):
    className = []
    x = []
    img = Image.open(imageFile)
    img.load()
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    pred = new_model.predict(x)
    print(pred[0].round(3))

    indices = np.argsort(pred[0])[-3:]
    three_largest_elements = pred[0][indices]
    sorted_indices = indices[np.argsort(-three_largest_elements)]
    for index in sorted_indices:
        if pred[0][index].round(3) > 0.:
            className.append(new_label[index])
    print(className)

img_path = '../DATN/TestImage/x8.jpg'
# classify_image(img_path)
new_classify_image(img_path)
