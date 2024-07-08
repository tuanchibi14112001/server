import io

import PIL.Image
import fastapi
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# model = tf.keras.models.load_model("mobilenet_v3_best_model.h5")
model = tf.keras.models.load_model("mobilenet_v3_2_best_model.h5")

# Load label
with open("new_label.txt") as f:
    content = f.readlines()
label = []
for i in content:
    label.append(i[:-1])


def classify_image(image_file):
    x = []
    class_name = []

    image_file = image_file.resize((224, 224), Image.Resampling.LANCZOS)
    x = tf.keras.utils.img_to_array(image_file)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    accuracy = max(pred[0]).astype(float).round(3)
    indices = np.argsort(pred[0])[-3:]
    three_largest_elements = pred[0][indices]
    sorted_indices = indices[np.argsort(-three_largest_elements)]
    for index in sorted_indices:
        # if pred[0][index].round(3) > 0.:
        print(label[index], pred[0][index].round(3))
        class_name.append(label[index])
    return {"result": class_name[0], "similar": class_name[1:], "accuracy": accuracy}


app = FastAPI()


@app.get("/")
def index():
    return {"detail": "Hello"}


@app.post("/file")
async def predict_img(file: UploadFile = File(...)):
    contents = await (file.read())
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    return classify_image(image)


@app.post("/filename")
async def predict(filename: UploadFile = File(...)):
    return {"detail": filename.filename}
