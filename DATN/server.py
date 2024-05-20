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

model = tf.keras.models.load_model("my_model_2.h5")

# Load label
with open("label.txt") as f:
    content = f.readlines()
label = []
for i in content:
    label.append(i[:-1])


def classify_image(image_file):
    x = []
    image_file = image_file.resize((224, 224), Image.Resampling.LANCZOS)
    x = tf.keras.utils.img_to_array(image_file)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    return np.argmax(pred)


app = FastAPI()


@app.get("/")
def index():
    return {"detail": "Hello"}


@app.post("/file")
async def predict_img(file: UploadFile = File(...)):
    contents = await (file.read())
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    position = classify_image(image)
    return {"result: ": label[position]}

