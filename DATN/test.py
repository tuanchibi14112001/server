import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
print(tf.__version__)
model = tf.keras.models.load_model("my_model_2.h5")

def classify_image(imageFile):
  x = []
  img = Image.open(imageFile)
  img.load()
  img = img.resize((224,224),Image.Resampling.LANCZOS)

  x = tf.keras.utils.img_to_array(img)
  x = np.expand_dims(x, axis = 0)
  print(x.shape)
  pred = model.predict(x)
  catagory = np.argmax(pred, axis = 1)
  print(catagory)

img_path = 'cua.jpg'
classify_image(img_path)


