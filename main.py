from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import io


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preproc(img):
    img = img / 255.0
    img = tf.image.resize(img, [128, 128])
    return img


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    model = tf.keras.models.load_model("mymodelv1.2.h5")
    contents = await image.read()

    img = Image.open(io.BytesIO(contents))
    # image_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    image_array = np.array(img)

    image_array = np.flip(image_array, axis=2)
    # image_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # img_bgr = np.array(img)

    # # Convert BGR to RGB format
    # img_rgb = np.transpose(img_bgr, (2, 0, 1))

    # Display the converted image
    # image_array = np.array(img_rgb)
    image_array = preproc(image_array)
    image_array = tf.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)

    predictions = predictions.flatten().tolist()

    return JSONResponse(content={"filename": image.filename, "predictions": predictions})
