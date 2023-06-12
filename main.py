from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import numpy as np
import tensorflow as tf
from PIL import Image
import io
from base64 import b64encode, b64decode
from pydantic import BaseModel


app = FastAPI()
fresh_model = tf.keras.models.load_model("mymodelv1.3.h5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
                   "https://ml-fruit-freshness-classifier.vercel.app", 
                   "https://ml-fruit-freshness-classifier-git-crazy-smtnhacker.vercel.app/"],
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
    contents = await image.read()

    img = Image.open(io.BytesIO(contents))
    image_array = np.array(img)
    image_array = np.flip(image_array, axis=2)
    image_array = preproc(image_array)
    image_array = tf.expand_dims(image_array, axis=0)
    predictions = fresh_model.predict(image_array)

    predictions = predictions.flatten().tolist()

    return JSONResponse(content={"filename": image.filename, "predictions": predictions})

class FrameData(BaseModel):
    frame: str

@app.post("/frame")
async def upload_pic(i: FrameData):
    x = i.frame.replace('data:image/jpeg;base64,', '')
    x = io.BytesIO(b64decode(x))
    
    image_array= Image.open(x)
    image_array = np.array(image_array)
    image_array = np.flip(image_array, axis=2)
    image_array = preproc(image_array)
    image_array = tf.expand_dims(image_array, axis=0)
    predictions = fresh_model.predict(image_array)

    predictions = predictions.flatten().tolist()

    return JSONResponse(content={"predictions": predictions})

@app.post("/stream")
async def upload_frame(i: FrameData):

    x = i.frame.replace('data:image/jpeg;base64,', '')
    x = io.BytesIO(b64decode(x))
    img = Image.open(x)
    img = np.array(img)
    img = preproc(img)
    img = tf.expand_dims(img, axis=0)
    predictions = fresh_model.predict(img)

    predictions = predictions.flatten().tolist()

    return JSONResponse(content={'predictions': predictions})

@app.post("/simple_stream")
async def upload_frame(i: FrameData):

    x = i.frame.replace('data:image/jpeg;base64,', '')
    x = io.BytesIO(b64decode(x))
    img = Image.open(x)
    img = np.array(img)
    img = preproc(img)
    img = tf.expand_dims(img, axis=0)
    predictions = fresh_model.predict(img)

    predictions = predictions.flatten().tolist()

    return JSONResponse(content={'predictions': predictions})