from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import io
from base64 import b64encode, b64decode
from pydantic import BaseModel


app = FastAPI()
obj_detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
fresh_model = tf.keras.models.load_model("mymodelv1.3.h5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","https://ml-fruit-freshness-classifier.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preproc(img):
    img = img / 255.0
    img = tf.image.resize(img, [128, 128])
    return img

def score_frame(frame):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    obj_detection_model.to(device)
    frame = [frame]
    results = obj_detection_model(frame)
    labels = results.xyxyn[0][:, -1].numpy()
    cord = results.xyxyn[0][:, :-1].numpy()
    return labels, cord

def plot_boxes(results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        # If score is less than 0.2 we avoid making a prediction.
        if row[4] < 0.2: 
            continue
        x1 = int(row[0]*x_shape)
        y1 = int(row[1]*y_shape)
        x2 = int(row[2]*x_shape)
        y2 = int(row[3]*y_shape)
        
        img = frame[y1:y2, x1:x2]
        img = preproc(img)
        img = tf.expand_dims(img, axis=0)
        predictions = fresh_model.predict(img)
        predictions = predictions.flatten().tolist()
        
        pred_label = 'rotten' if predictions[0] > 0.5 else 'fresh'
        bgr = (0, 0, 255) if predictions[0] > 0.5 else (0, 255, 0)
        print(pred_label, row)
        
        # classes = model.names # Get the name of label index
        label_font = cv2.FONT_HERSHEY_SIMPLEX #Font for the label.
        cv2.rectangle(frame, \
                      (x1, y1), (x2, y2), \
                       bgr, 2) #Plot the boxes
        cv2.putText(frame,\
                    pred_label, \
                    (x1, y1), \
                    label_font, 0.9, bgr, 2) #Put a label over box.
        return frame

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

@app.post("/stream")
async def upload_frame(i: FrameData):

    x = i.frame.replace('data:image/jpeg;base64,', '')
    x = io.BytesIO(b64decode(x))
    img = Image.open(x)
    img = np.array(img)
    results = score_frame(img)
    res = plot_boxes(results, img)

    try:
        image = Image.fromarray(res)
        image_stream = io.BytesIO()
        image.save(image_stream, format='JPEG')
        image_stream.seek(0)
        image_bytes = image_stream.getvalue()
        return JSONResponse(content={'frame': 'data:image/jpeg;base64,' + b64encode(image_bytes).decode('ascii')})
    except:
        return JSONResponse(content={'frame': i.frame})