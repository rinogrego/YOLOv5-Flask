from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO, checks, hub
from dotenv import load_dotenv

from utils import draw_bb, draw_original

import os

app = Flask(__name__)

load_dotenv()
checks()
hub.login(os.environ["ULTRALYTICS_API"])

# Load the YOLO model
MODEL_PATH = os.path.join("model", "yolov5su-coco128.onnx")
pretrained_model = YOLO(MODEL_PATH, task="detect")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    try:
        image = cv2.cvtColor(cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"})

    try:
        results = pretrained_model(image, conf=0.25, iou=0.5)
    except Exception as e:
        return jsonify({"error": f"Error during inference: {str(e)}"})

    # Process results list
    boxes = []
    for result in results:
        boxes.append({
            "confidences": result.boxes.conf.numpy().astype(np.float64).tolist(),
            "classes": [str(result.names[int(idx)]) for idx in result.boxes.cls],
            "xywhn": result.boxes.xywhn.numpy().astype(np.float64).tolist() # because somehow np.float64 works while np.float32 didn't... Ref: https://github.com/numpy/numpy/issues/18994#issue-889585211
        })
        
    boxes_res = []
    for cf, cl, xywhn in zip(boxes[0]["confidences"], boxes[0]["classes"], boxes[0]["xywhn"]):
        boxes_res.append((cf, cl, xywhn))
    
    print(boxes_res)
    image_bb_encoded = draw_bb(image, file.filename, results)
    image_bb_html = f'data:image/png;base64,{image_bb_encoded}'
    
    preprocess = results[0].speed["preprocess"]
    inference = results[0].speed["inference"]
    postprocess = results[0].speed["postprocess"]
    inference_method = "Local"
    
    # process original image
    image_orig_encoded = draw_original(image)
    image_orig_html = f'data:image/png;base64,{image_orig_encoded}'
    
    return render_template(
        "index.html",
        image_orig_html=image_orig_html, 
        image_bb_html=image_bb_html, 
        boxes_res=boxes_res,
        preprocess=preprocess,
        inference=inference,
        postprocess=postprocess,
        inference_method=inference_method
    )
    # return jsonify({"boxes": boxes})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    try:
        image = cv2.cvtColor(cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"})

    try:
        results = pretrained_model(image, conf=0.25, iou=0.5)
    except Exception as e:
        return jsonify({"error": f"Error during inference: {str(e)}"})

    # Process results list
    boxes = []
    for result in results:
        boxes.append({
            "confidences": result.boxes.conf.numpy().astype(np.float64).tolist(),
            "classes": [str(result.names[int(idx)]) for idx in result.boxes.cls],
            "xywhn": result.boxes.xywhn.numpy().astype(np.float64).tolist() # because somehow np.float64 works while np.float32 didn't... Ref: https://github.com/numpy/numpy/issues/18994#issue-889585211
        })
    
    print(boxes)

    return jsonify({"boxes": boxes})

if __name__ == '__main__':
    app.run(debug=True)
