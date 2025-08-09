import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import numpy as np
from collections import Counter

# === Flask Setup ===
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model/coconut_cnn.pth'
DISEASE_JSON_PATH = 'disease_info.json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Device Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Trained Model ===
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint['class_names']

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# === Image Transformations ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Normalize Disease JSON Keys ===
def normalize_key(name):
    return ''.join(e.lower() for e in name.strip() if e.isalnum())

# === Load disease details from JSON ===
with open(DISEASE_JSON_PATH, 'r') as f:
    raw_disease_details = json.load(f)

disease_details = {
    normalize_key(key): value for key, value in raw_disease_details.items()
}

# === Config ===
REGION_GRID = (2, 2)
CONFIDENCE_THRESHOLD = 0.4

# === Utility: Split image into subregions ===
def split_image_regions(image, grid=(2, 2)):
    width, height = image.size
    w_step = width // grid[0]
    h_step = height // grid[1]
    regions = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            left = i * w_step
            top = j * h_step
            right = left + w_step
            bottom = top + h_step
            region = image.crop((left, top, right, bottom))
            regions.append(region)
    return regions

@app.route('/')
def index():
    return render_template('index.html')

# === Image Prediction ===
@app.route('/predict', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return "❌ No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "❌ No file selected", 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    image = Image.open(image_path).convert('RGB')
    sub_images = split_image_regions(image, REGION_GRID)

    predictions = []
    for region in sub_images:
        image_tensor = transform(region).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, predicted = torch.max(probs, 1)
            if conf.item() >= CONFIDENCE_THRESHOLD:
                label = class_names[predicted.item()].strip()
                predictions.append(label)

    if not predictions:
        return "⚠️ No disease confidently detected in image.", 500

    unique_labels = list(set(predictions))
    diseases_info = []
    for label in unique_labels:
        norm_label = normalize_key(label)
        raw_details = disease_details.get(norm_label, {})
        diseases_info.append({
            "label": label,
            "details": {
                "explanation": raw_details.get("explanation", f"The image shows signs of '{label}'."),
                "water": raw_details.get("water", "N/A"),
                "fertilizer": raw_details.get("fertilizer", "N/A"),
                "medicine": raw_details.get("medicine", ["N/A"]),
                "organic_medicine": raw_details.get("organic_medicine", ["N/A"]),
                "prevention": raw_details.get("prevention", "N/A"),
            }
        })

    image_url = url_for('static', filename='uploads/' + filename)
    return render_template('index.html',
                           multi_predictions=diseases_info,
                           image_url=image_url)

# === Video Prediction ===
@app.route('/predict_video', methods=['POST'])
def predict_video():
    if 'video' not in request.files:
        return "❌ No video uploaded", 400

    file = request.files['video']
    if file.filename == '':
        return "❌ No video selected", 400

    filename = secure_filename(file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate) if frame_rate > 0 else 10

    all_predictions = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            if brightness < 40 or brightness > 220:
                frame_count += 1
                continue

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            sub_images = split_image_regions(pil_image, REGION_GRID)

            for region in sub_images:
                image_tensor = transform(region).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, predicted = torch.max(probs, 1)
                    if conf.item() >= CONFIDENCE_THRESHOLD:
                        label = class_names[predicted.item()].strip()
                        all_predictions.append(label)

        frame_count += 1

    cap.release()
    os.remove(video_path)

    if not all_predictions:
        return "⚠️ No diseases detected with confidence in video.", 500

    label_counter = Counter(all_predictions)
    most_common_labels = [label for label, count in label_counter.items() if count >= 2]
    if not most_common_labels:
        most_common_labels = list(label_counter.keys())

    diseases_info = []
    for label in most_common_labels:
        norm_label = normalize_key(label)
        raw_details = disease_details.get(norm_label, {})
        diseases_info.append({
            "label": label,
            "details": {
                "explanation": raw_details.get("explanation", f"The video shows signs of '{label}'."),
                "water": raw_details.get("water", "N/A"),
                "fertilizer": raw_details.get("fertilizer", "N/A"),
                "medicine": raw_details.get("medicine", ["N/A"]),
                "organic_medicine": raw_details.get("organic_medicine", ["N/A"]),
                "prevention": raw_details.get("prevention", "N/A"),
            }
        })

    return render_template('index.html',
                           multi_predictions=diseases_info,
                           image_url=None)

# === Run Server ===
if __name__ == '__main__':
    app.run(debug=True)
