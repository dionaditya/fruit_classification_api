from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from joblib import load
import pickle

app = Flask(__name__)


# Load the trained KNN model
loaded_model = load('./models/knn_model.joblib')
# Load the model from the file
with open('./models/extractor.pkl', 'rb') as file:
    model = pickle.load(file)

data_transform = transforms.Compose([
        transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

classes = ['Apple', 'Banana', 'avocado', 'cherry', 'kiwi', 'mango', 'orange', 'pinenapple', 'strawberries', 'watermelon']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        img = Image.open(file).convert('RGB')
        img_transformed = data_transform(img).unsqueeze(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            out = model(img_transformed.to(device))
            features = out.cpu().numpy()
            prediction = loaded_model.predict(features)
            predicted_class = classes[prediction[0]]

            return jsonify({'predicted_class': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)