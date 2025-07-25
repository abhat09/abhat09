from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from torchvision.models import resnet18
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load model
model = resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Regression for BMI
model_path = os.path.join(os.path.dirname(__file__), "../model/best_model_resnet.pth")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception:
        return jsonify({'error': 'Invalid image format'}), 400

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor).item()

    return jsonify({'predicted_bmi': round(prediction, 2)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)