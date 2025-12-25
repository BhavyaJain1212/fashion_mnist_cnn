# app.py
from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

app = Flask(__name__)

class MyNN(nn.Module):
    def __init__(self, input_features):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_features, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

model = MyNN(input_features=1)

model.load_state_dict(torch.load("cnn_best.pth", map_location="cpu"))
model.eval()

labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# --------------------
def preprocess_pixels(pixel_str: str) -> torch.Tensor:
    values = []
    pixels = []
    
    for p in pixel_str.split(','):
        p.strip()
        values.append(p)

    for p in values:
        pixels.append(float(p))
    
    # print(f'values: {values}')
    # print(f'pixels: {pixels}')

    if len(pixels) != 784:
        raise ValueError(f"Input must contain exactly 784 pixel values (28 * 28)")
    
    # to numpy and reshape to (1, 28, 28) => (C, H, W)
    image = np.array(pixels, dtype=np.float32).reshape(1, 28, 28)

    # to tensor
    x = torch.from_numpy(image) / 255.0 # scaling between [0,1]
    x = x.unsqueeze(0) # shape to (1, 1, 28, 28) => (B, C, H, W)

    return x


# Routes

@app.route("/", methods=["GET", "POST"]) # whenever there is an http get or post request, the function 
                                             # home will be calles
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            pixel_str = request.form.get("pixels", "")
            x = preprocess_pixels(pixel_str)

            with torch.no_grad():
                output = model(x)
                pred = torch.argmax(output, dim=1).item() # gives index
                prediction = labels[pred]

        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)
