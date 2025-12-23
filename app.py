# app.py
from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import torch.nn.functional as F

app = Flask(__name__)

# --------------------
# Model: VGG16 (must match training)
# --------------------
model = models.vgg16(weights=None)  # avoids downloading weights in deployment

model.classifier = nn.Sequential(
    nn.Linear(25088, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 10)
)

model.load_state_dict(torch.load("vgg16_best.pth", map_location="cpu"))
model.eval()

labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# --------------------
# Preprocess: pixels -> VGG input tensor
# Matches your dataset steps:
# reshape(28,28) -> uint8 -> stack 3 channels -> tensor -> resize to 224x224
# --------------------
def preprocess_pixels(pixel_str: str) -> torch.Tensor:
    pixels = [float(p.strip()) for p in pixel_str.split(",")]
    if len(pixels) != 784:
        print(len(pixels))
        raise ValueError("Input must contain exactly 784 pixel values (28x28).")

    # (28, 28)
    image = np.array(pixels, dtype=np.float32).reshape(28, 28)

    # uint8
    image = image.astype(np.uint8)

    # grayscale -> RGB (28, 28, 3)
    image = np.stack([image] * 3, axis=-1)

    # to torch tensor: (1, 3, 28, 28) in [0,1]
    x = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    x = x.unsqueeze(0)

    # VGG expects 224x224 for classifier input = 25088
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

    return x

# --------------------
# Routes
# --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            pixel_str = request.form.get("pixels", "")
            x = preprocess_pixels(pixel_str)

            with torch.no_grad():
                output = model(x)
                pred = torch.argmax(output, dim=1).item()
                prediction = labels[pred]

        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, error=error)


if __name__ == "__main__":
    app.run(debug=True)
