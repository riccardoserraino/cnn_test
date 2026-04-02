from libraries import *
from nn_utils import *

# Initialize
class_names = ['Bee', 'Butterfly', 'Ladybug']

# Load Model
model = load_model("/models/v1.keras")

# Image processing 
img = Image.open("/jpg_image_to_test").convert("RGB")
img = np.array(img, dtype=np.float32)
img = np.expand_dims(img, axis=0)  # current shape to match load model constraints: (1, 28, 28, 1)

# Prediction
prediction, confidence = predict(model, class_names, img)
print(f"Classe predetta: {prediction}")
print(f"Confidence: {confidence:.2f}%")

