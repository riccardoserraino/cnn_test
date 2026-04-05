from libraries import *
from cnn_utils import *


# Initialize
class_names = ['Bee', 'Butterfly', 'Ladybug']

# Load Model
model = load_model("models/v5.keras")

# Image Processing
img = Image.open("other/Bee_1.jpg").convert("RGB")
img = img.resize((256, 256)) # resize to match training
img = np.array(img, dtype=np.float32)
img = np.expand_dims(img, axis=0)  # (1, 256, 256, 3)

print("Input shape:", img.shape)

# Predict
predicted_class, confidence, is_confident = predict(model, class_names, img)
print(f'Is the model confident enough? {"Yes" if is_confident else "No"}')    
print(f"Predicted class: {predicted_class}\nConfidence: {confidence:.2f}%")
