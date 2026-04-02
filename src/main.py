from libraries import *
from utils import *
from keras.models import load_model
from PIL import Image

model = load_model("/home/giuliano-livi/Desktop/Master/FRE-2026/Agro_Insect_Vision/models/model_v1.keras")


img = Image.open("/home/giuliano-livi/Desktop/Master/FRE-2026/Agro_Insect_Vision/datasets/farfals1.jpg").convert("RGB")
img = np.array(img, dtype=np.float32)
# load_model si aspetta un batch, quindi aggiungi una dimensione
img = np.expand_dims(img, axis=0)  # shape: (1, 28, 28, 1)

prediction = model.predict(img)
predicted_class = np.argmax(prediction)
confidence = prediction[0][predicted_class] * 100  # confidence as percentage

class_names = ['Bee', 'Butterfly', 'Ladybug']
print(f"Classe predetta: {class_names[predicted_class]}")
print(f"Confidence: {confidence:.2f}%")
