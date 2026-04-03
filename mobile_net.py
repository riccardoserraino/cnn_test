import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import shutil
import random
import itertools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.metrics import categorical_crossentropy
from IPython.display import Image




######mobilenet import
mobile = tf.keras.applications.mobilenet.MobileNet()

###### functions
def prepare_image(file):
    img_path = 'other/'
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

def show_image(img, seconds=3):
    plt.imshow(img)
    plt.axis("off")
    plt.show(block=False)
    plt.pause(seconds)
    plt.close()

show_image(image.load_img('other/Butterfly_1.jpg'))
#####e.g. usage

preprocessed_image = prepare_image('Butterfly_1.jpg')
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
print(results)



