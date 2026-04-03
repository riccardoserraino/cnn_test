from libraries import *
from nn_utils import *
from plot_utils import *

IMAGE_SIZE = 256
BATCH_SIZE = 32

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "datasets",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)
class_names = dataset.class_names

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

data_augmentation = tf.keras.Sequential([
    # Occlusion
    layers.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),

    # Geometrical
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(
        height_factor=(-0.2, 0.2), 
        width_factor=(-0.2, 0.2)        
    ),
    layers.RandomTranslation(0.15, 0.15),

    # Color & Lighting
    layers.RandomBrightness(0.3),
    layers.RandomContrast(0.3),
    layers.RandomHue(0.05),
    layers.Lambda(lambda x: tf.map_fn(safe_saturation, x)),

    # Noise
    layers.GaussianNoise(0.02),
])

plot_augmentation_effect(
    train_ds,
    data_augmentation,
    class_names,
    save_path="plots/augmentation_test.png",
)

