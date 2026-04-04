from libraries import *
from cnn_utils import *
from plot_utils import *


##############
# PARAMETERS #
##############


IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 1

random_seed = 42



#################
# PREPROCESSING #
#################

# Set random seed for reproducibility
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

# Load datasets
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "datasets",
    seed=random_seed,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)
class_names = dataset.class_names

#print(f"\n\nClasses found: {class_names}")
#print(f"\n\nDataset size: {len(dataset)}\n(Note it is approx. n° of elements in our datasets/batches)")
#plot_random_samples(dataset, class_names)

# Split dataset into train, validation and test sets
train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)


#print(f"\n\nTraining:   {len(train_ds)}")
#print(f"Validation: {len(val_ds)}")
#print(f"Testing:    {len(test_ds)}")



#############
# CNN MODEL #
#############

# Data Normalization 
resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.Rescaling(1./255),
])


# Model architecture
# Conv2D -- feature extraction
# MaxPooling2D -- dimensionality reduction
# GlobalAveragePooling -- summarization of feature maps (vector out)
# Flatten -- flatten the output of prev. layer (vector out)
# Dense -- decision making
# Dropout -- regularization tech. to drop random neurons
#
# IDEA: Conv2D -> Conv2D -> MaxPooling2D 
#       Conv2D -> Conv2D -> MaxPooling2D 
#       Conv2D -> Conv2D -> MaxPooling2D
#       Flatten 
#       Dense 
#       Dropout 
#       Dense 

n_classes = 3

model = models.Sequential([
    tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),    
    resize_and_rescale,
    layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
    layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
    layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3), # To avoid overfitting we drop 30% of the neurons in the fully connected layer during training 
    layers.Dense(n_classes, activation='softmax'),
])

print(model.summary())



############
# TRAINING #
############

# Data Augmentation
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

# Apply data augmentation to training dataset
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)

# Compile and train the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=EPOCHS,
)

# Evaluate the model on test dataset
scores = model.evaluate(test_ds)
print(f"\n\naccuracy: {scores[1]*100}%")
print(f"loss: {scores[0]}")



##################
# LEARNING STATS #
##################

# Plot Learning Curves
plot_learning_curves(history, EPOCHS)

# Plot Prediction Grid
plot_predictions_grid(model, test_ds, class_names, predict)

# Confusion Matrix
cm = compute_confusion_matrix(model, test_ds)
plot_confusion_matrix(cm, class_names)

# Classification Report
cr = compute_classification_report(model, test_ds)
print(f"\n\nClassification Report:\n{cr}")

# Augmentation Visualization e.g. 3 random versions from train_ds
plot_augmentation_effect(
    train_ds,
    data_augmentation,
    class_names,
    save_path="plots/augmentation_test.png",
)


##############
# MODEL SAVE #
##############

model.save("models/v6.keras")
