from libraries import *
from nn_utils import *
from plot_utils import *


##############
# PARAMETERS #
##############


IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 20



#################
# PREPROCESSING #
#################

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "datasets",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)
class_names = dataset.class_names

'''
print(f"\n\nClasses found: {class_names}")
print(f"\n\nDataset size: {len(dataset)}\n(Note it is approx. n° of elements in our datasets/batches)")

plt.figure(figsize=(10, 10))
for image_batch, labels_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[labels_batch[i]])
        plt.axis("off")
plt.tight_layout()
plt.savefig("plots/samples.png")
print("\n\nSamples grid saved to plots/samples.png")
'''


# Split dataset into train, validation and test sets
train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

'''
print(f"\n\nTraining:   {len(train_ds)}")
print(f"Validation: {len(val_ds)}")
print(f"Testing:    {len(test_ds)}")
'''



#############
# CNN MODEL #
#############

# Data Normalization 
resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.Rescaling(1./255),
])


# Model architecture
n_classes = 3

model = models.Sequential([
    tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),    
    resize_and_rescale,
    layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3), # To avoid overfitting we drop 20% of the neurons in the fully connected layer during training 
    layers.Dense(n_classes, activation='softmax'),
])

# print(model.summary())



############
# TRAINING #
############

# Data Augmentation
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2),
  layers.RandomContrast(0.2),
  layers.RandomBrightness(0.2),
  layers.RandomTranslation(0.2, 0.2),
  layers.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
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
#


##############
# MODEL SAVE #
##############

model.save("models/v4.keras")
