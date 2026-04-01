from libraries import *
from utils import *


###################
# HYPERPARAMETERS #
###################


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
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

#print(model.summary())



############
# TRAINING #
############

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

scores = model.evaluate(test_ds)
print(f"\n\naccuracy: {scores[1]*100}%")
print(f"loss: {scores[0]}")



###################
# LEARNING CURVES #
###################

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig("plots/learning_curves.png")
print("\n\nLearning curves saved to plots/learning_curves.png")



###############
# PREDICTIONS #
###############

plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    plt.figure(figsize=(15, 15))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class, confidence = predict(model, class_names, images[i].numpy())
        actual_class = class_names[labels[i]] 
        
        plt.title(f"actual: {actual_class},\n predicted: {predicted_class}.\n confidence: {confidence}%")
        
        plt.axis("off")

    plt.savefig("plots/predictions.png")
print("\n\nPredictions grid example saved to plots/predictions.png")



##############
# MODEL SAVE #
##############

model.save("../models/v1")
