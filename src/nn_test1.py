from libraries import *
from utils import *

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 15
directory = "/home/giuliano-livi/Desktop/Master/FRE-2026/cnn_test/datasets/"

# Loading all the dataset in a tensoerflow dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(directory, 
                                                    shuffle=True, 
                                                    batch_size=BATCH_SIZE, 
                                                    image_size=(IMAGE_SIZE, IMAGE_SIZE))

# Folder names are the class names
class_names = dataset.class_names
print(class_names)

# Let's look at the shape of the images and labels
for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())

'''# Plot some images
plt.figure(figsize=(10, 10))
for images, labels in dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()
'''


#############################
#    DATA PREPROCESSING     #
#############################


# Let's split the dataset into training, validation and test sets
# 80% ==> training, 10% ==> validation, 10% ==> test
train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

# Let's look at the number of batches in each dataset
print(f"Number of batches in the training dataset: {len(train_ds)}")
print(f"Number of batches in the validation dataset: {len(val_ds)}")
print(f"Number of batches in the test dataset: {len(test_ds)}")

# To improve the performance of the dataset, we can use the cache
train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# We can create a layer to resize and rescale the images
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1./255),
])

# Now we can use data augmentation to improve the performance of the model
# (We slightly rotate the images, flip them horizontally and vertically, and zoom in and out)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])



#############################
#     MODEL DEFINITION      #
#############################


input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3
# Model definition
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

model.summary()


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

model.save("/home/giuliano-livi/Desktop/Master/FRE-2026/cnn_test/models/model_v1.keras")
