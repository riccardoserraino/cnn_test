from libraries import *
from nn_utils import *

# Random image sample 
def plot_random_samples(dataset, class_names, n_images=12, save_path="plots/random_samples.png"):
    plt.figure(figsize=(8,8))
    images_collected = 0

    for image_batch, labels_batch in dataset.shuffle(1000).take(10):
        for i in range(len(image_batch)):
            if images_collected >= n_images:
                break

            plt.subplot(3, 4, images_collected + 1)

            img = image_batch[i].numpy().astype("uint8")
            label = labels_batch[i].numpy()

            plt.imshow(img)
            plt.title(class_names[label])
            plt.axis("off")

            images_collected += 1

        if images_collected >= n_images:
            break

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n\nSaved random samples to {save_path}")



# learning curves
def plot_learning_curves(history, epochs, save_path="plots/learning_curves.png"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), acc, label='Training Accuracy')
    plt.plot(range(epochs), val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), loss, label='Training Loss')
    plt.plot(range(epochs), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig(save_path)
    plt.close()

    print(f"\n\nLearning curves saved to {save_path}")


# prediction grid example
import matplotlib.pyplot as plt

def plot_predictions_grid(model, dataset, class_names, predict_fn, 
                          save_path="plots/predictions.png", grid_size=3):
    plt.figure(figsize=(15, 15))

    for images, labels in dataset.take(1):
        for i in range(grid_size * grid_size):
            ax = plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))

            predicted_class, confidence = predict_fn(
                model, class_names, images[i].numpy()
            )
            actual_class = class_names[labels[i]]

            if predicted_class != "UNCERTAIN":
                plt.title(
                    f"actual: {actual_class}\n"
                    f"pred: {predicted_class} ({confidence}%)"
                )
                print(f"Predicted: {predicted_class} ({confidence}%)")
            else:
                plt.title("UNCERTAIN")

            plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"\n\nPredictions grid example saved to {save_path}")


# Confusion matrix
def plot_confusion_matrix(cm, class_names):
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion Matrix saved to plots/confusion_matrix.png")


# Augmentation
def plot_augmentation_effect(dataset, data_augmentation, class_names, save_path="plots/augmentation.png"):

    # Get all images and labels from dataset
    all_images = []
    all_labels = []
    
    for images, labels in dataset:
        all_images.extend(images.numpy())
        all_labels.extend(labels.numpy())
    
    # Pick 3 random different images
    random_indices = random.sample(range(len(all_images)), 3)
    sample_images = [all_images[idx] for idx in random_indices]
    sample_labels = [all_labels[idx] for idx in random_indices]

    plt.figure(figsize=(10, 12))

    # Row 1: Original 1 and Augmented 1
    plt.subplot(3, 2, 1)
    plt.imshow(sample_images[0].astype("uint8"))
    plt.title(f"Original 1: {class_names[sample_labels[0]]}")
    plt.axis("off")

    augmented_1 = data_augmentation(tf.expand_dims(tf.convert_to_tensor(sample_images[0], dtype=tf.float32), 0), training=True)
    plt.subplot(3, 2, 2)
    plt.imshow(augmented_1[0].numpy().astype("uint8"))
    plt.title("Augmented 1")
    plt.axis("off")

    # Row 2: Original 2 and Augmented 2
    plt.subplot(3, 2, 3)
    plt.imshow(sample_images[1].astype("uint8"))
    plt.title(f"Original 2: {class_names[sample_labels[1]]}")
    plt.axis("off")

    augmented_2 = data_augmentation(tf.expand_dims(tf.convert_to_tensor(sample_images[1], dtype=tf.float32), 0), training=True)
    plt.subplot(3, 2, 4)
    plt.imshow(augmented_2[0].numpy().astype("uint8"))
    plt.title("Augmented 2")
    plt.axis("off")

    # Row 3: Original 3 and Augmented 3
    plt.subplot(3, 2, 5)
    plt.imshow(sample_images[2].astype("uint8"))
    plt.title(f"Original 3: {class_names[sample_labels[2]]}")
    plt.axis("off")

    augmented_3 = data_augmentation(tf.expand_dims(tf.convert_to_tensor(sample_images[2], dtype=tf.float32), 0), training=True)
    plt.subplot(3, 2, 6)
    plt.imshow(augmented_3[0].numpy().astype("uint8"))
    plt.title("Augmented 3")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n\nSaved augmentation comparison to {save_path}")