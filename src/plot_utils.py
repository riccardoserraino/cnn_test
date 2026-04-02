from libraries import *
from nn_utils import *


# LEARNING CURVES
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


# PREDICTION GRID EXAMPLE
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


# CONFUSION MATRIX
def plot_confusion_matrix(cm, class_names):
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion Matrix saved to plots/confusion_matrix.png")

