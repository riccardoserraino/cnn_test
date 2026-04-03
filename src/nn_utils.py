from libraries import *

# Partitioning the dataset into train, validate and test splits
def get_dataset_partitions_tf(ds, train_split=0.70, val_split=0.15, test_split=0.15, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds
 
# Prediction output 
def predict(model, class_names, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    if confidence >= 85:
        return predicted_class, confidence
    else:
        return "UNCERTAIN", confidence

# Confusion Matrix 
def compute_confusion_matrix(model, test_ds):
    
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())
    
    return confusion_matrix(y_true, y_pred)

# Classification report 
def compute_classification_report(model, test_ds):
    
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())
    
    return classification_report(y_true, y_pred)

# Safe saturation
def safe_saturation(x):
    return tf.image.random_saturation(x, lower=0.75, upper=1.25)



