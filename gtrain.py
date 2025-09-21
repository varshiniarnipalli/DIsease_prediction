import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. Load the Dataset ---
# Set the path to your main dataset directory, which contains 'train', 'val', and 'test' subfolders.
# IMPORTANT: Replace this with the actual path.
DATA_DIR = r"C:\Users\ASUS\OneDrive\Desktop\ML\datasets_split"

# Define paths for each split
train_dir = os.path.join(DATA_DIR, 'Train')
val_dir = os.path.join(DATA_DIR, 'val')
test_dir = os.path.join(DATA_DIR, 'Test')

# Check if the directories exist
if not all([os.path.exists(p) for p in [train_dir, val_dir, test_dir]]):
    print(f"Error: Make sure the '{DATA_DIR}' directory contains 'train', 'val', and 'test' subdirectories.")
    exit()

# --- Parameters ---
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 15

# Load the datasets directly from their respective directories
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False # No need to shuffle validation data
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False # No need to shuffle test data
)

# Get class names from the training dataset
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")
print(f"Training batches: {len(train_ds)}, Validation batches: {len(val_ds)}, Test batches: {len(test_ds)}")

# --- 2. Normalize Images and Use Data Augmentation ---

# Create a data augmentation layer
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# Use AUTOTUNE for performance optimization
AUTOTUNE = tf.data.AUTOTUNE

def prepare(ds, augment=False):
    # Normalize pixel values to [0, 1]
    rescale = layers.Rescaling(1./255)
    ds = ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)

    # Apply data augmentation only to the training set
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    # Use buffered prefetching
    return ds.prefetch(buffer_size=AUTOTUNE)

train_ds = prepare(train_ds, augment=True)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)


# --- 3. Build the CNN Model (Transfer Learning) ---

# Load the pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(256, 256, 3),
    include_top=False,  # Don't include the final classification layer
    weights='imagenet'
)

# Freeze the base model layers so they are not updated during training
base_model.trainable = False

# Build the final model
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax') # Output layer
])

model.summary()

# --- 4. Compile and Train the Model ---

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy', # Use categorical_crossentropy for one-hot labels
    metrics=['accuracy']
)

print("\nStarting model training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)
print("Model training finished.")


# --- 5. Plot Training & Validation Accuracy/Loss Graphs ---

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.show()

plot_history(history)


# --- 6. Evaluate the Model on the Test Set ---

print("\nEvaluating model on the test set...")
loss, accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")


# --- 7. Save the Trained Model ---

model_filename = 'plant_disease_model.keras'
model.save(model_filename)
print(f"\nModel saved successfully as {model_filename}")


# --- 8. Function to Predict the Class of a Single Image ---

def predict_single_image(model, image_path, class_names):
    """Loads an image, preprocesses it, and predicts its class."""  
    if not os.path.exists(image_path):
        print(f"Error: Image path not found at '{image_path}'")
        return None

    # Load and preprocess the image
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    # Get the predicted class name and confidence
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    return predicted_class, confidence

# --- Example Usage of the Prediction Function ---
# IMPORTANT: Replace this with a path to an actual image from your test folder.
test_image_path = r"C:\Users\ASUS\OneDrive\画像\Screenshots\Screenshot 2025-09-16 205331.png" 

if os.path.exists(test_image_path):
    predicted_class, confidence = predict_single_image(model, test_image_path, class_names)
    print(f"\nPrediction for image: {test_image_path}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
else:
    print(f"\nSkipping single image prediction because the test image path was not found: '{test_image_path}'")
    print("Please update the 'test_image_path' variable to test the prediction function.")