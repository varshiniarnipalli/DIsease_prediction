import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ================================
# 1. Dataset Setup
# ================================
# Assuming your dataset has folders like:
# datasets/
# ├── train/
# │   ├── Apple___Black_rot/
# │   ├── Apple___healthy/
# │   └── Tomato___Late_blight/
# └── val/
#     ├── Apple___Black_rot/
#     ├── Apple___healthy/
#     └── Tomato___Late_blight/

train_dir = "C:/Users/ASUS/OneDrive/Desktop/ML/datasets_split/Train"
val_dir = "C:/Users/ASUS/OneDrive/Desktop/ML/datasets_split/val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

num_classes = len(train_gen.class_indices)
print("Classes:", train_gen.class_indices)

# ================================
# 2. Model Architecture
# ================================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")   # Output layer
])

model.summary()

# ================================
# 3. Compile the Model
# ================================
model.compile(optimizer="adam", 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

# ================================
# 4. Train the Model
# ================================
EPOCHS = 10
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ================================
# 5. Save the Trained Model
# ================================
model.save("Model.hdf5")
print("✅ Model saved as Model.hdf5")
