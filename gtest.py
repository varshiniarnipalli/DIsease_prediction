import tensorflow as tf
import numpy as np
import os

# --- 1. Define Constants ---
# This must be the same image size used during training.
IMG_SIZE = (256, 256)

# --- 2. Prediction Function ---
def predict_single_image(model, image_path, class_names):
    """
    Loads an image, preprocesses it, and predicts its class using the loaded model.

    Args:
        model: The loaded Keras model.
        image_path (str): The path to the image file.
        class_names (list): A list of class names in the correct order.

    Returns:
        tuple: A tuple containing the predicted class name and the confidence score.
    """
    print(f"--- Predicting image: {os.path.basename(image_path)} ---")
    
    # Load and preprocess the image
    # The image is loaded into a PIL format and resized.
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    
    # Convert the PIL image to a NumPy array.
    img_array = tf.keras.utils.img_to_array(img)
    
    # Normalize the image array to the [0, 1] range.
    img_array = img_array / 255.0
    
    # Create a batch by adding an extra dimension.
    # The model expects a batch of images, so (256, 256, 3) becomes (1, 256, 256, 3).
    img_array = tf.expand_dims(img_array, 0)

    # Make the prediction
    predictions = model.predict(img_array)
    
    # Apply softmax to get confidence scores.
    score = tf.nn.softmax(predictions[0])
    
    # Get the predicted class name and confidence
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    return predicted_class, confidence

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    # --- TO-DO: UPDATE THESE THREE VARIABLES ---
    
    # 1. Path to your saved Keras model file.
    MODEL_PATH = 'plant_disease_model.keras'
    
    # 2. Path to the single image you want to test.
    # Example: r"C:\Users\YourUser\Desktop\test_images\apple_scab.JPG"
    TEST_IMAGE_PATH = r"C:\Users\ASUS\OneDrive\画像\Screenshots\Screenshot 2025-09-16 205331.png" 
    
    # 3. List of your class names.
    # IMPORTANT: The order MUST be the same as the folders read during training.
    # You can find the order from the output when you first ran the training script.
    CLASS_NAMES = ['coffee___healthy', 'coffee___rust', 'coffee__phoma', 'corn_blight', 'corn_common_rust', 'corn_healthy', 'cotton_bacterial_blight', 'cotton_curl_virus', 'cotton_healthy_leaf', 'cotton_herbicide_growth_damage', 'cotton_leaf_hopper_jassids', 'cotton_leaf_redding', 'cotton_leaf_variegation', 'potato___early_blight', 'potato___healthy', 'potato___late_blight', 'rice_bacterialblight', 'rice_brown_spot', 'rice_healthy', 'rice_leafsmut']
    # -------------------------------------------

    # Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
    # Check if the test image file exists
    elif not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Test image not found at '{TEST_IMAGE_PATH}'")
    else:
        # Load the trained model
        print(f"Loading model from '{MODEL_PATH}'...")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            
            # Make a prediction
            predicted_class, confidence = predict_single_image(model, TEST_IMAGE_PATH, CLASS_NAMES)
            
            # Print the results
            print("\n--- Prediction Result ---")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2f}%")
            print("-------------------------\n")

        except Exception as e:
            print(f"An error occurred while loading the model or predicting: {e}")