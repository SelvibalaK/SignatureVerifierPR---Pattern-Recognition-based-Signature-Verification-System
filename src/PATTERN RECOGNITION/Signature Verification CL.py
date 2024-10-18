# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:00:20 2024

@author: Selvibala
"""

import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

# Define data paths
forged_path ="C:\\Users\\Selvibala\\Downloads\\Data Signature Final\\dataset1\\forge"

real_path = "C:\\Users\\Selvibala\\Downloads\\Data Signature Final\\dataset1\\real"

# Load images (error handling for missing directories)
forged_images = []
real_images = []
try:
    for filename in os.listdir(forged_path):
        forged_images.append(cv2.imread(os.path.join(forged_path, filename)))
    for filename in os.listdir(real_path):
        real_images.append(cv2.imread(os.path.join(real_path, filename)))
except FileNotFoundError:
    print("Error: Data directories 'forged' or 'real' not found. Please ensure they exist.")
    exit()

# Preprocess images (resize, grayscale conversion if needed)
image_size = (224, 224)  # Adjust based on your dataset's typical size
preprocessed_images = []
for image in forged_images + real_images:
    resized_image = cv2.resize(image, image_size)
    # Convert to grayscale if necessary (e.g., grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY))
    preprocessed_images.append(resized_image)

# Create labels (0 for forged, 1 for real)
labels = [0] * len(forged_images) + [1] * len(real_images)

# Convert images and labels to NumPy arrays for training
X = np.array(preprocessed_images)
y = np.array(labels)

# Data augmentation (optional, but can improve performance)
datagen = ImageDataGenerator(rotation_range=20, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model (adjust architecture or hyperparameters as needed)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=image_size + (3,)))  # Adjust for grayscale if converted earlier
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))  # Sigmoid for binary classification

# Compile the model
model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"])

# Train the model with data augmentation (if enabled)
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

# Save the model (optional)
model.save("signature_model.h5")

# Function to predict on a new image
def predict_signature(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, image_size)
    # Convert to grayscale if necessary (same as preprocessing)
    image_array = np.expand_dims(resized_image, axis=0)  # Add dimension for model input
    prediction = model.predict(image_array)
    if prediction[0][0] > 0.5:
        return "Genuine"
    else:
        return "Forged (Predicted with lower confidence)"  # Indicate less certainty for forgeries

new_image_path = "path/to/new_image.jpg"
prediction = predict_signature(new_image_path)  # Call the function and assign the return value
print(f"Predicted Label for '{new_image_path}': {prediction}")


# Function to generate random noise image (limited functionality)
def generate_noise_image(size=(224, 224), intensity=0.2):
    noise = np.random.rand(size[0], size[1], 3) * intensity  # Adjust intensity for noise level
    return (noise * 255).astype(np.uint8)  # Convert to uint8 for image format

# Example usage for random noise image prediction
noise_image = generate_noise_image()
noise_image_path = "noise_image.jpg"  # Optional: Save the noise image for visualization
cv2.imwrite(noise_image_path, noise_image)  # Optional: Save the noise image

prediction = predict_signature(noise_image_path)
print(f"Predicted Label for '{noise_image_path}': {prediction}")
