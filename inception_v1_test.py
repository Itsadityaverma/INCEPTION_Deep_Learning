import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained InceptionV1 model
model = load_model('tomato_leaf_inception_model.h5')

# Load class indices
with open('class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)

# Set the path of your test image folder
test_folder_path = '/Users/macpro/Documents/PLANT DETECTION/KAGGLE/test'

# Create a list to store images
images = []

# Load and preprocess each image in the folder
for filename in os.listdir(test_folder_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Update extensions accordingly
        img_path = os.path.join(test_folder_path, filename)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.
        images.append(img_array)

# Concatenate the images
images = np.concatenate(images, axis=0)

# Predict the categories of the input images
predictions = model.predict(images)
predicted_classes = np.argmax(predictions, axis=1)

# Get the class labels from class indices
class_labels = list(class_indices.keys())

# Print the results
for i, predicted_class in enumerate(predicted_classes):
    print(f"Image {i + 1}: Predicted class - {class_labels[predicted_class]}")
