#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[2]:


import os
import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score 
from tensorflow.keras.models import load_model


# # Load the dataset

# In[3]:


def load_data(folder_path):
    images = []
    labels = []
    label_mapping = {"healthy": 0, "unhealthy": 1}

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path) and img_path.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(label_mapping[os.path.basename(folder_path)])

    return np.array(images), np.array(labels)

healthy_folder = "dataset/healthy"
unhealthy_folder = "dataset/unhealthy"

healthy_images, healthy_labels = load_data(healthy_folder)
unhealthy_images, unhealthy_labels = load_data(unhealthy_folder)


# # Dataset Splitting

# In[4]:


# Concatenate healthy and unhealthy data
all_images = np.concatenate((healthy_images, unhealthy_images), axis=0)
all_labels = np.concatenate((healthy_labels, unhealthy_labels), axis=0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    all_images, all_labels, test_size=0.3, random_state=42  # Fix: add the missing closing parenthesis here
)


# # Build CNN model for feature extraction

# In[5]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# # Normalize pixel values to be between 0 and 1

# In[6]:


X_train = X_train / 255.0
X_test = X_test / 255.0


# # Train the model

# In[7]:


model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.3)


# # Save the CNN model

# In[8]:


model.save("breast_cancer_cnn_model1.h5")


# # Use CNN for feature extraction

# In[9]:


X_train_features = model.predict(X_train)
X_test_features = model.predict(X_test)


# # SVM model Train

# In[10]:


svm_model = SVC(kernel="linear")
svm_model.fit(X_train_features, y_train)


# # Save the SVM model

# In[11]:


joblib.dump(svm_model, "breast_cancer_svm_modelNEW1.pkl")


# # SVM Model Test

# In[12]:


y_pred = svm_model.predict(X_test_features)


# # Evaluate the model

# In[13]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# # Plot the Confusion Matrix

# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Assuming y_test and y_pred are your true labels and predicted labels, respectively
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16})
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# # Classification report

# In[15]:


from sklearn.metrics import classification_report

# Print classification report
print(classification_report(y_test, y_pred))


# # Detection on New Data

# In[4]:


# Function to load and preprocess new data
def load_and_preprocess_new_data(folder_path):
    images = []
    original_filenames = []  # Store the original filenames

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path) and img_path.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            original_filenames.append(filename)  # Store the original filename

    return np.array(images), original_filenames

# Load the new data
new_data_folder = "test dataset"  # Replace with the path to your new data folder
new_data_images, original_filenames = load_and_preprocess_new_data(new_data_folder)

# Normalize pixel values
new_data_images = new_data_images / 255.0

# Load the CNN model
cnn_model = load_model("breast_cancer_cnn_model1.h5")

# Use CNN for feature extraction on new data
new_data_features = cnn_model.predict(new_data_images)

# Load the SVM model
svm_model = joblib.load("breast_cancer_svm_modelNEW1.pkl")

# Use SVM for classification on new data
new_data_predictions = svm_model.predict(new_data_features)

# Map numeric labels to class names
class_names = {0: "healthy", 1: "unhealthy"}
predicted_labels = [class_names[label] for label in new_data_predictions]

# Create a table with original filenames
output_table = {'Original Filename': original_filenames, 'Predicted Label': predicted_labels}
output_df = pd.DataFrame(output_table)

# Display the output table
print(output_df)


# In[5]:


# Function to load and preprocess new data
def load_and_preprocess_new_data(folder_path):
    images = []
    original_filenames = []  # Store the original filenames

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path) and img_path.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            original_filenames.append(filename)  # Store the original filename

    return np.array(images), original_filenames

# Load the new data
new_data_folder = "test dataset1"  # Replace with the path to your new data folder
new_data_images, original_filenames = load_and_preprocess_new_data(new_data_folder)

# Normalize pixel values
new_data_images = new_data_images / 255.0

# Load the CNN model
cnn_model = load_model("breast_cancer_cnn_model1.h5")

# Use CNN for feature extraction on new data
new_data_features = cnn_model.predict(new_data_images)

# Load the SVM model
svm_model = joblib.load("breast_cancer_svm_modelNEW1.pkl")

# Use SVM for classification on new data
new_data_predictions = svm_model.predict(new_data_features)

# Map numeric labels to class names
class_names = {0: "healthy", 1: "unhealthy"}
predicted_labels = [class_names[label] for label in new_data_predictions]

# Create a table with original filenames
output_table = {'Original Filename': original_filenames, 'Predicted Label': predicted_labels}
output_df = pd.DataFrame(output_table)

# Display the output table
print(output_df)


# In[ ]:




