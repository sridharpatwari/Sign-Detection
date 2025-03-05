import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Sequential
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Directory paths for input data
input_dir = "C:\\Users\\joshi\\Sign-Language-To-Text-Conversion\\Skeleton_Images"

# Preprocess images
def load_and_preprocess_images(image_folder, target_size=(128, 128)):
    images = []
    labels = []

    # Process folders from A to Z
    for folder_name in os.listdir(image_folder):
        folder_path = os.path.join(image_folder, folder_name)
        if os.path.isdir(folder_path):
            # Label for the folder (A=0, B=1, ..., Z=25)
            label = ord(folder_name) - ord('A')
            
            # Process all images in the folder
            for filename in os.listdir(folder_path):
                # Consider only image files
                if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
                    image_path = os.path.join(folder_path, filename)
                    
                    # Load image and convert to grayscale if not already
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                    image = cv2.resize(image, target_size)  # Resize the image

                    # Normalize pixel values to [0, 1]
                    image = image / 255.0
                    
                    # Append the processed image and its label
                    images.append(image)
                    labels.append(label)
    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Load and preprocess images from the dataset
images, labels = load_and_preprocess_images(input_dir)

# Split the dataset into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Check max value in y_train and y_val to ensure labels are between 0 and 25
print(f"Max label in y_train: {np.max(y_train)}")  # should be <= 25
print(f"Min label in y_train: {np.min(y_train)}")  # should be >= 0
print(f"Max label in y_val: {np.max(y_val)}")  # should be <= 25
print(f"Min label in y_val: {np.min(y_val)}")  # should be >= 0

# Clip the labels in case there are any out-of-bounds values
y_train = np.clip(y_train, 0, 25)
y_val = np.clip(y_val, 0, 25)

# One-hot encode the labels (since we are doing classification with multiple classes)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=26)  # 26 classes (A-Z)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=26)

# Build the CNN model
model = Sequential()

# Add Convolutional layers with ReLU activation and max-pooling
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output for the fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)))

model.add(layers.Dropout(0.5))  # Add more dropout after the dense layer
model.add(layers.Dense(26, activation='softmax'))  # 26 classes (A-Z)

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()

# Data augmentation for better generalization
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

datagen.fit(X_train)

# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=5,  # Change epochs as needed
                    validation_data=(X_val, y_val))

# Save the model
model.save("cnn9grps_rad1_model.h5")

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# Classification report and confusion matrix
# Convert predictions and labels from one-hot encoding to class labels
y_pred = model.predict(X_val)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_val, axis=1)

# Print unique classes
print("Unique classes in y_pred_labels:", np.unique(y_pred_labels))
print("Unique classes in y_true_labels:", np.unique(y_true_labels))

# Generate the classification report with available labels
labels = list(np.unique(y_true_labels))
print("Classification Report:")
print(classification_report(y_true_labels, y_pred_labels, labels=labels, target_names=[chr(i + 65) for i in labels]))


# Generate confusion matrix
conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[chr(i + 65) for i in range(26)], yticklabels=[chr(i + 65) for i in range(26)])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()