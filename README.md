Waste Classifier – Dry vs Wet Waste
1. Project Overview

The Waste Classifier project is a machine learning–based web application that automatically classifies waste images into Dry Waste and Wet Waste categories. The goal of the project is to support proper waste segregation by using computer vision and deep learning techniques.

The system uses transfer learning with a pre-trained MobileNetV2 model to extract visual features from images. Instead of training a neural network from scratch, the project leverages knowledge learned from the ImageNet dataset and fine-tunes it for waste classification. This approach improves accuracy while reducing training time and computational cost.

The trained model is integrated into a web application where users can upload an image and instantly receive a prediction, making the solution practical and user-friendly.

2. How to Run the Project on a Local Machine
Prerequisites

Make sure the following are installed on your system:

Python 3.8 or higher

pip (Python package manager)

Git (optional)

A system with internet access (required to download MobileNetV2 weights)

Step-by-Step Setup
1. Clone the Repository
git clone https://github.com/ankitbeura123/Waste-Classifier.git
cd Waste-Classifier

2. Create a Virtual Environment (Recommended)
python -m venv venv


Activate it:

Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate

3. Install Required Dependencies
pip install -r requirement.txt

4. Prepare the Dataset

Organize the dataset in the following format:

dataset/
│
├── train/
│   ├── dry/
│   └── wet/
│
└── test/
    ├── dry/
    └── wet/


Each folder should contain images belonging to that class.

5. Train the Model
python train_model.py


After training, the model will be saved as:

model/dry_wet_model.h5

6. Run the Web Application
python app.py


Open your browser and visit:

http://localhost:5000

3. Project File Structure
Waste-Classifier/
│
├── app.py                  # Web application logic
├── train_model.py          # Model training script
├── requirement.txt         # Python dependencies
│
├── dataset/
│   ├── train/              # Training images
│   └── test/               # Testing images
│
├── model/
│   └── dry_wet_model.h5    # Trained model
│
├── templates/
│   └── index.html          # Web UI template
│
└── static/
    └── style.css           # Frontend styling

4. Code Explanation and File Interaction
4.1 train_model.py – Model Training Logic
Importing Required Libraries
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


What this does:

Imports TensorFlow and Keras tools required for deep learning.

Loads MobileNetV2, a lightweight CNN optimized for image classification.

How it works:

MobileNetV2 acts as a feature extractor.

Custom layers are added on top for waste classification.

Defining Training Parameters
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15


Purpose:

Ensures compatibility with MobileNetV2.

Controls memory usage and training duration.

Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)


What this does:

Artificially increases dataset size.

Improves model generalization.

How it works:

Random transformations prevent overfitting by exposing the model to varied image conditions.

Loading the Dataset
train_data = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)


How it works:

Automatically assigns labels based on folder names.

Converts images into batches for efficient training.

MobileNetV2 Base Model
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False


What this does:

Loads pre-trained weights.

Freezes the base layers to retain learned features.

Why this matters:

Reduces training time.

Prevents overfitting on a small dataset.

Custom Classification Head
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dense(2, activation="softmax")
])


How it works:

GlobalAveragePooling2D reduces feature maps.

Dense layers learn task-specific patterns.

Softmax outputs probabilities for dry and wet waste.

Model Training
model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    class_weight={0:1.0, 1:1.3}
)


Key detail:

Class weights help handle class imbalance by prioritizing wet waste samples.

Saving the Model
model.save("model/dry_wet_model.h5")


The trained model is stored and reused by the web application.

4.2 app.py – Model Inference and Web Interface

Loads the trained .h5 model.

Accepts image uploads from users.

Preprocesses images to match training format.

Predicts waste type using the trained model.

Displays results on the web page.

4.3 File Interaction Summary

train_model.py trains and saves the model.

app.py loads the saved model.

templates/ displays the UI.

static/ handles styling.

User uploads an image → model predicts → result displayed.
