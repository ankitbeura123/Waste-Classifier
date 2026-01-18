# Waste Classifier – Dry and Wet Waste Classification

## Project Overview

The **Waste Classifier** is a machine learning–based web application that classifies waste images into **Dry Waste** and **Wet Waste** categories. The objective of this project is to promote proper waste segregation by using deep learning and computer vision techniques.

The system uses **transfer learning** with a pre-trained **MobileNetV2** model to extract features from waste images. Instead of training a model from scratch, the project leverages knowledge learned from the ImageNet dataset, resulting in faster training and better accuracy. The trained model is integrated into a web application that allows users to upload an image and receive real-time classification results.

---

## How to Run the Project Locally

### Prerequisites

- Python 3.8 or higher  
- pip (Python package manager)  
- Internet connection (for downloading pre-trained model weights)

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/ankitbeura123/Waste-Classifier.git
cd Waste-Classifier
```

---

### Step 2: Create and Activate a Virtual Environment (Recommended)

```bash
python -m venv venv
```

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

---

### Step 3: Install Dependencies

```bash
pip install -r requirement.txt
```

---

### Step 4: Dataset Structure

```
dataset/
├── train/
│   ├── dry/
│   └── wet/
└── test/
    ├── dry/
    └── wet/
```

---

### Step 5: Train the Model

```bash
python train_model.py
```

Model output:
```
model/dry_wet_model.h5
```

---

### Step 6: Run the Web Application

```bash
python app.py
```

Visit:
```
http://localhost:5000
```

---

## Project Structure

```
Waste-Classifier/
├── app.py
├── train_model.py
├── requirement.txt
├── dataset/
├── model/
├── templates/
└── static/
```

---

## Code Explanation

### train_model.py

- Uses MobileNetV2 with ImageNet weights
- Applies data augmentation
- Freezes base layers
- Trains custom classification layers
- Saves trained model for inference

### app.py

- Loads trained model
- Accepts image uploads
- Preprocesses images
- Displays predictions via web UI

---

## Workflow Summary

1. Prepare dataset
2. Train model
3. Save model
4. Run web app
5. Upload image
6. View prediction

---

## License

Educational use only.
