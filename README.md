# Scene Style Classification

🏆 **Achievement:** 2nd Place out of 37 Teams!

**Competition Link:** [Kaggle - NN 26 Scene Style Classification](https://www.kaggle.com/competitions/nn-26-scene-style-classification)

## Project Overview
This project focuses on classifying scene styles using various machine learning techniques. The aim is to develop a model that can accurately predict the style of a given scene using a dataset of labeled images.

This repository contains the code and resources for a Neural Network image classification project aimed at identifying 17 distinct interior design styles from images. It features a custom Vision Transformer (ViT) implementation based on the CLIP architecture, alongside evaluations of several standard Convolutional Neural Networks (CNNs).

## Classes
The models classify images into one of the following 17 interior design styles:
`asian`, `boho`, `coastal`, `contemporary`, `craftsman`, `eclectic`, `farmhouse`, `french-country`, `industrial`, `mediterranean`, `minimalist`, `modern`, `scandinavian`, `shabby-chic-style`, `southwestern`, `tropical`, `victorian`

## Project Structure

- **`scene-style-classification.ipynb`**: The main Jupyter Notebook containing Exploratory Data Analysis (EDA), model training, evaluation, and visualizations.
- **`Report_SC_5.pdf`**: The detailed project report documenting the methodology, architectures, experiments, and results.
- **`Test Script/`**: A directory containing scripts to run inferences using pre-trained models on new datasets.
  - `Test Script.py`: The main execution script that loads models, processes test images, calculates accuracy (if labels are present), and generates CSV submission files.
  - `CLIP_VIT_Implementation.py`: Contains the custom TensorFlow/Keras implementation of the CLIP-based Vision Transformer, including `CLIPAttention`, `CLIPMLP`, and `CLIPEncoderLayer`.
  - `Script_Helpers.py`: Provides utility functions for image loading, preprocessing (e.g., custom Caffe preprocessing), data augmentation (`RandomSaturation`, `PlanckianJitter`), and normalization.
  - `Submissions/`: Directory where the predicted CSV files are saved.

## Models Evaluated

The project evaluates and compares various architectures:
- Custom Vision Transformer (ViT) L14
- Custom Vision Transformer (ViT) Patch32
- ResNet
- EfficientNetB0
- EfficientNetB3
- InceptionNet

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Pillow (PIL)

## How to use the Test Script

1. Place your pre-trained `.keras` model files in a `Models` directory located one level above the `Test Script` directory (e.g., `../Models/`).
2. Update the `TEST_DATA_DIR` path inside `Test Script/Test Script.py` to point to your image dataset.
3. Run the script:
   ```bash
   cd "Test Script"
   python "Test Script.py"
   ```
4. Predictions will be saved as `.csv` files in the `Test Script/Submissions/` folder.

## Conclusion
This project demonstrates a method for classifying scene styles and provides valuable insights into the effectiveness of different models in this domain.
