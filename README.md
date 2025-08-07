# Breast Cancer Prediction with a Neural Network

This project details the development of a neural network model to predict whether a breast tumor is malignant or benign based on diagnostic medical measurements. The model was trained on the Breast Cancer Wisconsin (Diagnostic) Dataset from Kaggle. After a comprehensive hyperparameter tuning process, the final model achieved an F1-score of **0.988** on the test set, demonstrating exceptional accuracy and reliability.

This repository contains two versions of the project:
1.  A baseline **Logistic Regression** model.
2.  An improved and fine-tuned **Neural Network** model built with TensorFlow.

---

## Table of Contents
- [Project Goal](#project-goal)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Using the Pre-trained Model](#using-the-pre-trained-model)

---

## Project Goal
The objective was to build and evaluate a robust machine learning model capable of accurately classifying breast tumors from a set of 30 diagnostic features. The primary focus was on maximizing **recall** to ensure that the model could correctly identify as many malignant cases as possible, minimizing the risk of false negatives.

---

## Methodology

The project followed a standard machine learning workflow:

1.  **Data Exploration (EDA):** The dataset was analyzed to understand feature distributions, correlations, and the balance between the target classes (Malignant vs. Benign).

2.  **Data Preprocessing:**
    * The categorical target variable (`diagnosis`) was encoded into numerical format (0 for Benign, 1 for Malignant).
    * Features were scaled using `StandardScaler` to normalize their ranges, which is crucial for neural network performance.
    * The data was split into an 80% training set and a 20% testing set, using stratification to maintain the original class distribution in both sets.

3.  **Model Architecture and Tuning:**
    * A sequential neural network was designed with an input layer, two hidden layers using the `ReLU` activation function (16 and 8 neurons, respectively), and a final output layer.
    * The output layer uses a single neuron with a `Sigmoid` activation function to produce a probability score between 0 and 1.
    * A systematic hyperparameter tuning process was conducted, experimenting with different learning rates, network architectures (wider and simpler), and activation functions (Leaky ReLU). The initial architecture with a learning rate of 0.001 was confirmed to be the optimal configuration.

4.  **Training and Evaluation:**
    * The model was trained using the **Adam optimizer** and **Binary Cross-Entropy** loss function.
    * Performance was evaluated on the unseen test set using key classification metrics, including a detailed classification report and a confusion matrix.

---

## Model Performance

The final, fine-tuned model achieved the following performance on the test set:

* **Accuracy:** 99%
* **Precision (Malignant):** 1.00
* **Recall (Malignant):** 0.98
* **F1-Score (Malignant):** 0.988

### Confusion Matrix
The confusion matrix below visualizes the model's predictions on the test set, highlighting its strong ability to correctly identify both benign and malignant cases while making very few errors.

<img width="1006" height="788" alt="image" src="https://github.com/user-attachments/assets/925e4263-35c8-4884-a75d-4875b861cb9b" />


---

## Installation

To set up the environment and run this project locally, please follow these steps. This project was developed using Python 3.12.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/VinayRatnam/logistic-regression-tumor-prediction.git](https://github.com/VinayRatnam/logistic-regression-tumor-prediction.git)
    cd logistic-regression-tumor-prediction
    ```

2.  **Install the required packages:**
    This command will install all the necessary libraries listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage
The primary code for this project is contained within the Jupyter Notebook located in the `neural_network_model/` directory. To use it, simply launch Jupyter Notebook and open the file.

```bash
jupyter notebook
```

You can then run the cells sequentially to load the data, preprocess it, train the model, and view the evaluation results. The trained model weights can be saved and loaded as shown in the notebook.

---

## Using the Pre-trained Model
The trained model weights are saved, allowing you to use the model for inference without retraining it. Ensure you have a fitted `StandardScaler` from the training process to preprocess any new data before making predictions.

```bash
from tensorflow import keras
import numpy as np

# Load the trained model
model = keras.models.load_model('breast_cancer_model.keras')

# Example: Make a prediction on a new data sample
# NOTE: new_data must be a 2D array with 30 features and scaled with the same scaler used for training.
# new_data_scaled = scaler.transform(new_data) 
# prediction_proba = model.predict(new_data_scaled)
# prediction_class = (prediction_proba > 0.5).astype("int32")

# print(f"Predicted Class: {'Malignant' if prediction_class[0] == 1 else 'Benign'}")
```
