# Logistic Regression Model for Tumor Prediction
This project uses logistic regression to predict whether a tumor is malignant or benign based on various features of cell nuclei obtained from digitized images of a fine needle aspirate (FNA) of a breast mass. The model achieves high accuracy by analyzing and tuning features from the Breast Cancer Wisconsin dataset.

# Project Overview
This notebook demonstrates:

- Loading and preprocessing data.
- Implementing logistic regression from scratch with L2 regularization (ridge regularization).
- Training, evaluating, and interpreting the logistic regression model.
- Achieving a high prediction accuracy on both training and test sets.

# Dataset
The dataset used is the Breast Cancer Wisconsin (Diagnostic) Dataset. This dataset is publicly available on Kaggle and includes 569 samples with 30 features. Each sample is labeled as either malignant (M) or benign (B), which are converted to 1 and 0 for the model, respectively.

# Key Features
Radius Mean: Mean of distances from center to points on the perimeter.
Texture Mean: Standard deviation of gray-scale values.
Smoothness Mean: Local variation in radius lengths.

# Model Implementation
The model uses logistic regression to classify each sample as either malignant or benign. It includes the following steps:
- Sigmoid Function: Used as the activation function to output probabilities between 0 and 1.
- Cost Function: Calculates the binary cross-entropy (log loss), with an additional term for L2 regularization to penalize large weights.
- Gradient Descent: Updates weights and bias using gradient descent, accounting for L2 regularization.
- Regularization: L2 regularization is applied to reduce overfitting by penalizing high weight values.

# Model Performance
Training Accuracy: 93.7%
Test Accuracy: 94.7%
The high accuracy on both training and test sets suggests the model generalizes well to new data.

# Project Structure
logistic_regression_tumor_prediction.ipynb: The main notebook containing the full code for data loading, preprocessing, model implementation, training, and evaluation.
README.md: Project documentation (this file).
data.csv: data from the Breast Cancer Wisconsin (Diagnostic) Data Set

# Getting Started
To run this project, you need the following setup:

Google Colab or a local environment with Jupyter Notebook.
Python 3.7+ and the following packages:
- numpy
- pandas
- scikit-learn (if you use it for data preprocessing)

# Running the Notebook on Colab
Upload the notebook to Google Colab.
If you use a dataset from Google Drive, mount the drive and set the path to the dataset in the notebook.
Run the cells sequentially to train and evaluate the model.

# Results
The model achieved the following results:
Training Accuracy: 93.7%
Testing Accuracy: 94.7%

# Future Improvements
Experiment with additional features or feature engineering.
Implement cross-validation to fine-tune the regularization parameter.
Explore other classification algorithms, such as SVM or decision trees, to compare performance.

# License
This project is licensed under the MIT License.

