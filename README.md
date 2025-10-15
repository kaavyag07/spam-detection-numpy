📧 Spam Detection with Logistic Regression (NumPy Only)
🧠 Overview

This project implements a Spam Detection System using a Logistic Regression model built entirely from scratch with NumPy — without relying on any external machine learning libraries like scikit-learn or TensorFlow. The goal is to provide a clear understanding of the underlying mathematical concepts and implementation details behind logistic regression, including data preprocessing, feature extraction, and gradient-based optimization.

Spam detection is a classic binary classification problem, where the system must determine whether a given SMS message is “Spam” or “Not Spam.” By manually constructing the training pipeline — from tokenization to prediction — this project demonstrates how logistic regression can be implemented at the most fundamental level.

🚀 Features

End-to-end logistic regression implementation using NumPy only.

Bag-of-Words (BoW) model for converting text messages into numerical vectors.

Gradient Descent Optimization for updating model weights iteratively.

Binary cross-entropy loss function used for error minimization.

Performance metrics such as accuracy, precision, recall, and F1-score for evaluation.

Lightweight and transparent — ideal for educational and experimental purposes.

🧩 Workflow

Data Preprocessing:

Lowercasing, removing punctuation, and tokenizing text messages.

Creating a vocabulary from the dataset.

Converting each message into a feature vector using the Bag-of-Words approach.

Model Training:

Initializing model parameters (weights and bias).

Applying the sigmoid activation function to predict probabilities.

Updating weights using gradient descent to minimize the loss function.

Prediction & Evaluation:

Predicting class labels (Spam or Not Spam) for new SMS samples.

Evaluating the model using test data and accuracy metrics.

🛠 Technologies Used

Python 3

NumPy (for matrix operations and numerical computation)

No external ML libraries — all algorithms implemented manually.

📊 Results

The model achieves high accuracy on a small, balanced SMS dataset, demonstrating the effectiveness of logistic regression when properly trained and vectorized — even with minimal dependencies.
