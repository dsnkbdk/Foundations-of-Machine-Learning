# Machine Learning Assignments

This repository contains three assignments focused on different machine learning algorithms: Decision Tree, Naive Bayes Classification, and Support Vector Machine (SVM). Each assignment involves implementing the respective algorithm, tuning hyperparameters, and evaluating the model's performance.

## Assignment 1: Decision Tree Learning

### Objective
This assignment aims to help you establish the capability of implementing your first machine learning model and the mindset of designing a proper evaluation procedure for your model.

### Datasets
The datasets used in this assignment include:
- Website Phishing
- Breast Cancer Prediction (BCP)
- Arrhythmia

### Key Algorithm: Decision Tree
A Decision Tree is a non-parametric supervised learning method used for classification and regression. It splits the dataset into subsets based on the value of input features, creating a tree-like model of decisions.

### Steps
1. **Importing Libraries**: Utilize libraries such as `numpy`, `pandas`, `matplotlib.pyplot`, `scipy.stats`, and `sklearn.tree`.
2. **Data Preparation**: Load datasets and split them into training and testing sets.
3. **Model Initialization**: Initialize the Decision Tree model using `DecisionTreeClassifier()`.
4. **Hyperparameter Tuning**: Use techniques like `RandomizedSearchCV` to find the best hyperparameters.
5. **Model Training**: Train the Decision Tree model on the training dataset.
6. **Model Evaluation**: Evaluate the model using cross-validation techniques and measure performance metrics like accuracy, precision, recall, and F1 score.

## Assignment 2: Naive Bayes Classification

### Objective
The task of this assignment is to implement an improved version of the Naive Bayes algorithm to predict the category of points of interest from the Yelp dataset - one of "Restaurants", "Shopping", and "Nightlife". Predictions are then submitted to Kaggle for evaluation.

### Key Algorithm: Naive Bayes
Naive Bayes is a probabilistic classifier based on Bayes' Theorem with an assumption of independence between every pair of features. It is particularly useful for text classification and categorical data.

### Steps
1. **Importing Libraries**: Utilize libraries such as `numpy`, `pandas`, and `sklearn`.
2. **Data Preparation**: Load the training and test datasets and split the training data into training and validation sets.
3. **Feature Extraction**: Extract features from text data using methods like TF-IDF or Count Vectorizer.
4. **Model Initialization**: Initialize the Naive Bayes model using `MultinomialNB()` or `GaussianNB()`.
5. **Model Training**: Train the Naive Bayes model on the training dataset.
6. **Model Evaluation**: Evaluate the model using cross-validation techniques and measure performance metrics like accuracy, precision, recall, and F1 score.
7. **Prediction and Submission**: Apply the trained model to the test set and submit predictions to Kaggle.

## Assignment 3: Support Vector Machines (SVMs)

### Objective
The tasks of this assignment are to implement the SVM algorithm on three different datasets, investigate the classification results with different kernels, and explore the impact of hyperparameters on model evaluation.

### Key Algorithm: Support Vector Machine (SVM)
Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification or regression. It works by finding the hyperplane that best divides a dataset into classes.

### Steps
1. **Importing Libraries**: Utilize libraries such as `numpy`, `pandas`, `matplotlib.pyplot`, and `sklearn.svm`.
2. **Data Preparation**: Generate or load datasets and split them into training and testing sets.
3. **Model Initialization**: Initialize the SVM model using `SVC()`.
4. **Hyperparameter Tuning**: Use techniques like `GridSearchCV` to find the best hyperparameters.
5. **Model Training**: Train the SVM model on the training dataset using different kernels (e.g., linear, polynomial, RBF).
6. **Model Evaluation**: Evaluate the model using cross-validation techniques and measure performance metrics like accuracy, precision, recall, and F1 score.
