# Overview

This project is designed to classify dry beans based on their physical parameters. The classification process involves data preprocessing, model training, and evaluation using Support Vector Machines (SVM). The goal is to predict bean types from a dataset.


# Project Structure

The project is organized into three main Python files and a data file:
- preprocess.py: Contains all the necessary functions to preprocess the data. This includes loading the data, scaling features, splitting the dataset, and applying SMOTE to address class imbalance.
- model.py: Contains the training using SVM. This includes the training of the model and its evaluation on both training and validation sets to measure performance metrics like accuracy and F1-score.
- main.py: Entry point of the project. It calls the necessary functions to run the model.
- dry-beans-data.csv: The dataset file containing features and labels for dry bean classification.

# Running the Project
To run the project simply run the main.py file after installing the dependencies in requirements.txt
