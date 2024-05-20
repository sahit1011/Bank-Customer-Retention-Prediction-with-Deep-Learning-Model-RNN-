# Predicting Customer Churn in the Banking Sector Using Deep Learning

## Project Overview
This project aims to predict customer churn in the banking sector using a deep learning model. The dataset contains various features related to customer demographics and banking activities. The goal is to accurately predict whether a customer will exit the bank based on these features.

## Table of Contents
- Project Overview
- Dataset
- Workflow
- Dependencies
- Installation
- Usage
- Model Training
- Evaluation
- Results and Conclusion


## Dataset
- **Source**: The dataset is derived from a hypothetical banking scenario. You can find the csv file above.
- **Description**: The dataset includes customer information such as:
  - `CustomerId`: Unique identifier for each customer.
  - `Surname`: Customer's surname.
  - `CreditScore`: Customer's credit score.
  - `Geography`: Customer's country.
  - `Gender`: Customer's gender.
  - `Age`: Customer's age.
  - `Tenure`: Number of years the customer has been with the bank.
  - `Balance`: Customer's account balance.
  - `NumOfProducts`: Number of products the customer has.
  - `HasCrCard`: Whether the customer has a credit card.
  - `IsActiveMember`: Whether the customer is an active member.
  - `EstimatedSalary`: Customer's estimated salary.
  - `Exited`: Whether the customer exited the bank (target variable).

## Workflow
1. **Data Loading**: Load the dataset from the CSV file.
2. **Data Preprocessing**: Clean and preprocess the data, including handling missing values, normalization, and encoding categorical variables.
3. **Model Building**: Construct the deep learning model using appropriate layers and architecture.
4. **Model Training**: Train the model on the dataset.
5. **Model Evaluation**: Evaluate the model's performance using relevant metrics.
6. **Prediction**: Use the trained model to make predictions on new data.

## Dependencies
List of required libraries and dependencies:
- Python 3.x
- NumPy
- Pandas
- TensorFlow/Keras
- Scikit-learn
- Matplotlib

## Usage
- Clone the repository.
- Navigate to the project directory.
- Open the Jupyter Notebook.
- Run the cells in the notebook to execute the code for loading data, preprocessing, training the model, and evaluating the results.

## Model Training
Steps to train the model:
1. **Data Loading**: Load the CSV file into a Pandas DataFrame.
2. **Data Preprocessing**:
    - Handle missing values.
    - Normalize numerical features.
    - Encode categorical variables.
    - Split the data into training and testing sets.
3. **Model Building**: Define the architecture of the neural network, including input layer, hidden layers, and output layer.
4. **Model Training**: Compile the model, specify the loss function and optimizer, and train the model using the training data.
5. **Model Evaluation**: Evaluate the model's performance on the test data using relevant metrics.

## Evaluation
Model performance evaluation includes:
- Metrics used: Accuracy, precision.
- Validation techniques: Train-test split.
- Visualizations: Confusion matrix, ROC curve, etc.

## Results and Conclusion
Key findings and results:
- The deep learning model achieved an accuracy of 86% on the test set.
- Potential improvements for future work include Experimenting with different neural network architectures and hyperparameters, Using a larger dataset to improve model generalization, and Implementing additional feature engineering to extract more meaningful information from the data.





