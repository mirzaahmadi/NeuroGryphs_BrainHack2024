""" 
This script will take in the FILTERED EEG data and use it to train a RandomForest classifier (classifies into either "resting' or 'imagining faces')
"""
import pickle  # This package is used to save the trained classifier as a pickle object
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, cross_val_predict, train_test_split, learning_curve
from sklearn.metrics import confusion_matrix, roc_auc_score, RocCurveDisplay
import os

# These algorithms were also tried 
    #from sklearn.svm import SVC
    #from sklearn.neighbors import KNeighborsClassifier
    #from sklearn.linear_model import LogisticRegression
    #from sklearn.naive_bayes import GaussianNB
 
 
def main():
    # Checks command line arguments to ensure arguments are correctly formatted
    if len(sys.argv) != 2:
        sys.exit("Usage: python brain_wave_classifier.py 'dataset.csv' \n Incorrect number of arguments")
        
    if not re.search(r".+\.csv$" ,sys.argv[1]):
        sys.exit("Usage: python brain_wave_classifier.py 'dataset.csv' \n Please make sure the dataset is in .CSV format")
        
    features, targets = load_data(sys.argv[1]) #features = brain_wave_variables, targets = resting/looking at images
    
    # Split data for ROC-AUC curve
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    # Initiate model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Perform cross-validation and get multiple metrics - accuracy, precision, etc.
    cv_results = cross_validate(model, features, targets, cv=5, 
                                scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'], 
                                return_train_score=False)
    
    print("")
    print("TRAINING DATA - ACCURACY METRICS")
    print("Accuracy scores:", cv_results['test_accuracy'])
    print("Mean accuracy:", cv_results['test_accuracy'].mean())

    print("Precision scores:", cv_results['test_precision_macro'])
    print("Mean precision:", cv_results['test_precision_macro'].mean())

    print("Recall scores:", cv_results['test_recall_macro'])
    print("Mean recall:", cv_results['test_recall_macro'].mean())

    print("F1 scores:", cv_results['test_f1_macro'])
    print("Mean F1 score:", cv_results['test_f1_macro'].mean())
    print('')
    
    #Obtain the confusion matrix
    predicted_targets = cross_val_predict(model, features, targets, cv=5)
    conf_matrix = confusion_matrix(targets, predicted_targets)
    print("Confusion Matrix:\n", conf_matrix)
    
    # Plot ROC-AUC curve
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {roc_auc:.2f}")

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC-AUC Curve")
    folder_path3 = './Visualizations/Curves'
    os.makedirs(folder_path3, exist_ok=True)
    plt.savefig(os.path.join(folder_path3, 'ROC-AUC_Curve.png'), format='png', dpi=300)

    # Plot learning curve
    plot_learning_curve(model, features, targets, cv=5)
    
    # Fit the model on features and targets, and save it as a 'pickle' once fully trained
    trained_mod = model.fit(features, targets)
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(trained_mod, f)


def load_data(filename):
    """
    Load and split data from CSV - One list for features and another for targets
    """
    # Load data into a Pandas DataFrame and shuffle it
    dataframe_data = pd.read_csv(filename)  # Use pandas to read the CSV directly
    df_randomized = dataframe_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Separate features and targets
    features = df_randomized.iloc[:, 0:5].values.tolist()  # First 5 columns are features
    targets = df_randomized.iloc[:, 5].tolist()  # 6th column is the target

    # Convert target values to numerical
    converted_target_list = []
    for target_value in targets:
        if target_value == "Rest":
            converted_target_list.append(0.0)
        elif target_value == "Face":
            converted_target_list.append(1.0)
        else:
            raise ValueError(f"Unexpected target value: {target_value}")

    # Convert feature values to floats
    converted_feature_list = [[float(item) for item in feature] for feature in features]

    return converted_feature_list, converted_target_list


def plot_learning_curve(estimator, X, y, cv):
    """Plots the learning curve."""
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label="Training score", color="r")
    plt.plot(train_sizes, test_scores_mean, label="Cross-validation score", color="g")

    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid()
    
    #Save plots to folder
    folder_path3 = './Visualizations/Curves'
    plt.savefig(os.path.join(folder_path3, 'learning_curve.png'), format='png', dpi=300)
    
if __name__ == "__main__":
    main()