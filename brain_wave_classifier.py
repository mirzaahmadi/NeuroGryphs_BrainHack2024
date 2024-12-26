""" 
This script will take in the FILTERED EEG data and use it to train a RandomForest classifier (classifies into either "resting' or 'imagining faces')
"""
import pickle #This package is used to save the trained classifier as a pickle object
import csv
import sys
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import re
import pandas as pd

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
    
    #Initiate model
    model = RandomForestClassifier(n_estimators=100)
    
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
    
if __name__ == "__main__":
    main()