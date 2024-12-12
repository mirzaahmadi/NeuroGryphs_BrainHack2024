""" 
This script will take in the filtered EEG data and use it to train a classifier
"""
import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import re

TEST_SIZE = 30 #70/30 split training/testing

def main():
    # Checks command line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python brain_wave_classifier.py 'dataset.csv' \n Incorrect number of arguments")
        
    if not re.search(r".+\.csv$" ,sys.argv[1]):
        sys.exit("Usage: python brain_wave_classifier.py 'dataset.csv' \n Please make sure the dataset is in .CSV format")
        
    features, targets = load_data(sys.argv[1]) #features = brain_wave variables, targets = activity
    
    #Tweak the below hyperparameters for chosen model
    model = SVC()
    
    # Perform cross-validation and get multiple metrics
    cv_results = cross_validate(model, features, targets, cv=5, 
                                scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'], 
                                return_train_score=False)
    #Calculates accuracy metrics
    print("Accuracy scores:", cv_results['test_accuracy'])
    print("Mean accuracy:", cv_results['test_accuracy'].mean())

    #Calculates precision metrics
    print("Precision scores:", cv_results['test_precision_macro'])
    print("Mean precision:", cv_results['test_precision_macro'].mean())

    #Calculates the recall_macro scores
    print("Recall scores:", cv_results['test_recall_macro'])
    print("Mean recall:", cv_results['test_recall_macro'].mean())

    #Calculates the test_f1_macro scores
    print("F1 scores:", cv_results['test_f1_macro'])
    print("Mean F1 score:", cv_results['test_f1_macro'].mean())
    
    #Obtain the confusion matrix
    predicted_targets = cross_val_predict(model, features, targets, cv=5)
    conf_matrix = confusion_matrix(targets, predicted_targets)
    print("Confusion Matrix:\n", conf_matrix)

    
def load_data(filename):
    "Load data from CSV and split into different lists - one contained the features and one just containing the targets"
    
    #Initialize feature and target list
    feature_list = []
    target_list = []
    
    #Open and read the training data file
    with open (filename, 'r') as file:
        csv_reader = csv.reader(file) #make the file a csv_reader object
        row = next(csv_reader) #The 'next' skips the header row
        
        for row in csv_reader:
            feature_variables = row[0:0] #column names that hold features 
            target_variables = row[0] #column names that hold targets
            feature_list.append(feature_variables) 
            target_list.append(target_variables)
            
    return feature_list, target_list          
    
if __name__ == "__main__":
    main()