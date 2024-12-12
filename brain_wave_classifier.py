""" 
This script will take in the filtered EEG data and use it to train a classifier
"""
import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import re

TEST_SIZE = 30 #70/30 split training/testing

def main():
    #Command line arguments
    if len(sys.argv) != 2:                                
        sys.exit("Usage: python brain_wave_classifier.py 'dataset.csv'")
        
    features, targets = load_data(sys.argv[1]) #features = brain_wave variables, targets = activity
    
    model = SVC()
    
    #Using the feature target pairs we can cross validate
    accuracy_scores = cross_val_score(model, features, targets, cv=5, scoring='accuracy', random_state=27) 
    
    #Calculates accuracy metrics
    print("Accuracy for each fold:", accuracy_scores)
    print("Mean accuracy:", accuracy_scores.mean())
    
    #Calculates precision metrics
    precision_scores = cross_val_score(model, features, targets, cv=5, scoring = 'precision_macro', random_state=27)
    print("Precision for each fold:", precision_scores)
    print("Mean accuracy:", precision_scores.mean())
    
    #Calculates the recall_macro scores
    recall_scores = cross_val_score(model, features, targets, cv=5, scoring='recall_macro', random_state=27)
    print("Recall for each fold:", recall_scores)
    print("Mean recall:", recall_scores.mean())
    
    
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