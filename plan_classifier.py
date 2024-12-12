""" 
This script will take in the filtered EEG data and use it to train a classifier
"""
import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import re

TEST_SIZE = 30 #70/30 split training/testing

def main():
    #Command line arguments
    if len(sys.argv) != 2:                                
        sys.exit("Usage: python classifier.py 'dataset.csv'")
        
    features, targets = load_data(sys.argv[1]) #features = brain_wave variables, targets = activity
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size = TEST_SIZE)
    
    model = train_model(x_train, y_train) #Training the model based off of x_training data and y_training data
    predictions = model.predict(x_test) 
    no_tr, no_tm, no_ti = evaluate(y_test, predictions) #Extra accuracy values
    
    # Print # correct, # incorrect, true positive rate, 
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True rest Rate: {100 * no_tr:.2f}%")
    print(f"True math Rate: {100 * no_tm:.2f}%")
    print(f"True image Rate : {100 * no_ti:.2f}%")
    
    
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


def train_model(ftrs, trgts):  
    model = SVC() #Choose a machine learning algorithm
    model.fit(ftrs, trgts) # Input the ftrs and trgts into the model to train it
    
    return model

def evaluate(actual_targets, predicted_targets):
    # actual_targets = actual targets from the data
    # predicted_targets = targets the model predicted
    
    number_of_true_rests = 0
    number_of_false_rests = 0
    number_of_true_maths = 0
    number_of_false_maths = 0
    number_of_true_images = 0
    number_of_false_images = 0 
    
    for actual, predicted in zip(actual_targets, predicted_targets):
        """zip (actual_targets, predicted_targets) => makes them into pairs"""
        if actual == "x" and actual == predicted: #TODO: Change x
            number_of_true_rests += 1 
        elif actual == "x" and actual != predicted:
            number_of_false_rests += 1
        elif actual == "y" and actual == predicted: #TODO: Change y
            number_of_true_maths += 1
        elif actual == "y" and actual != predicted:
            number_of_false_maths += 1
        elif actual == "z" and actual == predicted: #TODO: Change z
            number_of_true_images += 1
        elif actual == "z" and actual != predicted: 
            number_of_true_images += 1
            
    
    no_true_rests = float(number_of_true_rests / (number_of_true_rests + number_of_false_rests))
    no_true_maths = float(number_of_true_maths / (number_of_true_maths + number_of_false_maths)) 
    no_true_images = float(number_of_true_images / (number_of_true_images + number_of_false_images))
    
    return no_true_rests, no_true_maths, no_true_images
          
    
if __name__ == "__main__":
    main()