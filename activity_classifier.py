# Command line takes in training dataset 
# We want to seperate the data into features and labels = create my own load data function
    # LOAD_DATA: This will then convert all the feature columns (however many at the beginning) into and keep it in list (training) and then the last one(presumably the label) it will be the last column adn we plut that into testing data set list
""" POTENTIAL
# Pass those two lists into other seperate function
    # One will convert all features to floats, and the other will convert all labels to numeric form as well
"""
# Then, those two are fed back into the main function and the data is split into 2 and loaded into training and testing lists that are of the correct format

# After using sci kit learn package, we can convert all that data into random training and testing categories - 70/30 for instance - and those will be used in the model

# train the model on the x_testing values and x and y training values (the features and labels for the training dataset)
    # Within this training, I will pick the algorithm I want to use using SCIKITLEARN, fit the evidence and the labels onto that and return the model - this will be our trained model which we can use in our dataset
    #FROM HERE - we can simply plug in our testing datasets and gauge the accuracy
    
TEST_SIZE = 30
    
import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def main():
    #Command line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python plan_classifier.py 'dataset'")
        
    features, targets = load_data(sys.argv[1])
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size = TEST_SIZE)
    
def load_data(filename):
    "Load data from CSV and split into different lists"
    
    feature_list = []
    target_list = []
    
    with open (filename, 'r') as file:
        csv_reader = csv.reader(file)
        row = next(csv_reader)
        
        for row in csv_reader:
            feature_variables = row[0:0]
            target_variables = row[0]
            feature_list.append(feature_variables)
            target_list.append(target_variables)
            
    return feature_list, target_list
            
    
    
    
if __name__ == "__main__":
    main()